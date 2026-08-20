"""Weak-calibration diagnostics + deterministic prior-shift correction.

Logistic recalibration model (Cox, 1958):

    P(y = 1 | p) = sigmoid(a + b * logit(p))

    b == 1  -> predictions have the right spread (weak calibration)
    b <  1  -> over-extreme / over-confident predictions
    a == 0  -> no systematic over/under-prediction

Two intercepts are reported:
  * ``intercept``             : a from the joint (a, b) fit.
  * ``intercept_fixed_slope`` : a with b constrained to 1 ("calibration-in-the-
                                large"), i.e. an offset-only GLM; the quantity
                                that isolates pure prevalence/bias shift.

Fits are statsmodels Binomial GLMs (IRLS = MLE). CIs are Wald by default,
percentile bootstrap on request.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.special import logit as _logit
from statsmodels.tools.sm_exceptions import PerfectSeparationError

EPS = 1e-6
MODES = ("joint", "fixed_slope", "both")
CI_METHODS = ("wald", "bootstrap")

_METRIC_KEYS = (
    "slope",
    "slope_ci_low",
    "slope_ci_high",
    "intercept",
    "intercept_ci_low",
    "intercept_ci_high",
    "intercept_fixed_slope",
    "intercept_fixed_slope_ci_low",
    "intercept_fixed_slope_ci_high",
)


def nan_calibration_result(n: int = 0, n_pos: int = 0, **extra: Any) -> dict[str, Any]:
    """All-NaN result with the full key set (for degenerate/failed fits)."""
    result: dict[str, Any] = {key: float("nan") for key in _METRIC_KEYS}
    result.update(
        n=int(n),
        n_pos=int(n_pos),
        prevalence=(float(n_pos) / n) if n else float("nan"),
        converged=False,
    )
    result.update(extra)
    return result


def _validate(y_true, y_pred_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred_prob {p.shape}")

    finite = np.isfinite(y) & np.isfinite(p)
    if not finite.all():
        warnings.warn(
            f"Dropping {int((~finite).sum())} non-finite row(s).", RuntimeWarning
        )
    y, p = y[finite], p[finite]

    if y.size and not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary (0/1)")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError("y_pred_prob must be probabilities in [0, 1]")
    return y, p


def _glm(endog: np.ndarray, exog: np.ndarray, offset: np.ndarray | None = None):
    """Binomial GLM MLE; returns None instead of raising on unfittable data."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.GLM(
                endog, exog, family=sm.families.Binomial(), offset=offset
            ).fit()
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(f"Logistic calibration fit failed: {exc}", RuntimeWarning)
        return None


def calibration_slope_intercept(
    y_true,
    y_pred_prob,
    *,
    mode: str = "both",
    alpha: float = 0.05,
    ci_method: str = "wald",
    n_boot: int = 1000,
    random_state: int | None = None,
    eps: float = EPS,
) -> dict[str, Any]:
    """Calibration slope and intercept with confidence intervals.

    Parameters
    ----------
    mode : {"joint", "fixed_slope", "both"}, default "both"
        "joint"       -> fit a and b together (slope + joint intercept).
        "fixed_slope" -> fit a only, with b fixed at 1 (calibration-in-the-large).
        "both"        -> fit both models (default).
    alpha : two-sided CI level (0.05 -> 95% CI).
    ci_method : "wald" (from the GLM covariance) or "bootstrap" (percentile).

    Returns
    -------
    dict with keys: slope, slope_ci_low/high, intercept, intercept_ci_low/high,
    intercept_fixed_slope, intercept_fixed_slope_ci_low/high, n, n_pos,
    prevalence, converged, mode, ci_method, alpha. Unrequested / unidentified
    quantities are NaN.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    if ci_method not in CI_METHODS:
        raise ValueError(f"ci_method must be one of {CI_METHODS}; got {ci_method!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")

    y, p = _validate(y_true, y_pred_prob)
    n, n_pos = int(y.size), int(y.sum())
    result = nan_calibration_result(
        n, n_pos, mode=mode, ci_method=ci_method, alpha=float(alpha)
    )
    if n < 2 or n_pos in (0, n):
        warnings.warn(
            "Degenerate input (n < 2 or a single observed class); returning NaN.",
            RuntimeWarning,
        )
        return result

    lp = _logit(np.clip(p, eps, 1.0 - eps))
    constant_preds = bool(np.ptp(lp) == 0.0)
    converged: list[bool] = []

    if mode in ("joint", "both"):
        if constant_preds:
            warnings.warn(
                "Constant predictions: the calibration slope is not identified.",
                RuntimeWarning,
            )
        else:
            res = _glm(y, np.column_stack([np.ones(n), lp]))
            if res is not None:
                ci = np.asarray(res.conf_int(alpha=alpha), dtype=float)
                result["intercept"] = float(res.params[0])
                result["slope"] = float(res.params[1])
                result["intercept_ci_low"] = float(ci[0, 0])
                result["intercept_ci_high"] = float(ci[0, 1])
                result["slope_ci_low"] = float(ci[1, 0])
                result["slope_ci_high"] = float(ci[1, 1])
                converged.append(bool(getattr(res, "converged", True)))

    if mode in ("fixed_slope", "both"):
        res = _glm(y, np.ones((n, 1)), offset=lp)
        if res is not None:
            ci = np.asarray(res.conf_int(alpha=alpha), dtype=float)
            result["intercept_fixed_slope"] = float(res.params[0])
            result["intercept_fixed_slope_ci_low"] = float(ci[0, 0])
            result["intercept_fixed_slope_ci_high"] = float(ci[0, 1])
            converged.append(bool(getattr(res, "converged", True)))

    result["converged"] = bool(converged) and all(converged)

    if ci_method == "bootstrap":
        result.update(
            _bootstrap_cis(
                y,
                lp,
                mode=mode,
                alpha=alpha,
                n_boot=n_boot,
                random_state=random_state,
            )
        )
    return result


def _bootstrap_cis(
    y: np.ndarray,
    lp: np.ndarray,
    *,
    mode: str,
    alpha: float,
    n_boot: int,
    random_state: int | None,
) -> dict[str, float]:
    """Percentile bootstrap CIs (useful when the slope CI is visibly skewed)."""
    rng = np.random.default_rng(random_state)
    n = int(y.size)
    draws: dict[str, list[float]] = {
        "slope": [],
        "intercept": [],
        "intercept_fixed_slope": [],
    }
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        yb, lb = y[idx], lp[idx]
        if yb.sum() in (0, yb.size):
            continue
        if mode in ("joint", "both") and np.ptp(lb) > 0.0:
            res = _glm(yb, np.column_stack([np.ones(n), lb]))
            if res is not None:
                draws["intercept"].append(float(res.params[0]))
                draws["slope"].append(float(res.params[1]))
        if mode in ("fixed_slope", "both"):
            res = _glm(yb, np.ones((n, 1)), offset=lb)
            if res is not None:
                draws["intercept_fixed_slope"].append(float(res.params[0]))

    lo_q, hi_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    out: dict[str, float] = {}
    for key, values in draws.items():
        if len(values) >= 20:
            lo = float(np.percentile(values, lo_q))
            hi = float(np.percentile(values, hi_q))
        else:
            lo = hi = float("nan")
        out[f"{key}_ci_low"], out[f"{key}_ci_high"] = lo, hi
    return out


def saerens_prior_correction(
    y_pred_prob, prior_train: float, prior_test: float, *, eps: float = 0.0
):
    """Deterministic prior (prevalence) shift correction, Saerens et al. (2002).

    Reweights posteriors for a *known* change in the class prior:

        p' = p * w1 / (p * w1 + (1 - p) * w0)
        w1 = prior_test / prior_train
        w0 = (1 - prior_test) / (1 - prior_train)

    Algebraically identical to
    ``logit(p') = logit(p) - logit(prior_train) + logit(prior_test)``, but
    evaluated in probability space as requested (no clipping needed: p in {0, 1}
    maps to itself). Saerens et al. additionally give an EM procedure that
    *estimates* ``prior_test`` from unlabelled target data; this is the
    closed-form correction for the case where the target prevalence is known.
    """
    p = np.asarray(y_pred_prob, dtype=float)
    for name, prior in (("prior_train", prior_train), ("prior_test", prior_test)):
        prior = float(prior)
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError(f"{name} must lie strictly in (0, 1); got {prior!r}")
    if p.size and (np.nanmin(p) < 0.0 or np.nanmax(p) > 1.0):
        raise ValueError("y_pred_prob must be probabilities in [0, 1]")
    if eps > 0.0:
        p = np.clip(p, eps, 1.0 - eps)

    w1 = float(prior_test) / float(prior_train)
    w0 = (1.0 - float(prior_test)) / (1.0 - float(prior_train))
    num = p * w1
    return num / (num + (1.0 - p) * w0)


if __name__ == "__main__":
    from scipy.special import expit

    rng = np.random.default_rng(0)
    z = rng.normal(size=200_000)
    y = rng.binomial(1, expit(z))

    for name, probs in (
        ("well calibrated   (expect b~1, a~0)", expit(z)),
        ("under-confident   (expect b~2)     ", expit(0.5 * z)),
        ("over-confident+up (expect b<1, a>0)", expit(2.0 * z + 1.0)),
    ):
        r = calibration_slope_intercept(y, probs)
        print(
            f"{name}: slope={r['slope']:.3f} "
            f"[{r['slope_ci_low']:.3f}, {r['slope_ci_high']:.3f}] "
            f"intercept={r['intercept']:+.3f} "
            f"[{r['intercept_ci_low']:+.3f}, {r['intercept_ci_high']:+.3f}] "
            f"intercept_b1={r['intercept_fixed_slope']:+.3f}"
        )

    # Prior shift: train prevalence 0.5, deployment prevalence 0.1
    shifted = saerens_prior_correction(expit(z), 0.5, 0.1)
    print("prior-corrected mean prob:", float(shifted.mean()))
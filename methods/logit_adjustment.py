"""
Group-conditional logit adjustment ("prior-corrected BCE") for prevalence shortcuts.

Summary
-------
Train with binary cross-entropy on *shifted* logits,

    L(x, y, g) = BCE( z(x) + delta_g , y ),
    delta_g    = tau * ( logit(pi_g) - logit(pi_ref) ),

where z(x) is the raw model logit, g is the sample's group (here: the Drain label),
pi_g = P(y = 1 | g) is the group's prevalence in the training distribution, and pi_ref
is a single reference prevalence shared by all groups. The offset is a fixed constant
per group; it is added *inside* the sigmoid. (Adding -log pi_g + log pi_ref to the loss
value instead would be a per-sample constant with zero gradient.)

At inference nothing is added: sigmoid(z(x)) is the model's score. It is meant to be the
posterior that every group would have if its prevalence were pi_ref. No group labels are
needed at test time.

Why this removes the prevalence shortcut
----------------------------------------
Assume a group affects the image only through a label-independent signature (e.g. drain
pixels, scanner look) and through the label prior:

    p(x | y, g) = p(x_c | y) * s(x_s | g),     P(y = 1 | g) = pi_g,

with x_c the disease evidence and x_s the group signature.

* Plain BCE is minimised at the pooled posterior,

      z_BCE(x) = log[p(x_c|1) / p(x_c|0)] + logit( sum_g P(g | x_s) * pi_g ),

  so the model gains by reading the group from x_s: the signature is worth as much as the
  prevalence difference between groups. This is the "maximum likelihood forces
  group-specific priors" mechanism.

* The adjusted loss has population minimiser

      z*(x) = log[p(x_c|1) / p(x_c|0)] + logit(pi_ref),

  for *any* P(g | x). Each group's conditional risk is stationary at z* on its own, because
  sigmoid(z* + delta_g) is exactly that group's Bayes posterior. So the optimum does not
  depend on x_s, and every group is calibrated at prevalence pi_ref.

This is the Saerens et al. (2002) prior-shift correction moved into the training loss: plain
BCE followed by the post-hoc map z -> z - delta_g has the same population optimum. Training
with the offset also aligns the gradients during training, not only the end point.
Importance weighting with w = pi_ref(y) / pi_g(y) targets the same minimiser at higher
variance (cf. the nuisance-randomised distribution of Puli et al., 2022). With
pi_ref = 0.5 this is the in-training counterpart of this repo's post-hoc prevalence
calibration (`calibration/core.py`), which re-balances each drain group to 50/50 before
fitting group-wise beta calibrators.

Relation to the score-matching losses (combined variants)
---------------------------------------------------------
`ScoreMatchingLoss` / `DatasetScoreMatchingLoss` penalise the spread across groups of
E[sigmoid(z) | y, g], separately for y = 0 and y = 1.

* Under the factorisation above, the adjusted-loss optimum z*(x) depends on x_c only, and
  p(x_c | y) is the same in every group. So E[sigmoid(z*) | y, g] is equal across groups and
  the score-matching penalty is zero there. The two objectives are compatible, and the
  combination only adds pressure where training departs from that optimum.
* Plain BCE violates the constraint by construction, so score matching also penalises the
  prevalence shortcut, but indirectly: through its symptom, not its cause.
* The score-matching term must be evaluated on the raw logits z, the deployed score. On the
  shifted logits z + delta_g the group means differ by construction, and the penalty would
  fight the adjustment. The combined classes below do this correctly.
* Where the assumption fails, i.e. p(x_c | y, g) depends on g (drained pneumothoraces are
  treated and often smaller), the adjusted-loss optimum legitimately depends on g and score
  matching will push against it.
* The two can also conflict when the head memorises the training set. Score matching is
  computed on *training* predictions. A memorising model fits the adjusted target on the
  training rows, so its raw training scores are shifted by -delta_g. That is a group
  pattern opposite to the shortcut, and it does not carry over to new data. Score matching
  reads it as group dependence and removes it through the only feature that generalises
  by group, the signature x_s, which re-creates part of the shortcut at test time.
  Synthetic check (frozen features with 1150 noise dimensions, MLP head, train AUROC
  >= 0.99, 20k-sample test sets). Mean test logit shift g1 - g0 within label:
    - plain BCE: +1.30
    - adjusted loss: +0.23
    - dataset score matching, lambda = 10: +0.87 (train-set group means equalised)
    - adjusted + dataset score matching, lambda = 10: +0.64
    - adjusted + dataset score matching, lambda = 1: +0.31
  The Optuna range for lambda in the combined variants therefore extends down to 1e-3, so
  the search can reach the standalone limit.

Known limitation
----------------
Additive logit offsets remove the *incentive* to use the shortcut at the population
optimum; they do not *penalise* it. Overparameterised networks trained into the
interpolating regime lose the effect of additive adjustments (Kini et al., 2021, who
recommend multiplicative / VS-loss adjustments there). Early stopping, weight decay and
frozen-backbone heads keep training away from that regime. An explicit output penalty
such as score matching keeps acting even when the network interpolates, which is one
reason to benchmark the combination.

Implementation notes
--------------------
* Group prevalences are estimated once from the *training* split. This happens in
  `CXPLightningModule.setup` via `fit_group_priors`, before the per-split clones are
  made, so the train, val and test loss use the same offsets. If the train loader uses a
  `WeightedRandomSampler` (`balance_train=true`), the prevalences are computed under the
  sampling weights, i.e. for the distribution the optimiser actually sees. With the repo's
  (y, drain)-balanced sampler both groups then have prevalence 0.5 and the offsets vanish
  when pi_ref = 0.5.
* Missing group labels (Drain = NaN; ~82 % of SIIM training rows) follow
  `logit_adjustment_missing_group`:
    - "own_group" (default): treat "missing" as an extra stratum with its own estimated
      prevalence. This removes the prevalence difference between labelled and unlabelled
      rows. It cannot remove the shortcut *within* the unlabelled rows, because their
      group is unknown.
    - "no_offset": plain BCE for those rows (treated as if they had prevalence pi_ref).
    - "exclude": drop them from the classification loss altogether.
* `logit_adjustment_reference_prior` is "uniform" (0.5, default), "pooled" (the
  training-split marginal prevalence) or a number in (0, 1). In a binary problem, pi_ref
  only shifts every logit by the same constant, so AUROC-type metrics, including the
  cross-group pairwise AUROCs, do not depend on it. It matters for Brier score and
  calibration-in-the-large.
* `logit_adjustment_tau` scales the offsets as in Menon et al. (2021). tau = 1 is the exact
  Bayes correction and tau = 0 recovers plain BCE. It is fixed unless
  `logit_adjustment_tune_tau=true`, in which case Optuna searches tau in [0, 2].
* The reported "bce"/"wbce" components are computed on the raw logits, exactly as in every
  other method, so they stay comparable across methods. The training objective itself is
  reported as "la_bce". Note that `select_chkpt_on=BCE` therefore selects on the
  *unadjusted* val BCE. That quantity penalises a model whose scores are calibrated at
  pi_ref instead of at the val split's group prevalences.

To deploy at a known prevalence pi_s instead of pi_ref, add logit(pi_s) - logit(pi_ref) to
the logit, i.e. sigmoid(z + logit(pi_s) - logit(pi_ref)).

References
----------
- M. Saerens, P. Latinne, C. Decaestecker (2002). Adjusting the outputs of a classifier to
  new a priori probabilities: a simple procedure. Neural Computation 14(1):21-41.
- A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, S. Kumar (2021). Long-tail
  learning via logit adjustment. ICLR. https://arxiv.org/abs/2007.07314
- J. Ren, C. Yu, S. Sheng, X. Ma, H. Zhao, S. Yi, H. Li (2020). Balanced meta-softmax for
  long-tailed visual recognition. NeurIPS.
- S. Liu, X. Zhang, N. Sekhar, Y. Wu, P. Singhal, C. Fernandez-Granda (2023). Avoiding
  spurious correlations via logit correction. ICLR. https://arxiv.org/abs/2212.01433
  (LC loss: CE(f(x) + log P(y, a)), the softmax form of this loss with pi_ref uniform and
  with the spurious attribute inferred instead of labelled.)
- G. R. Kini, O. Paraskevas, S. Oymak, C. Thrampoulidis (2021). Label-imbalanced and
  group-sensitive classification under overparameterization. NeurIPS.
  https://arxiv.org/abs/2103.01550
- A. Puli, L. H. Zhang, E. K. Oermann, R. Ranganath (2022). Out-of-distribution
  generalization in the presence of nuisance-induced spurious correlations. ICLR.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from .base import BaseMethod
from .score_matching import ScoreMatchingLoss
from .score_matching_dataset import DatasetScoreMatchingLoss
from config import ExperimentConfig

MISSING_GROUP_POLICIES = ("own_group", "no_offset", "exclude")

# Prevalences are clipped to [eps, 1 - eps] before taking logits. A group without any
# positives (or negatives) would otherwise get an infinite offset, and BCE-with-logits
# returns NaN gradients for infinite inputs. logit(1e-4) ~ -9.2.
PRIOR_EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(float(p), PRIOR_EPS), 1.0 - PRIOR_EPS)
    return math.log(p) - math.log1p(-p)


def _unwrap_subset(dataset: Dataset) -> tuple[Dataset, np.ndarray]:
    """Resolve (possibly nested) `Subset`s into (base dataset, absolute row indices)."""
    ds = dataset
    idx: Optional[np.ndarray] = None
    while isinstance(ds, Subset):
        parent_idx = np.asarray(ds.indices, dtype=np.int64)
        idx = parent_idx if idx is None else parent_idx[idx]
        ds = ds.dataset
    if idx is None:
        idx = np.arange(len(ds), dtype=np.int64)  # type: ignore[arg-type]
    return ds, idx


def _is_missing_group(groups: np.ndarray | torch.Tensor):
    """NaN or negative group ids mean "unknown", consistent with `ScoreMatchingLoss`."""
    if isinstance(groups, torch.Tensor):
        return torch.isnan(groups) | (groups < 0)
    with np.errstate(invalid="ignore"):
        return np.isnan(groups) | (groups < 0)


@dataclass(frozen=True)
class GroupPriors:
    """Positive-class prevalences of the training distribution, per group."""

    by_group: dict[int, float]  # P(y=1 | g) for every observed group id
    missing: Optional[float]  # P(y=1 | group unknown); None if no such rows
    pooled: float  # P(y=1) over all rows
    counts: dict[str, float]  # (weighted) row counts, for logging


def estimate_group_priors(
    labels: np.ndarray,
    groups: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> GroupPriors:
    """Estimate P(y=1 | g), P(y=1 | g missing) and P(y=1), optionally under sampling weights.

    `sample_weights` are the relative probabilities with which rows are drawn during
    training (e.g. the weights of a `WeightedRandomSampler`). They are normalised
    internally, so only their ratios matter.
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=np.float64).reshape(-1)
    if labels.shape != groups.shape:
        raise ValueError(f"labels {labels.shape} and groups {groups.shape} differ")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("Logit adjustment expects binary labels in {0, 1}.")
    if sample_weights is None:
        w = np.ones_like(labels)
    else:
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if w.shape != labels.shape:
            raise ValueError(
                f"sample_weights {w.shape} do not match labels {labels.shape}"
            )
        if (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("sample_weights must be finite, non-negative, not all 0.")

    missing = _is_missing_group(groups)
    known_ids = groups[~missing]
    if (known_ids != np.round(known_ids)).any():
        raise ValueError("Group ids must be integer-valued (or NaN for missing).")

    def prevalence(mask: np.ndarray) -> float:
        return float((w[mask] * labels[mask]).sum() / w[mask].sum())

    by_group: dict[int, float] = {}
    counts: dict[str, float] = {}
    for g in np.unique(known_ids).astype(np.int64).tolist():
        mask = (~missing) & (groups == g)
        if w[mask].sum() <= 0:
            continue
        by_group[int(g)] = prevalence(mask)
        counts[f"g{g}"] = float(w[mask].sum())

    missing_prior = None
    if missing.any() and w[missing].sum() > 0:
        missing_prior = prevalence(missing)
        counts["missing"] = float(w[missing].sum())

    return GroupPriors(
        by_group=by_group,
        missing=missing_prior,
        pooled=prevalence(np.ones_like(missing, dtype=bool)),
        counts=counts,
    )


def _resolve_reference_prior(spec, pooled: float) -> float:
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "uniform":
            return 0.5
        if key == "pooled":
            return pooled
        try:
            value = float(key)
        except ValueError as e:
            raise ValueError(
                "logit_adjustment_reference_prior must be 'uniform', 'pooled' or a "
                f"number in (0, 1); got {spec!r}"
            ) from e
    else:
        value = float(spec)
    if not 0.0 < value < 1.0:
        raise ValueError(f"Reference prior must lie in (0, 1); got {value}")
    return value


class GroupLogitAdjustedBCE(nn.Module):
    """BCE on z + delta_g with fixed per-group offsets delta_g = tau*(logit pi_g - logit pi_ref).

    Call `set_priors` (or `fit` via the method) before the first forward pass.
    """

    offset_table: torch.Tensor  # offsets indexed by group id; NaN = id unseen in training
    missing_offset: torch.Tensor  # scalar offset for rows without a group label

    def __init__(
        self,
        reference_prior="uniform",
        tau: float = 1.0,
        missing_group: str = "own_group",
    ):
        super().__init__()
        if missing_group not in MISSING_GROUP_POLICIES:
            raise ValueError(
                f"logit_adjustment_missing_group must be one of {MISSING_GROUP_POLICIES}; "
                f"got {missing_group!r}"
            )
        if tau < 0:
            raise ValueError(f"logit_adjustment_tau must be >= 0; got {tau}")
        self.reference_prior_spec = reference_prior
        self.tau = float(tau)
        self.missing_group = missing_group
        self.priors: Optional[GroupPriors] = None
        self.reference_prior: Optional[float] = None
        self.register_buffer("offset_table", torch.empty(0))
        self.register_buffer("missing_offset", torch.tensor(0.0))
        self._warned_missing_fallback = False

    @property
    def is_fitted(self) -> bool:
        return self.priors is not None

    def set_priors(self, priors: GroupPriors) -> None:
        if not priors.by_group:
            raise ValueError(
                "No labelled group found in the training data; logit adjustment needs "
                "at least one row with a known group."
            )
        self.priors = priors
        self.reference_prior = _resolve_reference_prior(
            self.reference_prior_spec, priors.pooled
        )
        ref_logit = _logit(self.reference_prior)

        max_id = max(priors.by_group)
        table = torch.full((max_id + 1,), float("nan"), dtype=torch.float32)
        for g, p in priors.by_group.items():
            table[g] = self.tau * (_logit(p) - ref_logit)
        self.offset_table = table.to(self.offset_table.device)

        miss = 0.0
        if self.missing_group == "own_group" and priors.missing is not None:
            miss = self.tau * (_logit(priors.missing) - ref_logit)
        self.missing_offset = torch.tensor(miss, device=self.missing_offset.device)

    def describe(self) -> str:
        if self.priors is None:
            return "GroupLogitAdjustedBCE(unfitted)"
        parts = [
            f"g{g}: pi={p:.4f}, offset={float(self.offset_table[g]):+.4f}"
            for g, p in sorted(self.priors.by_group.items())
        ]
        if self.priors.missing is not None:
            parts.append(
                f"missing: pi={self.priors.missing:.4f}, policy={self.missing_group}, "
                f"offset={float(self.missing_offset):+.4f}"
            )
        return (
            f"pi_ref={self.reference_prior:.4f} (spec={self.reference_prior_spec!r}), "
            f"tau={self.tau:g}, pooled pi={self.priors.pooled:.4f}; " + "; ".join(parts)
        )

    def offsets(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample offsets and a mask of rows that enter the loss."""
        if self.priors is None:
            raise RuntimeError(
                "Group priors have not been fitted. CXPLightningModule.setup() does this "
                "from the training split; outside Lightning call "
                "method.fit_group_priors(train_dataset) first."
            )
        g = groups.reshape(-1).to(device=self.offset_table.device, dtype=torch.float32)
        missing = _is_missing_group(g)
        known = ~missing

        out = torch.zeros_like(g)
        if known.any():
            ids = g[known]
            if (ids != torch.round(ids)).any():
                raise ValueError("Group ids must be integer-valued (or NaN for missing).")
            ids = ids.long()
            if (ids >= self.offset_table.numel()).any():
                bad = ids[ids >= self.offset_table.numel()].unique().tolist()
                raise ValueError(f"Group id(s) {bad} were not seen in the training split.")
            vals = self.offset_table[ids]
            if torch.isnan(vals).any():
                bad = ids[torch.isnan(vals)].unique().tolist()
                raise ValueError(f"Group id(s) {bad} were not seen in the training split.")
            out[known] = vals

        keep = torch.ones_like(missing)
        if missing.any():
            if self.missing_group == "exclude":
                keep = known
            elif self.missing_group == "own_group":
                if self.priors.missing is None and not self._warned_missing_fallback:
                    logging.warning(
                        "[logit_adjustment] Rows without a group label encountered, but "
                        "the training split had none, so their prevalence is unknown. "
                        "Using offset 0 (as for missing_group='no_offset')."
                    )
                    self._warned_missing_fallback = True
                out[missing] = self.missing_offset
            # "no_offset": leave at 0
        return out, keep

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        z = logits.reshape(-1).float()
        y = targets.reshape(-1).float()
        # The method object is not an nn.Module, so Lightning never moves this loss to
        # the training device; follow the logits instead.
        if self.offset_table.device != z.device:
            self.to(z.device)
        offsets, keep = self.offsets(groups)
        if not keep.any():
            # Keep the graph connected so the caller can still call backward().
            return z.sum() * 0.0
        adjusted = z + offsets
        return F.binary_cross_entropy_with_logits(
            adjusted[keep], y[keep], reduction="mean"
        )


def _three_layer_head(num_features: int) -> nn.Module:
    # Same head as standard / score matching, so results are comparable.
    return nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, 1))


class LogitAdjustmentMethod(BaseMethod):
    """Standalone group-conditional logit adjustment (method key: `logit_adjustment`)."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.la_loss = GroupLogitAdjustedBCE(
            reference_prior=getattr(config, "logit_adjustment_reference_prior", "uniform"),
            tau=getattr(config, "logit_adjustment_tau", 1.0),
            missing_group=getattr(config, "logit_adjustment_missing_group", "own_group"),
        )

    # ------------------------------------------------------------------ priors
    @property
    def group_priors_fitted(self) -> bool:
        return self.la_loss.is_fitted

    def fit_group_priors(
        self, dataset: Optional[Dataset], sample_weights: Optional[np.ndarray] = None
    ) -> None:
        """Estimate the training prevalences from a dataset exposing `.labels`/`.drain`.

        `dataset` may be wrapped in (nested) `Subset`s. `sample_weights`, if given, are
        aligned with the rows *as exposed by `dataset`* (e.g. sampler weights).
        """
        if dataset is None:
            raise RuntimeError(
                "logit_adjustment needs the training dataset to estimate group "
                "prevalences, but none was available."
            )
        base, idx = _unwrap_subset(dataset)
        if not (hasattr(base, "labels") and hasattr(base, "drain")):
            raise TypeError(
                f"{type(base).__name__} does not expose `labels` and `drain`; cannot "
                "estimate group prevalences."
            )
        labels = np.asarray(base.labels)[idx]  # type: ignore[attr-defined]
        groups = np.asarray(base.drain, dtype=np.float64)[idx]  # type: ignore[attr-defined]
        self.set_group_priors(estimate_group_priors(labels, groups, sample_weights))

    def set_group_priors(self, priors: GroupPriors) -> None:
        self.la_loss.set_priors(priors)
        logging.info("[logit_adjustment] %s", self.la_loss.describe())

    # ------------------------------------------------------------------ model
    def get_model_components(
        self, num_features: int
    ) -> tuple[nn.Module, Optional[nn.Module]]:
        return _three_layer_head(num_features), None

    # ------------------------------------------------------------------ loss
    def _adjusted_terms(self, model_output, targets, extra_info, weight):
        assert extra_info is not None and "drain" in extra_info, (
            "logit_adjustment needs group labels in extra_info['drain']"
        )
        logits, _ = model_output
        la_bce = self.la_loss(logits, targets, extra_info["drain"])
        # Raw-logit BCE / wBCE, identical in definition to every other method.
        bce, wbce = self.compute_bce_terms(logits, targets, weight=weight)
        components = {"bce": bce.item(), "wbce": wbce.item(), "la_bce": la_bce.item()}
        return logits, la_bce, components

    def compute_loss(
        self,
        model_output: tuple[torch.Tensor, Optional[torch.Tensor]],
        targets: torch.Tensor,
        extra_info: Optional[dict] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        _, la_bce, components = self._adjusted_terms(
            model_output, targets, extra_info, weight
        )
        return la_bce, components


class LogitAdjustmentScoreMatchingMethod(LogitAdjustmentMethod):
    """Logit adjustment + batch-level score matching (`logit_adjustment_score_matching`).

    Total loss = la_bce + score_matching_lambda * ScoreMatchingLoss(sigmoid(z), y, g),
    with the penalty on the *raw* logits z (see module docstring). Re-uses the
    `score_matching_*` config fields, so lambda values / Optuna ranges are directly
    comparable to `method=score_matching`.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.score_matching_loss = ScoreMatchingLoss(
            min_subgroup_count=getattr(config, "score_matching_min_subgroup_count", 1)
        )
        self.lambda_val = getattr(config, "score_matching_lambda", 1.0)

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        logits, la_bce, components = self._adjusted_terms(
            model_output, targets, extra_info, weight
        )
        assert extra_info is not None
        sm = self.score_matching_loss(
            probs=torch.sigmoid(logits.view(-1)),
            labels=targets,
            groups=extra_info["drain"],
        )
        components["score_matching_loss"] = sm.item()
        return la_bce + self.lambda_val * sm, components


class LogitAdjustmentDatasetScoreMatchingMethod(LogitAdjustmentMethod):
    """Logit adjustment + dataset-level score matching
    (`logit_adjustment_dataset_score_matching`).

    Total loss = la_bce + dataset_score_matching_lambda * DatasetScoreMatchingLoss(...),
    penalty on raw-logit probabilities. Re-uses the `dataset_score_matching_*` config
    fields and the same buffer-update protocol (`update_loss`) as
    `method=dataset_score_matching`.
    """

    def __init__(self, config: ExperimentConfig, dataset_size: Optional[int] = None):
        super().__init__(config)
        self.dataset_size = dataset_size
        self.score_matching_loss: Optional[DatasetScoreMatchingLoss] = None
        if dataset_size is not None:
            self.score_matching_loss = DatasetScoreMatchingLoss(
                dataset_size=dataset_size,
                min_subgroup_count=getattr(
                    config, "dataset_score_matching_min_subgroup_count", 10
                ),
                device=config.device,
            )
        self.lambda_val = getattr(config, "dataset_score_matching_lambda", 1.0)

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        assert self.score_matching_loss is not None, (
            "Dataset size unknown: clone(dataset_size=...) must be called first."
        )
        logits, la_bce, components = self._adjusted_terms(
            model_output, targets, extra_info, weight
        )
        assert extra_info is not None and "indices" in extra_info
        sm = self.score_matching_loss(
            probs=torch.sigmoid(logits.view(-1)),
            labels=targets,
            groups=extra_info["drain"],
            indices=extra_info["indices"],
        )
        components["dataset_score_matching_loss"] = sm.item()
        return la_bce + self.lambda_val * sm, components

    def update_loss(self, model_output, targets, extra_info=None, weight=None):
        assert self.score_matching_loss is not None
        assert extra_info is not None and "indices" in extra_info
        logits, _ = model_output
        self.score_matching_loss.update(
            probs=torch.sigmoid(logits.view(-1)),
            labels=targets,
            groups=extra_info["drain"],
            indices=extra_info["indices"],
        )

    def clone(self, dataset_size: Optional[int] = None):
        new_size = dataset_size if dataset_size is not None else self.dataset_size
        cloned = LogitAdjustmentDatasetScoreMatchingMethod(self.config, new_size)
        cloned.lambda_val = self.lambda_val
        if self.la_loss.priors is not None:
            cloned.la_loss.set_priors(self.la_loss.priors)
        return cloned


def priors_from_mapping(
    by_group: Mapping[int, float], pooled: float, missing: Optional[float] = None
) -> GroupPriors:
    """Convenience constructor, e.g. for tests or for re-using published prevalences."""
    return GroupPriors(
        by_group={int(k): float(v) for k, v in by_group.items()},
        missing=None if missing is None else float(missing),
        pooled=float(pooled),
        counts={},
    )

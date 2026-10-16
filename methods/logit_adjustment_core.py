"""
Group-conditional logit adjustment: framework-free core.

This file depends only on numpy and torch. It has no imports from the rest of this
repository and can be copied into other projects as is. `methods/logit_adjustment.py` is
the binary, drain-specific integration used for the paper. Everything general lives here.

What it does
------------
Given a classifier with logits z(x) in R^K, a group variable g (site, scanner, drain,
an intersection of several attributes, ...) and the training distribution's
group-conditional class priors pi_g(k) = P(y = k | g), train with

    L(x, y, g) = CE( z(x) + delta_g , y ),
    delta_g(k) = tau * ( log pi_g(k) - log pi_ref(k) ),

where pi_ref is one reference class prior shared by all groups. At inference nothing is
added: softmax(z(x)) is meant to be the posterior every group would have under the class
prior pi_ref, and no group label is needed. The offsets are fixed constants per group and
go *inside* the softmax. Subtracting log pi_g(y) from the loss value instead would be a
per-sample constant with zero gradient.

Binary special case. With a single logit z (sigmoid/BCE), the same loss is
BCE(z + tau * (logit pi_g - logit pi_ref), y) with pi = P(y = 1). This is identical to
the K = 2 softmax form with logits (0, z). The module handles both: logits of shape
(B,) or (B, 1) use BCE, logits of shape (B, K) use softmax cross-entropy.

Why
---
Assume the group affects the input only through a label-independent signature and through
the class prior:

    p(x | y, g) = p(x_c | y) * s(x_s | g),     P(y | g) = pi_g(y).

* Plain cross-entropy is minimised at the pooled posterior,

      softmax(z_CE(x)) ∝ p(x_c | y) * sum_g P(g | x_s) * pi_g(y),

  so the model gains by inferring the group from x_s. A spurious signature is then worth
  exactly as much as the prior differences between groups ("maximum likelihood forces
  group-specific priors").

* The adjusted loss is minimised at

      z*(x)_k = log p(x_c | k) + log pi_ref(k) + const(x),

  for *any* P(g | x). Each group's conditional risk is stationary at z* on its own, because
  softmax(z* + delta_g) is exactly that group's Bayes posterior. So the optimum ignores x_s,
  and every group is calibrated at pi_ref.

This is Saerens et al.'s (2002) prior-shift correction moved into the training loss. Plain
CE followed by the post-hoc map z -> z - delta_g has the same population optimum, but
training with the offset also keeps the gradients aligned during training. Importance
weights pi_ref(y) / pi_g(y) target the same minimiser at higher variance (cf. Puli et al.,
2022). With pi_ref uniform this is the balanced-softmax / logit-adjusted loss of Ren et al.
(2020) and Menon et al. (2021) applied per group, and the softmax form of Liu et al.'s
(2023) logit correction with a labelled rather than inferred attribute.

Deploying at a known class prior pi_s (e.g. a site's prevalence) is the same correction
again: z -> z + log pi_s - log pi_ref (`GroupLogitAdjustedLoss.retarget`).

Caveats
-------
* Only the *prior* channel of the group effect is removed. If the group also changes how
  the class looks, i.e. p(x_c | y, g) depends on g, the optimum still depends on g. For a
  shortcut carrier (scanner, site artefact, drain) this is usually the case to worry
  about least. For attributes with real prevalence differences (sex, age, ...), adjusting
  makes the model "diagnostic" with respect to them: their prior is removed from the
  output and must be re-applied at deployment if the decision should use it.
* Additive logit offsets remove the *incentive* to use the shortcut at the population
  optimum; they do not penalise it. Overparameterised models trained into the
  interpolating regime lose the effect of additive adjustments (Kini et al., 2021).
  Early stopping and weight decay keep training away from that regime.
* Priors are plain (weighted) empirical frequencies; no smoothing is applied. Frequencies
  of 0 or 1 are clipped to [PRIOR_EPS, 1 - PRIOR_EPS] (offset about ±9.2) to keep
  gradients finite, and a warning names the affected groups. Small intersectional cells
  therefore deserve a look at `describe()` before training.
* Label smoothing changes the optimum, and the correction is then no longer exact. Per-sample
  loss weights are fine: pass the same weights to `fit(..., sample_weights=...)` so that
  the priors describe the weighted training distribution, and use `reduction="none"`.

Usage
-----
Single group variable:

    loss_fn = GroupLogitAdjustedLoss(reference_prior="uniform")
    loss_fn.fit(train_labels, train_site_ids)                 # numpy / lists
    loss = loss_fn(logits, targets, site_ids)                 # (B, K) or (B,) logits

Intersectional groups (site x sex x scanner vendor):

    enc = GroupEncoder().fit(train_df[["site", "sex", "vendor"]])
    loss_fn = GroupLogitAdjustedLoss().fit(train_df["label"], enc.transform(train_df))
    batch_groups = torch.as_tensor(enc.transform(batch_df))   # or precompute per row
    loss = loss_fn(logits, targets, batch_groups)
    print(loss_fn.describe(group_names=enc.names))

Building block for custom losses (weights, extra terms):

    adjusted, keep = loss_fn.adjust_logits(logits, groups)
    per_sample = loss_fn(logits, targets, groups, reduction="none")

Group ids are non-negative integers. NaN or negative values mean "group unknown";
`missing_group` controls those rows ("own_group", "no_offset" or "exclude").

References
----------
- M. Saerens, P. Latinne, C. Decaestecker (2002). Adjusting the outputs of a classifier to
  new a priori probabilities: a simple procedure. Neural Computation 14(1):21-41.
- J. Ren, C. Yu, S. Sheng, X. Ma, H. Zhao, S. Yi, H. Li (2020). Balanced meta-softmax for
  long-tailed visual recognition. NeurIPS.
- A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, S. Kumar (2021). Long-tail
  learning via logit adjustment. ICLR. https://arxiv.org/abs/2007.07314
- G. R. Kini, O. Paraskevas, S. Oymak, C. Thrampoulidis (2021). Label-imbalanced and
  group-sensitive classification under overparameterization. NeurIPS.
  https://arxiv.org/abs/2103.01550
- A. Puli, L. H. Zhang, E. K. Oermann, R. Ranganath (2022). Out-of-distribution
  generalization in the presence of nuisance-induced spurious correlations. ICLR.
- S. Liu, X. Zhang, N. Sekhar, Y. Wu, P. Singhal, C. Fernandez-Granda (2023). Avoiding
  spurious correlations via logit correction. ICLR. https://arxiv.org/abs/2212.01433
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PRIOR_EPS",
    "MISSING_GROUP_POLICIES",
    "GroupClassPriors",
    "estimate_group_class_priors",
    "resolve_reference_prior",
    "GroupLogitAdjustedLoss",
    "GroupEncoder",
    "is_missing_group",
]

PRIOR_EPS = 1e-4
MISSING_GROUP_POLICIES = ("own_group", "no_offset", "exclude")


def is_missing_group(groups):
    """NaN or negative group ids mean "unknown" (numpy array or torch tensor)."""
    if isinstance(groups, torch.Tensor):
        if groups.dtype.is_floating_point:
            return torch.isnan(groups) | (groups < 0)
        return groups < 0
    groups = np.asarray(groups, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isnan(groups) | (groups < 0)


# ----------------------------------------------------------------------------- priors


@dataclass(frozen=True, eq=False)
class GroupClassPriors:
    """Class priors of the training distribution, per group.

    by_group[g][k] = P(y = k | g); missing[k] = P(y = k | group unknown) or None;
    pooled[k] = P(y = k). `counts` holds the (weighted) number of rows per group.
    """

    num_classes: int
    by_group: dict[int, np.ndarray]
    missing: Optional[np.ndarray]
    pooled: np.ndarray
    counts: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_rates(
        cls,
        by_group: Mapping[int, Any],
        pooled: Any,
        missing: Any = None,
        num_classes: Optional[int] = None,
    ) -> "GroupClassPriors":
        """Build priors from known rates. Scalars are read as binary P(y = 1)."""

        def vec(p):
            a = np.atleast_1d(np.asarray(p, dtype=np.float64))
            if a.size == 1:
                a = np.array([1.0 - a[0], a[0]])
            if not np.isclose(a.sum(), 1.0) or (a < 0).any():
                raise ValueError(f"Not a probability vector: {a}")
            return a

        groups = {int(g): vec(p) for g, p in by_group.items()}
        pooled_v = vec(pooled)
        k = num_classes if num_classes is not None else pooled_v.size
        for g, v in groups.items():
            if v.size != k:
                raise ValueError(f"Group {g} has {v.size} classes, expected {k}.")
        return cls(
            num_classes=k,
            by_group=groups,
            missing=None if missing is None else vec(missing),
            pooled=pooled_v,
        )

    def positive_rate(self, group: int) -> float:
        """Binary convenience: P(y = 1 | group)."""
        return float(self.by_group[group][1])


def estimate_group_class_priors(
    labels,
    groups,
    num_classes: Optional[int] = None,
    sample_weights=None,
) -> GroupClassPriors:
    """Estimate P(y | g), P(y | g unknown) and P(y) from training labels.

    Args:
        labels: integer class labels in [0, num_classes).
        groups: integer group ids >= 0; NaN or negative = unknown.
        num_classes: K. Inferred as max(label) + 1 if omitted.
        sample_weights: relative sampling or loss weights of the rows (e.g. a
            WeightedRandomSampler's weights, or per-sample loss weights). Priors are then
            those of the weighted distribution the optimiser actually sees.
    """
    labels = np.asarray(labels).reshape(-1)
    groups = np.asarray(groups, dtype=np.float64).reshape(-1)
    if labels.shape != groups.shape:
        raise ValueError(f"labels {labels.shape} and groups {groups.shape} differ")
    if labels.size == 0:
        raise ValueError("No training rows given.")
    if not np.all(np.equal(np.mod(labels, 1), 0)):
        raise ValueError("Labels must be integer class indices.")
    labels = labels.astype(np.int64)
    k = int(labels.max()) + 1 if num_classes is None else int(num_classes)
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(f"Labels must lie in [0, {k}); got [{labels.min()}, {labels.max()}].")

    if sample_weights is None:
        w = np.ones(labels.shape, dtype=np.float64)
    else:
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if w.shape != labels.shape:
            raise ValueError(f"sample_weights {w.shape} do not match labels {labels.shape}")
        if (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("sample_weights must be finite, non-negative, not all 0.")

    missing = is_missing_group(groups)
    known_ids = groups[~missing]
    if (known_ids != np.round(known_ids)).any():
        raise ValueError("Group ids must be integer-valued (or NaN for missing).")

    def class_dist(mask):
        return np.bincount(labels[mask], weights=w[mask], minlength=k) / w[mask].sum()

    by_group: dict[int, np.ndarray] = {}
    counts: dict[str, float] = {}
    for g in np.unique(known_ids).astype(np.int64).tolist():
        mask = (~missing) & (groups == g)
        if w[mask].sum() > 0:
            by_group[int(g)] = class_dist(mask)
            counts[f"g{g}"] = float(w[mask].sum())

    miss = None
    if missing.any() and w[missing].sum() > 0:
        miss = class_dist(missing)
        counts["missing"] = float(w[missing].sum())

    return GroupClassPriors(
        num_classes=k,
        by_group=by_group,
        missing=miss,
        pooled=class_dist(np.ones(labels.shape, dtype=bool)),
        counts=counts,
    )


def resolve_reference_prior(spec, pooled: np.ndarray, num_classes: int) -> np.ndarray:
    """"uniform" -> 1/K, "pooled" -> training marginal, scalar -> binary P(y=1), or a K-vector."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "uniform":
            return np.full(num_classes, 1.0 / num_classes)
        if key == "pooled":
            return np.asarray(pooled, dtype=np.float64)
        try:
            spec = float(key)
        except ValueError as e:
            raise ValueError(
                "reference_prior must be 'uniform', 'pooled', a number (binary P(y=1)) "
                f"or a probability vector; got {spec!r}"
            ) from e
    ref = np.atleast_1d(np.asarray(spec, dtype=np.float64))
    if ref.size == 1:
        if num_classes != 2:
            raise ValueError("A scalar reference prior is only defined for binary problems.")
        if not 0.0 < ref[0] < 1.0:
            raise ValueError(f"Reference prior must lie in (0, 1); got {ref[0]}")
        ref = np.array([1.0 - ref[0], ref[0]])
    if ref.size != num_classes or (ref <= 0).any() or not np.isclose(ref.sum(), 1.0):
        raise ValueError(
            f"Reference prior must be a strictly positive {num_classes}-vector summing to 1; "
            f"got {ref}"
        )
    return ref


def _log_clipped(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, PRIOR_EPS, 1.0))


# ----------------------------------------------------------------------------- loss


class GroupLogitAdjustedLoss(nn.Module):
    """Cross-entropy on z + delta_g, delta_g = tau * (log pi_g - log pi_ref).

    Logits of shape (B, K) use softmax cross-entropy (targets: class indices). Logits of
    shape (B,) or (B, 1) use BCE (binary priors required; targets: 0/1). Call `fit` or
    `set_priors` before the first forward pass.
    """

    offset_table: torch.Tensor  # (max_group_id + 1, K); NaN rows = id unseen in training
    missing_offset: torch.Tensor  # (K,) offset for rows without a group

    def __init__(
        self,
        reference_prior="uniform",
        tau: float = 1.0,
        missing_group: str = "own_group",
        priors: Optional[GroupClassPriors] = None,
    ):
        super().__init__()
        if missing_group not in MISSING_GROUP_POLICIES:
            raise ValueError(
                f"missing_group must be one of {MISSING_GROUP_POLICIES}; got {missing_group!r}"
            )
        if tau < 0:
            raise ValueError(f"tau must be >= 0; got {tau}")
        self.reference_prior_spec = reference_prior
        self.tau = float(tau)
        self.missing_group = missing_group
        self.priors: Optional[GroupClassPriors] = None
        self.reference_prior: Optional[np.ndarray] = None
        self.register_buffer("offset_table", torch.empty(0, 0))
        self.register_buffer("missing_offset", torch.empty(0))
        self._warned_missing_fallback = False
        if priors is not None:
            self.set_priors(priors)

    # ---------------------------------------------------------------- fitting
    @property
    def is_fitted(self) -> bool:
        return self.priors is not None

    @property
    def num_classes(self) -> int:
        if self.priors is None:
            raise RuntimeError("Priors not fitted yet.")
        return self.priors.num_classes

    def fit(self, labels, groups, num_classes=None, sample_weights=None):
        """Estimate priors from training labels/groups (see estimate_group_class_priors)."""
        self.set_priors(
            estimate_group_class_priors(labels, groups, num_classes, sample_weights)
        )
        return self

    def set_priors(self, priors: GroupClassPriors) -> None:
        if not priors.by_group:
            raise ValueError("No labelled group in the training data.")
        k = priors.num_classes
        ref = resolve_reference_prior(self.reference_prior_spec, priors.pooled, k)
        log_ref = np.log(ref)

        clipped = [g for g, p in priors.by_group.items() if (p < PRIOR_EPS).any()]
        if clipped:
            logging.warning(
                "[logit_adjustment] %d group(s) have a class prior of 0 in the training "
                "data (e.g. %s); their offsets are clipped at log(%g). Check describe().",
                len(clipped),
                clipped[:5],
                PRIOR_EPS,
            )

        table = np.full((max(priors.by_group) + 1, k), np.nan)
        for g, p in priors.by_group.items():
            table[g] = self.tau * (_log_clipped(p) - log_ref)
        miss = np.zeros(k)
        if self.missing_group == "own_group" and priors.missing is not None:
            miss = self.tau * (_log_clipped(priors.missing) - log_ref)

        device = self.offset_table.device
        self.offset_table = torch.tensor(table, dtype=torch.float32, device=device)
        self.missing_offset = torch.tensor(miss, dtype=torch.float32, device=device)
        self.priors = priors
        self.reference_prior = ref

    # ---------------------------------------------------------------- offsets
    def class_offsets(self, groups) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample offsets (B, K) and a (B,) mask of rows that enter the loss."""
        if self.priors is None:
            raise RuntimeError(
                "Group priors have not been fitted; call fit()/set_priors() first."
            )
        g = torch.as_tensor(groups).reshape(-1).to(self.offset_table.device)
        g = g.to(torch.float32)
        missing = is_missing_group(g)
        known = ~missing
        out = torch.zeros(g.shape[0], self.num_classes, device=g.device)

        if known.any():
            ids = g[known]
            if (ids != torch.round(ids)).any():
                raise ValueError("Group ids must be integer-valued (or NaN for missing).")
            ids = ids.long()
            n_rows = self.offset_table.shape[0]
            unseen = ids >= n_rows
            if not unseen.any():
                unseen = torch.isnan(self.offset_table[ids, 0])
            if unseen.any():
                bad = ids[unseen].unique().tolist()
                raise ValueError(f"Group id(s) {bad} were not seen in the training data.")
            out[known] = self.offset_table[ids]

        keep = torch.ones_like(missing)
        if missing.any():
            if self.missing_group == "exclude":
                keep = known
            elif self.missing_group == "own_group":
                if self.priors.missing is None and not self._warned_missing_fallback:
                    logging.warning(
                        "[logit_adjustment] Rows without a group label encountered, but "
                        "the training data had none, so their prior is unknown. Using "
                        "offset 0 (as for missing_group='no_offset')."
                    )
                    self._warned_missing_fallback = True
                out[missing] = self.missing_offset
            # "no_offset": leave at 0
        return out, keep

    def _follow_device(self, ref: torch.Tensor) -> None:
        # Losses held by non-Module objects are not moved by training frameworks.
        if self.offset_table.device != ref.device:
            self.to(ref.device)

    def adjust_logits(self, logits: torch.Tensor, groups) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits + offsets, keep mask). Same shape as `logits` (float32)."""
        self._follow_device(logits)
        offsets, keep = self.class_offsets(groups)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            if self.num_classes != 2:
                raise ValueError(
                    f"Single-logit input needs binary priors; priors have {self.num_classes} classes."
                )
            delta = (offsets[:, 1] - offsets[:, 0]).view(logits.shape)
            return logits.float() + delta, keep
        if logits.dim() != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits of shape (B,), (B, 1) or (B, {self.num_classes}); "
                f"got {tuple(logits.shape)}"
            )
        return logits.float() + offsets, keep

    # ---------------------------------------------------------------- loss
    def forward(self, logits, targets, groups, reduction: str = "mean") -> torch.Tensor:
        """Adjusted cross-entropy.

        reduction: "mean" / "sum" over rows that enter the loss, or "none" for per-sample
        losses. With "none", rows excluded by missing_group="exclude" have loss 0.
        """
        adjusted, keep = self.adjust_logits(logits, groups)
        binary = adjusted.dim() == 1 or adjusted.shape[-1] == 1
        if binary:
            per = F.binary_cross_entropy_with_logits(
                adjusted.reshape(-1), targets.reshape(-1).float(), reduction="none"
            )
        else:
            per = F.cross_entropy(adjusted, targets.reshape(-1).long(), reduction="none")
        keep = keep.to(per.device)
        if reduction == "none":
            return torch.where(keep, per, per * 0.0)
        if not keep.any():
            return per.sum() * 0.0  # keeps the graph connected
        if reduction == "mean":
            return per[keep].mean()
        if reduction == "sum":
            return per[keep].sum()
        raise ValueError(f"Unknown reduction {reduction!r}")

    # ---------------------------------------------------------------- deployment
    def retarget(self, logits: torch.Tensor, target_prior) -> torch.Tensor:
        """Map raw logits (calibrated at pi_ref) to a deployment class prior pi_s.

        target_prior: scalar P(y=1) for binary single-logit models, or a K-vector.
        """
        if self.reference_prior is None or self.priors is None:
            raise RuntimeError("Priors not fitted yet.")
        k = self.reference_prior.size
        target = resolve_reference_prior(target_prior, self.priors.pooled, k)
        shift = torch.tensor(np.log(target) - np.log(self.reference_prior), dtype=torch.float32)
        shift = shift.to(logits.device)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
            return logits + (shift[1] - shift[0])
        return logits + shift

    # ---------------------------------------------------------------- logging
    def describe(self, group_names: Optional[Mapping[int, str]] = None) -> str:
        if self.priors is None:
            return "GroupLogitAdjustedLoss(unfitted)"
        binary = self.priors.num_classes == 2

        def fmt(v):
            return f"{v[1]:.4f}" if binary else "[" + ", ".join(f"{x:.3f}" for x in v) + "]"

        def fmt_off(v):
            return f"{v[1] - v[0]:+.4f}" if binary else "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"

        what = "P(y=1)" if binary else "P(y)"
        lines = [
            f"pi_ref={fmt(self.reference_prior)} (spec={self.reference_prior_spec!r}), "
            f"tau={self.tau:g}, pooled {what}={fmt(self.priors.pooled)}"
        ]
        for g, p in sorted(self.priors.by_group.items()):
            name = group_names.get(g, f"g{g}") if group_names else f"g{g}"
            n = self.priors.counts.get(f"g{g}")
            n_txt = f", n={n:.0f}" if n is not None else ""
            off = self.offset_table[g].cpu().numpy()
            lines.append(f"  {name}: {what}={fmt(p)}{n_txt}, offset={fmt_off(off)}")
        if self.priors.missing is not None:
            off = self.missing_offset.cpu().numpy()
            lines.append(
                f"  missing: {what}={fmt(self.priors.missing)}, policy={self.missing_group}, "
                f"offset={fmt_off(off)}"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------------------- groups


def _is_missing_value(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return type(v).__name__ in ("NAType", "NaTType")  # pandas NA / NaT


def _normalise_value(v):
    """numpy scalars -> Python scalars; integral floats -> int (1.0 and 1 are one level)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _sort_key(v):
    if isinstance(v, bool):
        return (0, int(v), "")
    if isinstance(v, (int, float)):
        return (0, v, "")
    if isinstance(v, str):
        return (1, 0, v)
    return (2, 0, repr(v))


class GroupEncoder:
    """Encode one or several group attributes as a single integer id per row.

    With several attributes the id identifies their *intersection* (e.g. site x sex x
    vendor). Ids are 0..n_groups-1, assigned in sorted order of the observed combinations
    during `fit`. Rows with any missing attribute (None / NaN / pandas NA), and
    combinations unseen during fit (unseen="missing"), map to `missing_value`.

    Accepted inputs (for fit and transform):
      * a pandas DataFrame (optionally `columns=[...]`), or a dict {name: values};
      * a single 1-D array-like (one attribute);
      * a list/tuple of 1-D array-likes (one per attribute);
      * a 2-D numpy array of shape (n_rows, n_attributes).

    Example:
        enc = GroupEncoder().fit(train_df[["site", "sex", "vendor"]])
        train_groups = enc.transform(train_df)   # float array, NaN = missing
        enc.names[3]                             # 'site=CAM|sex=F|vendor=GE'
    """

    def __init__(self, missing_value: float | int = np.nan, unseen: str = "missing"):
        if unseen not in ("missing", "error"):
            raise ValueError("unseen must be 'missing' or 'error'")
        if not (isinstance(missing_value, float) and math.isnan(missing_value)):
            if not float(missing_value).is_integer() or missing_value >= 0:
                raise ValueError("missing_value must be NaN or a negative integer.")
        self.missing_value = missing_value
        self.unseen = unseen
        self.attributes: list[str] = []
        self.cells: list[tuple[Hashable, ...]] = []
        self._index: dict[tuple[Hashable, ...], int] = {}

    # ---------------------------------------------------------------- input parsing
    @staticmethod
    def _columns(data, columns: Optional[Sequence[str]] = None):
        if hasattr(data, "columns") and hasattr(data, "iloc"):  # pandas DataFrame
            names = list(columns) if columns is not None else list(data.columns)
            return names, [data[c].to_numpy(dtype=object) for c in names]
        if columns is not None:
            raise ValueError("`columns` is only supported for DataFrames.")
        if isinstance(data, Mapping):
            return list(map(str, data.keys())), [
                np.asarray(v, dtype=object).reshape(-1) for v in data.values()
            ]
        if isinstance(data, np.ndarray) and data.ndim == 2:
            return [f"a{i}" for i in range(data.shape[1])], [
                data[:, i].astype(object) for i in range(data.shape[1])
            ]
        if isinstance(data, (list, tuple)) and data and all(
            np.ndim(c) == 1 and not isinstance(c, str) for c in data
        ):
            cols = [np.asarray(c, dtype=object).reshape(-1) for c in data]
            return [f"a{i}" for i in range(len(cols))], cols
        arr = np.asarray(getattr(data, "to_numpy", lambda: data)(), dtype=object)
        if arr.ndim != 1:
            raise ValueError(f"Cannot interpret group data of shape {arr.shape}.")
        return [str(getattr(data, "name", None) or "a0")], [arr]

    def _rows(self, data, columns):
        names, cols = self._columns(data, columns)
        n = {len(c) for c in cols}
        if len(n) != 1:
            raise ValueError(f"Group attributes have different lengths: {sorted(n)}")
        rows = []
        for vals in zip(*cols, strict=True):
            if any(_is_missing_value(v) for v in vals):
                rows.append(None)
            else:
                rows.append(tuple(_normalise_value(v) for v in vals))
        return names, rows

    # ---------------------------------------------------------------- API
    def fit(self, data, columns: Optional[Sequence[str]] = None) -> "GroupEncoder":
        names, rows = self._rows(data, columns)
        observed = {r for r in rows if r is not None}
        if not observed:
            raise ValueError("No row has all group attributes present.")
        self.attributes = names
        self.cells = sorted(observed, key=lambda t: tuple(_sort_key(v) for v in t))
        self._index = {c: i for i, c in enumerate(self.cells)}
        return self

    def transform(self, data, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        if not self._index:
            raise RuntimeError("GroupEncoder is not fitted.")
        names, rows = self._rows(data, columns if columns is not None else (
            self.attributes if hasattr(data, "columns") and hasattr(data, "iloc") else None
        ))
        if len(names) != len(self.attributes):
            raise ValueError(
                f"Expected {len(self.attributes)} attributes {self.attributes}; got {names}"
            )
        is_nan = isinstance(self.missing_value, float) and math.isnan(self.missing_value)
        out = np.empty(len(rows), dtype=np.float64 if is_nan else np.int64)
        n_unseen = 0
        for i, r in enumerate(rows):
            if r is None:
                out[i] = self.missing_value
            elif r in self._index:
                out[i] = self._index[r]
            else:
                if self.unseen == "error":
                    combo = dict(zip(self.attributes, r, strict=True))
                    raise ValueError(f"Unseen group combination {combo}")
                out[i] = self.missing_value
                n_unseen += 1
        if n_unseen:
            logging.warning(
                "[GroupEncoder] %d row(s) with combinations unseen during fit were mapped "
                "to the missing value.",
                n_unseen,
            )
        return out

    def fit_transform(self, data, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        return self.fit(data, columns).transform(data, columns)

    @property
    def n_groups(self) -> int:
        return len(self.cells)

    @property
    def names(self) -> dict[int, str]:
        return {
            i: "|".join(f"{a}={v}" for a, v in zip(self.attributes, cell, strict=True))
            for i, cell in enumerate(self.cells)
        }

    def decode(self, ids) -> list[Optional[tuple[Hashable, ...]]]:
        """Group ids -> attribute tuples (None for missing)."""
        out = []
        for i in np.asarray(ids, dtype=np.float64).reshape(-1):
            out.append(None if (np.isnan(i) or i < 0) else self.cells[int(i)])
        return out

"""
Group-conditional logit adjustment ("prior-corrected BCE"): integration for this repo.

This module wires the general, framework-free implementation in
`methods/logit_adjustment_core.py` into the binary pneumothorax / drain setting of the paper.
The core has the full derivation, caveats and references, supports K classes, any number of
groups and intersectional groups, and can be copied into other projects. Here:

    L(x, y, g) = BCE( z(x) + tau * (logit(pi_g) - logit(pi_ref)), y ),

with g = Drain label, pi_g = P(y = 1 | g) estimated on the training split, and pi_ref a
common reference prevalence. At inference the raw score sigmoid(z) is used, so no drain
label is needed at test time. With pi_ref = 0.5 (the default) this is the in-training
counterpart of the post-hoc prevalence calibration in `calibration/core.py`, which
re-balances each drain group to 50/50 before fitting group-wise beta calibrators.

Under p(x | y, g) = p(x_c | y) s(x_s | g), the population optimum of plain BCE contains
logit(sum_g P(g | x_s) pi_g), so the model gains by detecting the drain. The adjusted loss's
optimum is the drain-free likelihood ratio plus logit(pi_ref).

Relation to the score-matching losses (combined variants)
---------------------------------------------------------
`ScoreMatchingLoss` / `DatasetScoreMatchingLoss` penalise the spread across groups of
E[sigmoid(z) | y, g], separately for y = 0 and y = 1.

* Under the factorisation above, the adjusted-loss optimum depends on x_c only, and
  p(x_c | y) is the same in every group. So E[sigmoid(z*) | y, g] is equal across groups
  and the penalty is zero there. The objectives are compatible at the population level.
* The penalty must be evaluated on the raw logits z, the deployed score. On z + delta_g
  the group means differ by construction and the penalty would fight the adjustment. The
  combined classes below do this correctly.
* Where p(x_c | y, g) depends on g (drained pneumothoraces are treated and often smaller),
  the adjusted-loss optimum legitimately depends on g and score matching pushes against
  it.
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

Implementation notes
--------------------
* Prevalences are estimated once from the *training* split. This happens in
  `CXPLightningModule.setup` via `fit_group_priors`, before the per-split clones are made,
  so the train, val and test loss use the same offsets. With a `WeightedRandomSampler`
  (`balance_train=true`) they are computed under the sampling weights. The repo's
  (y, drain)-balanced sampler makes both groups 50/50, so the offsets vanish for
  pi_ref = 0.5.
* Drain = NaN (~82 % of SIIM training rows) follows `logit_adjustment_missing_group`:
    - "own_group" (default): its own estimated prevalence. This removes the difference
      between labelled and unlabelled rows, but cannot remove the shortcut *within* the
      unlabelled rows.
    - "no_offset": plain BCE.
    - "exclude": drop the rows from the classification loss.
* `logit_adjustment_reference_prior`: "uniform" (0.5), "pooled" or a number in (0, 1). It
  shifts all logits by one constant, so AUROC-type metrics (incl. the pairwise cross-group
  AUROCs) do not depend on it. Brier score and calibration-in-the-large do.
* `logit_adjustment_tau`: offset scale (1 = exact Bayes correction, 0 = plain BCE). It is
  searched in [0, 2] by Optuna only if `logit_adjustment_tune_tau=true`.
* The reported "bce"/"wbce" components are raw-logit BCE, identical in definition to every
  other method, so they stay comparable across methods. The objective is reported as
  "la_bce". Hence `select_chkpt_on=BCE` selects on the *unadjusted* val BCE.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from .base import BaseMethod
from .logit_adjustment_core import (
    GroupClassPriors,
    GroupLogitAdjustedLoss,
    estimate_group_class_priors,
)
from .score_matching import ScoreMatchingLoss
from .score_matching_dataset import DatasetScoreMatchingLoss
from config import ExperimentConfig


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


def _three_layer_head(num_features: int) -> nn.Module:
    # Same head as standard / score matching, so results are comparable.
    return nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, 1))


class LogitAdjustmentMethod(BaseMethod):
    """Standalone group-conditional logit adjustment (method key: `logit_adjustment`)."""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.la_loss = GroupLogitAdjustedLoss(
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
        # num_classes=2 makes the estimator reject non-binary labels.
        self.set_group_priors(
            estimate_group_class_priors(labels, groups, num_classes=2, sample_weights=sample_weights)
        )

    def set_group_priors(self, priors: GroupClassPriors) -> None:
        if priors.num_classes != 2:
            raise ValueError("This repository's logit adjustment is binary (Pneumothorax).")
        self.la_loss.set_priors(priors)
        logging.info("[logit_adjustment] drain-group priors:\n%s", self.la_loss.describe())

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
        la_bce = self.la_loss(logits.reshape(-1), targets, extra_info["drain"])
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
    with the penalty on the *raw* logits z. Re-uses the `score_matching_*` config fields.
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
    fields and the same buffer-update protocol (`update_loss`).
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

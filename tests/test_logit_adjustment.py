import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, WeightedRandomSampler

import methods
from config import ExperimentConfig
from lightning_datamodule import _group_sample_weights
from lightning_module import CXPLightningModule
from methods.logit_adjustment import (
    LogitAdjustmentDatasetScoreMatchingMethod,
    LogitAdjustmentMethod,
    LogitAdjustmentScoreMatchingMethod,
    estimate_group_priors,
    priors_from_mapping,
)
from methods.score_matching import ScoreMatchingLoss


def logit(p):
    return math.log(p / (1 - p))


def make_method(cls=LogitAdjustmentMethod, priors=None, **cfg):
    config = ExperimentConfig(**cfg)
    method = cls(config)
    method.set_group_priors(
        priors
        if priors is not None
        else priors_from_mapping({0: 0.25, 1: 0.75}, pooled=0.5, missing=0.2)
    )
    return method


class FakeDataset:
    """Minimal stand-in exposing what CXP_dataset exposes: `.labels`, `.drain`."""

    def __init__(self, labels, drain):
        self.labels = pd.Series(labels, dtype=int)
        self.drain = pd.Series(drain, dtype=np.float32)

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------- priors


def test_estimate_group_priors_counts_missing_and_weights():
    labels = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, np.nan, np.nan])
    p = estimate_group_priors(labels, groups)
    assert p.by_group == {0: 0.25, 1: 0.75}
    assert p.missing == 0.5
    assert p.pooled == 0.5

    # Weights re-balance group 0 to 50/50 (positive row gets weight 3).
    w = np.array([3, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    p_w = estimate_group_priors(labels, groups, w)
    assert p_w.by_group[0] == pytest.approx(0.5)
    assert p_w.by_group[1] == pytest.approx(0.75)


def test_estimate_group_priors_rejects_bad_input():
    with pytest.raises(ValueError):
        estimate_group_priors(np.array([0, 2]), np.array([0, 1]))
    with pytest.raises(ValueError):
        estimate_group_priors(np.array([0, 1]), np.array([0.5, 1.0]))


def test_fit_group_priors_from_subset_and_balanced_sampler():
    rng = np.random.default_rng(0)
    drain = rng.integers(0, 2, 400).astype(float)
    labels = (rng.random(400) < np.where(drain == 1, 0.75, 0.25)).astype(int)
    base = FakeDataset(labels, drain)
    subset = Subset(cast(Any, base), list(range(0, 400, 2)))

    m = LogitAdjustmentMethod(ExperimentConfig())
    m.fit_group_priors(subset)
    expected = estimate_group_priors(labels[::2], drain[::2])
    assert m.la_loss.priors.by_group == pytest.approx(expected.by_group)

    # The repo's balanced sampler (inverse frequency over (y, drain)) -> both groups 50/50
    weights = _group_sample_weights(subset).numpy()
    m.fit_group_priors(subset, sample_weights=weights)
    assert m.la_loss.priors.by_group[0] == pytest.approx(0.5)
    assert m.la_loss.priors.by_group[1] == pytest.approx(0.5)
    assert torch.allclose(m.la_loss.offset_table, torch.zeros(2), atol=1e-6)


# --------------------------------------------------------------------------- loss


def test_loss_is_bce_on_shifted_logits_and_bce_component_is_raw():
    m = make_method()
    z = torch.randn(8, 1, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    g = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    loss, comps = m.compute_loss((z, None), y, extra_info={"drain": g})

    offsets = torch.where(g == 1, logit(0.75), logit(0.25)).float()
    expected = F.binary_cross_entropy_with_logits(z.view(-1) + offsets, y.float())
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert comps["la_bce"] == pytest.approx(expected.item(), rel=1e-6)
    raw = F.binary_cross_entropy_with_logits(z.view(-1), y.float())
    assert comps["bce"] == pytest.approx(raw.item(), rel=1e-6)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_tau_zero_is_plain_bce():
    m = make_method(logit_adjustment_tau=0.0)
    z = torch.randn(6, 1)
    y = torch.tensor([0, 1, 0, 1, 1, 0])
    g = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    loss, _ = m.compute_loss((z, None), y, extra_info={"drain": g})
    assert loss.item() == pytest.approx(
        F.binary_cross_entropy_with_logits(z.view(-1), y.float()).item(), rel=1e-6
    )


@pytest.mark.parametrize(
    "spec, ref", [("uniform", 0.5), ("pooled", 0.3), ("0.2", 0.2), (0.4, 0.4)]
)
def test_reference_prior_options(spec, ref):
    priors = priors_from_mapping({0: 0.1, 1: 0.6}, pooled=0.3)
    m = make_method(priors=priors, logit_adjustment_reference_prior=spec)
    assert m.la_loss.reference_prior == pytest.approx(ref)
    assert m.la_loss.offset_table[0].item() == pytest.approx(logit(0.1) - logit(ref), rel=1e-5)
    assert m.la_loss.offset_table[1].item() == pytest.approx(logit(0.6) - logit(ref), rel=1e-5)


def test_invalid_options_raise():
    with pytest.raises(ValueError):
        make_method(logit_adjustment_reference_prior="balanced")
    with pytest.raises(ValueError):
        make_method(logit_adjustment_reference_prior="1.5")
    with pytest.raises(ValueError):
        LogitAdjustmentMethod(ExperimentConfig(logit_adjustment_missing_group="drop"))


def test_missing_group_policies():
    z = torch.zeros(4, 1, requires_grad=True)
    y = torch.tensor([1, 0, 1, 0])
    g = torch.tensor([0.0, 1.0, float("nan"), float("nan")])
    off0, off1, offm = logit(0.25), logit(0.75), logit(0.2)

    def bce(offs, ys):
        return F.binary_cross_entropy_with_logits(
            torch.tensor(offs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
        ).item()

    own = make_method(logit_adjustment_missing_group="own_group")
    assert own.compute_loss((z, None), y, {"drain": g})[0].item() == pytest.approx(
        bce([off0, off1, offm, offm], [1, 0, 1, 0]), rel=1e-6
    )
    no_off = make_method(logit_adjustment_missing_group="no_offset")
    assert no_off.compute_loss((z, None), y, {"drain": g})[0].item() == pytest.approx(
        bce([off0, off1, 0.0, 0.0], [1, 0, 1, 0]), rel=1e-6
    )
    excl = make_method(logit_adjustment_missing_group="exclude")
    assert excl.compute_loss((z, None), y, {"drain": g})[0].item() == pytest.approx(
        bce([off0, off1], [1, 0]), rel=1e-6
    )
    # Batch with only missing groups under "exclude": differentiable zero.
    loss, _ = excl.compute_loss((z[2:], None), y[2:], {"drain": g[2:]})
    assert loss.item() == 0.0
    loss.backward()


def test_unfitted_and_unseen_groups_raise():
    m = LogitAdjustmentMethod(ExperimentConfig())
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.compute_loss((torch.zeros(1, 1), None), torch.tensor([1]), {"drain": torch.tensor([0.0])})
    m = make_method()
    with pytest.raises(ValueError, match="not seen"):
        m.compute_loss((torch.zeros(1, 1), None), torch.tensor([1]), {"drain": torch.tensor([2.0])})


def test_extreme_group_prior_gives_finite_gradients():
    # A group without positives would give an infinite offset without clipping.
    m = make_method(priors=priors_from_mapping({0: 0.0, 1: 1.0}, pooled=0.5))
    z = torch.zeros(2, 1, requires_grad=True)
    loss, _ = m.compute_loss((z, None), torch.tensor([0, 1]), {"drain": torch.tensor([0.0, 1.0])})
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


# --------------------------------------------------------------------------- combined


def test_combined_batch_score_matching_uses_raw_scores():
    m = make_method(
        LogitAdjustmentScoreMatchingMethod,
        score_matching_lambda=3.0,
        score_matching_min_subgroup_count=1,
    )
    torch.manual_seed(0)
    z = torch.randn(16, 1, requires_grad=True)
    y = torch.tensor([0, 1] * 8)
    g = torch.tensor([0.0] * 8 + [1.0] * 8)

    loss, comps = m.compute_loss((z, None), y, extra_info={"drain": g})

    la, _ = make_method().compute_loss((z, None), y, extra_info={"drain": g})
    sm = ScoreMatchingLoss(min_subgroup_count=1)(torch.sigmoid(z.view(-1)), y, g)
    assert loss.item() == pytest.approx(la.item() + 3.0 * sm.item(), rel=1e-5)
    assert comps["score_matching_loss"] == pytest.approx(sm.item(), rel=1e-6)


def test_combined_dataset_score_matching_clone_and_update():
    template = make_method(
        LogitAdjustmentDatasetScoreMatchingMethod, dataset_score_matching_lambda=2.0,
        dataset_score_matching_min_subgroup_count=1,
    )
    template.lambda_val = 5.0
    m = template.clone(dataset_size=8)
    assert isinstance(m, LogitAdjustmentDatasetScoreMatchingMethod)
    assert m.lambda_val == 5.0
    assert m.la_loss.priors == template.la_loss.priors
    assert torch.equal(m.la_loss.offset_table, template.la_loss.offset_table)
    m.score_matching_loss = m.score_matching_loss.to("cpu")  # config.device may be cuda

    z = torch.randn(8, 1, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    g = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    info = {"drain": g, "indices": torch.arange(8)}
    m.update_loss((z.detach(), None), y, extra_info=info)
    assert torch.isfinite(m.score_matching_loss.score_buffer).all()

    loss, comps = m.compute_loss((z, None), y, extra_info=info)
    la, _ = make_method().compute_loss((z, None), y, extra_info={"drain": g})
    assert loss.item() == pytest.approx(
        la.item() + 5.0 * comps["dataset_score_matching_loss"], rel=1e-5
    )
    loss.backward()


def test_factory_and_standalone_has_no_update_hook():
    cfg = ExperimentConfig()
    assert isinstance(methods.get_method("logit_adjustment", cfg), LogitAdjustmentMethod)
    assert isinstance(
        methods.get_method("logit_adjustment_score_matching", cfg),
        LogitAdjustmentScoreMatchingMethod,
    )
    assert isinstance(
        methods.get_method("logit_adjustment_dataset_score_matching", cfg),
        LogitAdjustmentDatasetScoreMatchingMethod,
    )
    # The Lightning module re-runs a forward pass for methods with `update_loss`;
    # only the dataset-buffer variant should pay for that.
    assert not hasattr(methods.get_method("logit_adjustment", cfg), "update_loss")
    assert not hasattr(
        methods.get_method("logit_adjustment_score_matching", cfg), "update_loss"
    )


# --------------------------------------------------------------------------- Lightning hook


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.clf = nn.Linear(2, 1)
        self.projection_head = None

    def forward(self, x):
        return self.clf(x), None


def _attach_datamodule(module, train_dataset, sampler=None):
    dm = SimpleNamespace(
        train_dataset=train_dataset,
        val_dataset=train_dataset,
        train_sampler=sampler,
        test_dataset_aligned=train_dataset,
        test_dataset_misaligned=train_dataset,
    )
    module._trainer = SimpleNamespace(datamodule=dm)


def test_lightning_setup_fits_priors_before_cloning():
    labels = [1, 0, 0, 0, 1, 1, 1, 0]
    drain = [0, 0, 0, 0, 1, 1, 1, 1]
    cfg = ExperimentConfig(backbone="medsiglip")
    method = LogitAdjustmentMethod(cfg)
    module = CXPLightningModule(cast(Any, _DummyModel()), method, cfg)
    _attach_datamodule(module, FakeDataset(labels, drain))

    module.setup("fit")
    for m in (module.train_method, module.val_method):
        assert m.la_loss.priors.by_group == {0: 0.25, 1: 0.75}
    module.setup("test")
    assert module.test_method_aligned.la_loss.priors.by_group == {0: 0.25, 1: 0.75}


def test_lightning_setup_uses_weighted_sampler():
    labels = [1, 0, 0, 0, 1, 1, 1, 0]
    drain = [0, 0, 0, 0, 1, 1, 1, 1]
    ds = FakeDataset(labels, drain)
    sampler = WeightedRandomSampler(_group_sample_weights(ds), num_samples=8)
    cfg = ExperimentConfig(backbone="medsiglip")
    module = CXPLightningModule(cast(Any, _DummyModel()), LogitAdjustmentMethod(cfg), cfg)
    _attach_datamodule(module, ds, sampler)
    module.setup("fit")
    assert module.train_method.la_loss.priors.by_group[0] == pytest.approx(0.5)
    assert module.train_method.la_loss.priors.by_group[1] == pytest.approx(0.5)


# --------------------------------------------------------------------------- theory check


def _fit_linear(loss_fn, x, steps=200):
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(x @ w + b)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def test_adjusted_loss_removes_prevalence_shortcut_in_well_specified_model():
    """p(x|y,g) = p(x_c|y) s(x_s|g), P(y=1|g) in {0.25, 0.75}.

    Plain BCE puts weight on the signature x_s. The adjusted loss's optimum is
    z = 2 * x_c + logit(0.5), i.e. zero weight on x_s, and its group-wise mean scores
    within each label match (so the score-matching penalty is ~0 there).
    """
    rng = np.random.default_rng(0)
    n = 40_000
    g = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(g == 1, 0.75, 0.25)).astype(np.int64)
    x_c = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)  # log LR = 2 * x_c
    x_s = rng.normal(np.where(g == 1, 1.0, -1.0), 0.5)  # label-independent signature
    x = torch.tensor(np.stack([x_c, x_s], 1), dtype=torch.float32)
    yt, gt = torch.tensor(y), torch.tensor(g, dtype=torch.float32)

    m = make_method(priors=priors_from_mapping({0: 0.25, 1: 0.75}, pooled=0.5))
    w_bce, _ = _fit_linear(lambda z: F.binary_cross_entropy_with_logits(z, yt.float()), x)
    w_la, b_la = _fit_linear(lambda z: m.la_loss(z, yt, gt), x)

    assert w_bce[1] > 0.5  # plain BCE exploits the signature
    assert abs(w_la[1].item()) < 0.05  # adjusted loss does not
    assert w_la[0].item() == pytest.approx(2.0, abs=0.1)
    assert abs(b_la.item()) < 0.05

    sm = ScoreMatchingLoss(min_subgroup_count=1)
    sm_bce = sm(torch.sigmoid(x @ w_bce), yt, gt).item()
    sm_la = sm(torch.sigmoid(x @ w_la + b_la), yt, gt).item()
    assert sm_la < 0.2 * sm_bce

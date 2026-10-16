"""Tests for the framework-free core (methods/logit_adjustment_core.py)."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from methods.logit_adjustment_core import (
    GroupClassPriors,
    GroupEncoder,
    GroupLogitAdjustedLoss,
    estimate_group_class_priors,
    resolve_reference_prior,
)

CORE = Path(__file__).resolve().parents[1] / "methods" / "logit_adjustment_core.py"


def test_core_imports_without_the_rest_of_the_repo(tmp_path):
    shutil.copy(CORE, tmp_path / "la_core.py")
    code = (
        f"import sys; sys.path.insert(0, {str(tmp_path)!r}); import la_core; "
        "assert 'config' not in sys.modules and 'methods' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# --------------------------------------------------------------------------- priors


def test_multiclass_priors_with_missing_and_weights():
    labels = np.array([0, 1, 2, 2, 0, 0, 1, 1, 2, 0])
    groups = np.array([0, 0, 0, 0, 3, 3, 3, 3, np.nan, -1])
    p = estimate_group_class_priors(labels, groups)
    assert p.num_classes == 3
    assert sorted(p.by_group) == [0, 3]  # ids need not be contiguous
    np.testing.assert_allclose(p.by_group[0], [0.25, 0.25, 0.5])
    np.testing.assert_allclose(p.by_group[3], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(p.missing, [0.5, 0.0, 0.5])
    np.testing.assert_allclose(p.pooled, [0.4, 0.3, 0.3])

    w = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    p_w = estimate_group_class_priors(labels, groups, sample_weights=w)
    np.testing.assert_allclose(p_w.by_group[0], [0.4, 0.2, 0.4])


def test_num_classes_validation():
    with pytest.raises(ValueError):
        estimate_group_class_priors([0, 3], [0, 0], num_classes=3)
    p = estimate_group_class_priors([0, 1], [0, 0], num_classes=4)  # absent classes ok
    assert p.num_classes == 4 and p.by_group[0][3] == 0.0


@pytest.mark.parametrize(
    "spec, expected",
    [("uniform", [1 / 3] * 3), ("pooled", [0.5, 0.3, 0.2]), ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])],
)
def test_reference_prior_resolution(spec, expected):
    np.testing.assert_allclose(
        resolve_reference_prior(spec, np.array([0.5, 0.3, 0.2]), 3), expected
    )


def test_reference_prior_errors():
    with pytest.raises(ValueError):
        resolve_reference_prior(0.3, np.array([0.5, 0.3, 0.2]), 3)  # scalar needs K=2
    with pytest.raises(ValueError):
        resolve_reference_prior([0.5, 0.6, 0.1], np.ones(3) / 3, 3)
    np.testing.assert_allclose(resolve_reference_prior("0.3", np.ones(2) / 2, 2), [0.7, 0.3])


# --------------------------------------------------------------------------- loss


def _toy_multiclass():
    priors = GroupClassPriors.from_rates(
        {0: [0.6, 0.3, 0.1], 1: [0.2, 0.3, 0.5]}, pooled=[0.4, 0.3, 0.3]
    )
    torch.manual_seed(0)
    logits = torch.randn(10, 3, requires_grad=True)
    y = torch.randint(0, 3, (10,))
    g = torch.tensor([0, 1] * 5, dtype=torch.float32)
    return priors, logits, y, g


@pytest.mark.parametrize("tau", [1.0, 0.5])
def test_multiclass_loss_matches_manual(tau):
    priors, logits, y, g = _toy_multiclass()
    loss_fn = GroupLogitAdjustedLoss(reference_prior="uniform", tau=tau, priors=priors)
    log_pi = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]))
    offsets = tau * (log_pi[g.long()] - torch.log(torch.tensor(1 / 3)))
    expected = F.cross_entropy(logits + offsets, y)
    loss = loss_fn(logits, y, g)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_binary_single_logit_equals_two_class_softmax():
    priors = GroupClassPriors.from_rates({0: 0.2, 1: 0.7, 2: 0.45}, pooled=0.4, missing=0.3)
    loss_fn = GroupLogitAdjustedLoss(reference_prior="pooled", priors=priors)
    torch.manual_seed(1)
    z = torch.randn(12)
    y = torch.randint(0, 2, (12,))
    g = torch.tensor([0, 1, 2, float("nan")] * 3)
    single = loss_fn(z, y, g)
    two = loss_fn(torch.stack([torch.zeros_like(z), z], 1), y, g)
    assert single.item() == pytest.approx(two.item(), rel=1e-6)
    assert loss_fn(z.view(-1, 1), y, g).item() == pytest.approx(single.item(), rel=1e-6)


def test_shape_and_class_mismatch_errors():
    priors, logits, y, g = _toy_multiclass()
    loss_fn = GroupLogitAdjustedLoss(priors=priors)
    with pytest.raises(ValueError, match="binary"):
        loss_fn(logits[:, 0], y, g)
    with pytest.raises(ValueError, match="shape"):
        loss_fn(torch.randn(10, 4), y, g)


def test_reduction_none_supports_external_weights_and_exclusion():
    priors, logits, y, _ = _toy_multiclass()
    g = torch.tensor([0, 1, float("nan")] * 3 + [0])
    loss_fn = GroupLogitAdjustedLoss(missing_group="exclude", priors=priors)
    per = loss_fn(logits, y, g, reduction="none")
    assert per.shape == (10,)
    assert (per[torch.isnan(g)] == 0).all()
    w = torch.rand(10)
    weighted = (per * w).sum() / w[~torch.isnan(g)].sum()
    weighted.backward()
    assert loss_fn(logits, y, g).item() == pytest.approx(per[~torch.isnan(g)].mean().item())


def test_retarget_is_saerens_correction():
    priors, logits, _, _ = _toy_multiclass()
    loss_fn = GroupLogitAdjustedLoss(reference_prior="uniform", priors=priors)
    target = np.array([0.7, 0.2, 0.1])
    p_ref = torch.softmax(logits.detach(), 1)
    expected = p_ref * torch.tensor(target / (1 / 3), dtype=torch.float32)
    expected = expected / expected.sum(1, keepdim=True)
    got = torch.softmax(loss_fn.retarget(logits.detach(), target), 1)
    assert torch.allclose(got, expected, atol=1e-6)

    bin_fn = GroupLogitAdjustedLoss(priors=GroupClassPriors.from_rates({0: 0.3}, pooled=0.3))
    z = torch.randn(5)
    p = torch.sigmoid(z)
    odds = p / (1 - p) * (0.1 / 0.9)
    assert torch.allclose(torch.sigmoid(bin_fn.retarget(z, 0.1)), odds / (1 + odds), atol=1e-6)


def test_zero_prior_is_clipped_with_warning(caplog):
    priors = GroupClassPriors.from_rates({0: [1.0, 0.0, 0.0], 1: [0.3, 0.3, 0.4]}, pooled=[0.6, 0.2, 0.2])
    with caplog.at_level(logging.WARNING):
        loss_fn = GroupLogitAdjustedLoss(priors=priors)
    assert "clipped" in caplog.text
    assert torch.isfinite(loss_fn.offset_table).all()


def test_multiclass_adjustment_removes_prevalence_shortcut():
    """3 classes, group-dependent class priors, label-independent group signature x_s.

    Plain CE puts class-specific weight on x_s; the adjusted loss's optimum does not.
    (Softmax weights are identified up to a shift common to all classes, so compare
    centred weights.)
    """
    rng = np.random.default_rng(0)
    n = 60_000
    pis = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    g = rng.integers(0, 3, n)
    y = np.array([rng.choice(3, p=pis[k]) for k in g])
    mu_c = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.2]])
    x_c = mu_c[y] + rng.normal(size=(n, 2))
    mu_s = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x_s = mu_s[g] + 0.4 * rng.normal(size=(n, 2))
    x = torch.tensor(np.hstack([x_c, x_s]), dtype=torch.float32)
    yt, gt = torch.tensor(y), torch.tensor(g, dtype=torch.float32)

    def fit(loss):
        W = torch.zeros(4, 3, requires_grad=True)
        b = torch.zeros(3, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            v = loss(x @ W + b)
            v.backward()
            return v

        opt.step(closure)
        W = W.detach()
        return W - W.mean(1, keepdim=True)

    la = GroupLogitAdjustedLoss().fit(y, g)
    W_ce = fit(lambda z: F.cross_entropy(z, yt))
    W_la = fit(lambda z: la(z, yt, gt))
    assert W_ce[2:].abs().max() > 0.3
    assert W_la[2:].abs().max() < 0.06
    # the disease-evidence weights are the true (centred) mean vectors
    centred_means = torch.tensor(mu_c.T - mu_c.T.mean(1, keepdims=True), dtype=torch.float32)
    assert torch.allclose(W_la[:2], centred_means, atol=0.08)


# --------------------------------------------------------------------------- encoder


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "site": ["CAM", "RSH", "CAM", "UMCU", "RSH", None, "CAM"],
            "sex": ["F", "F", "M", "F", "F", "M", np.nan],
            "vendor": ["GE", "Siemens", "GE", "GE", "Siemens", "GE", "GE"],
            "label": [0, 1, 2, 1, 0, 1, 2],
        }
    )


def test_encoder_intersection_ids_names_and_missing(frame):
    enc = GroupEncoder().fit(frame[["site", "sex", "vendor"]])
    assert enc.n_groups == 4
    assert enc.names == {
        0: "site=CAM|sex=F|vendor=GE",
        1: "site=CAM|sex=M|vendor=GE",
        2: "site=RSH|sex=F|vendor=Siemens",
        3: "site=UMCU|sex=F|vendor=GE",
    }
    ids = enc.transform(frame)  # extra columns are ignored for DataFrames
    np.testing.assert_array_equal(ids[:5], [0, 2, 1, 3, 2])
    assert np.isnan(ids[5]) and np.isnan(ids[6])
    assert enc.decode([2, np.nan]) == [("RSH", "F", "Siemens"), None]


def test_encoder_unseen_policies_and_int_missing(frame, caplog):
    enc = GroupEncoder().fit(frame[["site", "sex"]])
    new = pd.DataFrame({"site": ["CAM", "MHA"], "sex": ["F", "F"]})
    with caplog.at_level(logging.WARNING):
        out = enc.transform(new)
    assert out[0] == 0 and np.isnan(out[1]) and "unseen" in caplog.text
    with pytest.raises(ValueError, match="Unseen"):
        GroupEncoder(unseen="error").fit(frame[["site", "sex"]]).transform(new)
    ints = GroupEncoder(missing_value=-1).fit_transform(frame[["site", "sex"]])
    assert ints.dtype == np.int64 and ints[5] == -1
    with pytest.raises(ValueError):
        GroupEncoder(missing_value=3)


def test_encoder_input_forms_agree():
    site = np.array([2, 0, 1, 2, 0])
    sex = ["M", "F", "F", "M", "M"]
    a = GroupEncoder().fit_transform([site, sex])
    b = GroupEncoder().fit_transform({"site": site, "sex": sex})
    c = GroupEncoder().fit_transform(np.array([site, sex], dtype=object).T)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    single = GroupEncoder().fit_transform(pd.Series(site, name="site"))
    np.testing.assert_array_equal(single, [2, 0, 1, 2, 0])
    # 1 and 1.0 (e.g. after pandas casts an int column with NaNs to float) are one level
    np.testing.assert_array_equal(
        GroupEncoder().fit_transform([1, 1.0, 2.0, np.nan])[:3], [0, 0, 1]
    )
    with pytest.raises(ValueError, match="lengths"):
        GroupEncoder().fit([site, sex[:3]])


def test_intersectional_end_to_end(frame):
    enc = GroupEncoder().fit(frame[["site", "sex", "vendor"]])
    groups = enc.transform(frame)
    loss_fn = GroupLogitAdjustedLoss(missing_group="exclude").fit(frame["label"], groups)
    # cell priors are the empirical class frequencies of each intersection
    np.testing.assert_allclose(loss_fn.priors.by_group[2], [0.5, 0.5, 0.0])
    logits = torch.randn(len(frame), 3, requires_grad=True)
    loss = loss_fn(logits, torch.tensor(frame["label"].to_numpy()), torch.tensor(groups))
    loss.backward()
    text = loss_fn.describe(group_names=enc.names)
    assert "site=RSH|sex=F|vendor=Siemens" in text

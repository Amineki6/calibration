"""
DFR Stage 2: last-layer retraining on frozen features.

Stage 2 fits the classification head with sklearn's ``MLPClassifier`` and copies
the learned parameters back into the PyTorch head.  Fitting outside Lightning is
deliberate: the in-graph Stage 2 was capped at ~150 Adam steps at lr=1e-4 from a
Xavier init (289 rows / batch 64 * 30 epochs), which left the head statistically
indistinguishable from its initialization, and the evaluated weights were an EMA
lagging that head by ~10 epochs.  Fitting to convergence here removes the
learning-rate, epoch-budget and EMA dependencies from the method entirely.

The head architecture is identical to `StandardMethod`'s
(Linear(d, 512) -> ReLU -> Linear(512, 1)), so DFR is compared against the other
baselines at matched capacity.  Features are deliberately *not* standardized:
the baselines do not standardize either, and sklearn's adam solver is
per-parameter scale-adaptive.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

GROUP_NAMES = {0: "y0_d0", 1: "y0_d1", 2: "y1_d0", 3: "y1_d1"}


def is_backbone_frozen(config) -> bool:
    """
    True when ERM Stage 1 cannot change a single encoder weight.

    Foundation-model backbones are built frozen (`model._freeze`) and
    `CXPLightningModule._should_train_encoder` never unfreezes them; with
    `use_cached_features` the encoder is replaced by `nn.Identity` outright.
    Densenet is the one backbone that is actually finetuned -- note that
    `CXP_Model.__init__` ignores `use_cached_features` for densenet, so it is
    never frozen regardless of that flag.
    """
    return config.backbone != "densenet"


def _balanced_subset_indices(groups: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    """
    Group-balanced subsample: keep every row of the smallest group and randomly
    subsample the others down to the same count (DFR Step 2).

    `groups` holds positional indices into the reweighting split; the returned
    indices are positional too.
    """
    rng = np.random.default_rng(seed)
    present = np.unique(groups)
    if len(present) < 4:
        logging.warning(
            "DFR Stage 2: only %d/4 subgroups present in the reweighting split "
            "(ids=%s); balancing across the groups that are present.",
            len(present),
            present.tolist(),
        )

    n_per_group = int(min((groups == g).sum() for g in present))
    picked = [
        rng.choice(np.flatnonzero(groups == g), size=n_per_group, replace=False)
        for g in present
    ]
    return np.sort(np.concatenate(picked)), n_per_group


@torch.no_grad()
def _extract_features(encoder: nn.Module, dataset, config, device):
    """Run the frozen encoder over `dataset` and return (features, labels, drains)."""
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    was_training = encoder.training
    encoder = encoder.to(device)
    encoder.eval()

    feats, labels, drains = [], [], []
    for batch in loader:
        out = encoder(batch.inputs.to(device))
        feats.append(out.float().cpu())
        labels.append(batch.labels.cpu())
        drains.append(batch.drains.cpu())

    encoder.train(was_training)

    return (
        torch.cat(feats).numpy().astype(np.float64),
        torch.cat(labels).numpy().astype(np.int64),
        torch.cat(drains).numpy(),
    )


def _linear_layers(head: nn.Module) -> list[nn.Linear]:
    layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
    if len(layers) != 2:
        raise RuntimeError(
            f"DFR Stage 2 expects a two-Linear head to mirror StandardMethod, "
            f"found {len(layers)} Linear layers in {head}."
        )
    return layers


def _copy_sklearn_mlp_into_head(sk_clf, head: nn.Module) -> None:
    """
    Copy `MLPClassifier` parameters into the PyTorch head.

    sklearn stores weights as (fan_in, fan_out) and torch as (fan_out, fan_in),
    hence the transposes.  For a binary target `MLPClassifier` uses a single
    logistic output unit, so `coefs_[1] @ h + intercepts_[1]` is exactly the
    logit consumed by `binary_cross_entropy_with_logits` downstream -- no sign
    or scale conversion is needed.
    """
    if list(sk_clf.classes_) != [0, 1]:
        raise RuntimeError(
            f"Expected sklearn classes_ == [0, 1] so that the output unit scores "
            f"the positive class, got {sk_clf.classes_!r}."
        )
    if sk_clf.out_activation_ != "logistic":
        raise RuntimeError(
            f"Expected a logistic output unit (raw score == logit), got "
            f"out_activation_={sk_clf.out_activation_!r}."
        )
    if len(sk_clf.coefs_) != 2:
        raise RuntimeError(
            f"Expected a single hidden layer, got {len(sk_clf.coefs_)} weight matrices."
        )

    fc1, fc2 = _linear_layers(head)
    w1 = torch.from_numpy(sk_clf.coefs_[0].T).float()
    b1 = torch.from_numpy(sk_clf.intercepts_[0]).float()
    w2 = torch.from_numpy(sk_clf.coefs_[1].T).float()
    b2 = torch.from_numpy(sk_clf.intercepts_[1]).float()

    for name, tensor, param in (
        ("fc1.weight", w1, fc1.weight),
        ("fc1.bias", b1, fc1.bias),
        ("fc2.weight", w2, fc2.weight),
        ("fc2.bias", b2, fc2.bias),
    ):
        if param is None:
            raise RuntimeError(f"DFR Stage 2: head is missing {name}.")
        if tuple(tensor.shape) != tuple(param.shape):
            raise RuntimeError(
                f"DFR Stage 2 shape mismatch for {name}: sklearn gives "
                f"{tuple(tensor.shape)}, head expects {tuple(param.shape)}."
            )

    with torch.no_grad():
        fc1.weight.copy_(w1)
        fc1.bias.copy_(b1)
        fc2.weight.copy_(w2)
        fc2.bias.copy_(b2)


def _worst_group_accuracy(logits: np.ndarray, y: np.ndarray, groups: np.ndarray):
    per_group = {}
    for g in np.unique(groups):
        mask = groups == g
        per_group[GROUP_NAMES.get(int(g), str(int(g)))] = float(
            ((logits[mask] > 0.0).astype(np.int64) == y[mask]).mean()
        )
    return min(per_group.values()), per_group


def run_dfr_stage2(config, model, datamodule) -> dict:
    """
    Fit the DFR head on a group-balanced subsample of the reweighting split and
    write the result into `model.clf` in place.

    `datamodule.setup("fit")` must already have run: `datamodule.train_dataset`
    is the head-training half of the validation split produced by
    `CXPDataModule._split_dfr_pool`, and is used here unchanged.

    Returns a diagnostics dict (also logged).
    """
    from sklearn.neural_network import MLPClassifier

    # Imported lazily so this module stays importable (and unit-testable)
    # without pulling in Lightning.
    from lightning_datamodule import _subgroup_ids

    split = datamodule.train_dataset
    if split is None:
        raise RuntimeError(
            "DFR Stage 2: datamodule.train_dataset is None; call setup('fit') first."
        )

    groups, drain = _subgroup_ids(split)
    if np.isnan(drain).any():
        raise RuntimeError(
            "DFR Stage 2 needs subgroup labels on every reweighting row, but the "
            "split contains Drain=NaN rows."
        )

    balanced_idx, n_per_group = _balanced_subset_indices(groups, config.dfr_split_seed)
    logging.info(
        "DFR Stage 2 reweighting set: %d rows from a %d-row split "
        "(%d per subgroup; original counts=%s).",
        len(balanced_idx),
        len(groups),
        n_per_group,
        np.bincount(groups, minlength=4).tolist(),
    )

    balanced_split = Subset(split, balanced_idx.tolist())
    device = config.device
    X, y, _ = _extract_features(model.encoder, balanced_split, config, device)
    balanced_groups = groups[balanced_idx]

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "DFR Stage 2 reweighting set contains a single class; cannot fit a head."
        )

    logging.info(
        "DFR Stage 2: fitting MLPClassifier(hidden_layer_sizes=(512,), relu) on "
        "X%s, alpha=%.3g, max_iter=%d (no feature standardization by design).",
        X.shape,
        config.dfr_stage2_weight_decay,
        config.dfr_stage2_max_iter,
    )

    sk_clf = MLPClassifier(
        hidden_layer_sizes=(512,),
        activation="relu",
        solver="adam",
        alpha=config.dfr_stage2_weight_decay,
        max_iter=config.dfr_stage2_max_iter,
        n_iter_no_change=config.dfr_stage2_n_iter_no_change,
        tol=config.dfr_stage2_tol,
        random_state=config.dfr_split_seed,
        shuffle=True,
        verbose=False,
    )
    sk_clf.fit(X, y)

    converged = sk_clf.n_iter_ < config.dfr_stage2_max_iter
    if not converged:
        logging.warning(
            "DFR Stage 2: MLPClassifier hit max_iter=%d without meeting tol=%.3g "
            "(final loss=%.5f). Raise dfr_stage2_max_iter.",
            config.dfr_stage2_max_iter,
            config.dfr_stage2_tol,
            sk_clf.loss_,
        )

    head_dim = _linear_layers(model.clf)[0].in_features
    if X.shape[1] != head_dim:
        raise RuntimeError(
            f"DFR Stage 2: encoder emitted {X.shape[1]}-d features but the head "
            f"expects {head_dim}-d."
        )
    _copy_sklearn_mlp_into_head(sk_clf, model.clf)

    # Verify the transplant end-to-end on the fitting data: the torch head must
    # reproduce sklearn's predicted probabilities to floating-point tolerance.
    # (MLPClassifier exposes no decision_function, and comparing probabilities
    # avoids the blow-up of log-odds at saturated predictions.)
    with torch.no_grad():
        torch_logits = (
            model.clf(torch.from_numpy(X).float())
            .view(-1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    torch_probs = 1.0 / (1.0 + np.exp(-torch_logits))
    sk_probs = sk_clf.predict_proba(X)[:, 1].astype(np.float64)
    max_abs_diff = float(np.abs(torch_probs - sk_probs).max())
    if max_abs_diff > 1e-4:
        raise RuntimeError(
            f"DFR Stage 2 weight transplant failed: torch head and sklearn "
            f"predict_proba differ by up to {max_abs_diff:.3e}."
        )

    train_acc = float(((torch_logits > 0.0).astype(np.int64) == y).mean())
    worst_acc, per_group = _worst_group_accuracy(torch_logits, y, balanced_groups)
    logging.info(
        "DFR Stage 2 complete: n_iter=%d, converged=%s, loss=%.5f, "
        "reweighting-set acc=%.4f, worst-group acc=%.4f %s "
        "(transplant max|Δp|=%.2e).",
        sk_clf.n_iter_,
        converged,
        sk_clf.loss_,
        train_acc,
        worst_acc,
        per_group,
        max_abs_diff,
    )

    return {
        "n_reweighting_rows": int(len(balanced_idx)),
        "n_per_group": int(n_per_group),
        "n_iter": int(sk_clf.n_iter_),
        "converged": bool(converged),
        "final_loss": float(sk_clf.loss_),
        "fit_accuracy": train_acc,
        "fit_worst_group_accuracy": float(worst_acc),
        "transplant_max_abs_prob_diff": max_abs_diff,
    }

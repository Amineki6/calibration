import logging

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torch
from typing import Optional
from pathlib import Path

from config import ExperimentConfig
from dataset import CXP_dataset
from utils import get_jtt_loader


def _resolve_base_and_indices(dataset):
    """Unwrap nested Subsets into (base_dataset, absolute row indices)."""
    ds = dataset
    idx = None
    while isinstance(ds, Subset):
        parent_idx = np.asarray(ds.indices, dtype=np.int64)
        idx = parent_idx if idx is None else parent_idx[idx]
        ds = ds.dataset
    if idx is None:
        idx = np.arange(len(ds), dtype=np.int64)
    return ds, idx


def _subgroup_ids(dataset):
    """(Pneumothorax, Drain) subgroup id per row exposed by `dataset`."""
    base, idx = _resolve_base_and_indices(dataset)
    labels = base.labels.to_numpy()[idx].astype(np.int64)
    drain = base.drain.to_numpy()[idx]
    # NaN drain rows get a placeholder id; callers filter them via the returned
    # drain array (casting NaN to int directly is undefined).
    drain_int = np.where(np.isnan(drain), 0.0, drain).astype(np.int64)
    return labels * 2 + drain_int, drain


def _group_sample_weights(dataset) -> torch.Tensor:
    """Inverse-frequency weights over the four (Pneumothorax, Drain) subgroups."""
    groups, drain = _subgroup_ids(dataset)
    if np.isnan(drain).any():
        raise ValueError(
            "Group balancing requires non-missing Drain labels, but the split "
            "contains Drain=NaN rows."
        )
    counts = np.bincount(groups, minlength=4).astype(np.float32)
    inv = np.zeros_like(counts)
    inv[counts > 0] = 1.0 / counts[counts > 0]
    return torch.from_numpy(inv[groups])


class CXPDataModule(pl.LightningDataModule):
    def __init__(self, config: ExperimentConfig, debug: bool = False):
        super().__init__()
        self.config = config
        self.debug = debug
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset_aligned = None
        self.test_dataset_misaligned = None
        self.train_sampler = None
        self.jtt_error_indices = None  # For JTT Stage 2
        # DFR Stage 2 balances its own (val-derived) head-training split,
        # independently of config.balance_train.
        self._dfr_stage2_balance = False

    def _split_dfr_pool(self, pool):
        """
        Stratified split of the val pool into a head-training half and a
        disjoint selection half. Rows without a Drain label are dropped: DFR
        needs subgroup annotations on both halves.
        """
        groups, drain = _subgroup_ids(pool)
        known = ~np.isnan(drain)
        if not known.all():
            logging.info(
                "DFR: dropping %d/%d val rows with missing Drain labels.",
                int((~known).sum()),
                len(known),
            )

        rng = np.random.default_rng(self.config.dfr_split_seed)
        train_idx, select_idx = [], []
        for g in np.unique(groups[known]):
            g_idx = np.flatnonzero(known & (groups == g))
            rng.shuffle(g_idx)
            n_train = int(round(len(g_idx) * self.config.dfr_stage2_train_fraction))
            n_train = min(max(n_train, 1), max(len(g_idx) - 1, 1))
            train_idx.append(g_idx[:n_train])
            select_idx.append(g_idx[n_train:])

        train_idx = np.sort(np.concatenate(train_idx))
        select_idx = np.sort(np.concatenate(select_idx))
        logging.info(
            "DFR Stage 2 split: %d head-training rows, %d selection rows (disjoint).",
            len(train_idx),
            len(select_idx),
        )
        return Subset(pool, train_idx.tolist()), Subset(pool, select_idx.tolist())

    def setup(self, stage: Optional[str] = None):
        csv_dir = Path(self.config.csv_dir)
        data_dir = Path(self.config.data_dir)
        features_dir = (
            Path(self.config.features_dir) if self.config.features_dir else None
        )

        if stage == "fit" or stage is None:
            if self.config.balance_val:
                train_csv = csv_dir / "train_drain_shortcut_v2.csv"
                val_csv = csv_dir / "val_drain_shortcut_v2.csv"
            else:
                train_csv = csv_dir / "train_drain_shortcut.csv"
                val_csv = csv_dir / "val_drain_shortcut.csv"

            if self.config.method == "dfr":
                # Stage 2 retrains the head on the val split, so the val pool is
                # halved: one half trains the head, the other selects the
                # checkpoint. Both come from one dataset instance (no augment).
                dfr_pool = CXP_dataset(
                    data_dir,
                    val_csv,
                    augment=False,
                    compute_sample_weights=True,
                    split_name="dfr_stage2_pool",
                    backbone=self.config.backbone,
                    use_cached_features=self.config.use_cached_features,
                    features_dir=features_dir,
                )
                self.train_dataset, self.val_dataset = self._split_dfr_pool(dfr_pool)
                self._dfr_stage2_balance = True
            else:
                self.train_dataset = CXP_dataset(
                    data_dir,
                    train_csv,
                    augment=True,
                    compute_sample_weights=self.config.balance_train,
                    split_name="train",
                    backbone=self.config.backbone,
                    use_cached_features=self.config.use_cached_features,
                    features_dir=features_dir,
                )
                self.val_dataset = CXP_dataset(
                    data_dir,
                    val_csv,
                    augment=False,
                    compute_sample_weights=True,
                    split_name="val",
                    backbone=self.config.backbone,
                    use_cached_features=self.config.use_cached_features,
                    features_dir=features_dir,
                )

        if stage == "test" or stage is None:
            self.test_dataset_aligned = CXP_dataset(
                data_dir,
                csv_dir / "test_drain_shortcut_aligned.csv",
                augment=False,
                compute_sample_weights=False,
                split_name="test_aligned",
                backbone=self.config.backbone,
                use_cached_features=self.config.use_cached_features,
                features_dir=features_dir,
            )
            self.test_dataset_misaligned = CXP_dataset(
                data_dir,
                csv_dir / "test_drain_shortcut_misaligned.csv",
                augment=False,
                compute_sample_weights=False,
                split_name="test_misaligned",
                backbone=self.config.backbone,
                use_cached_features=self.config.use_cached_features,
                features_dir=features_dir,
            )

        if self.debug:

            def fast_subset(ds, n=50):
                if ds is None:
                    return None
                return torch.utils.data.Subset(ds, range(min(len(ds), n)))

            if hasattr(self, "train_dataset"):
                self.train_dataset = fast_subset(self.train_dataset, 50)
            if hasattr(self, "val_dataset"):
                self.val_dataset = fast_subset(self.val_dataset, 20)
            if hasattr(self, "test_dataset_aligned"):
                self.test_dataset_aligned = fast_subset(self.test_dataset_aligned, 20)
            if hasattr(self, "test_dataset_misaligned"):
                self.test_dataset_misaligned = fast_subset(
                    self.test_dataset_misaligned, 20
                )

            self.config.balance_train = False  # specific to debug mode logic in utils

        # Built last, so the weights always match whatever train_dataset ended up
        # as (debug subset, DFR Stage-2 half, ...).
        needs_balancing = self._dfr_stage2_balance or (
            self.config.balance_train and not self.jtt_error_indices
        )
        if needs_balancing:
            sample_weights = _group_sample_weights(self.train_dataset)
            self.train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )
        else:
            self.train_sampler = None

    def train_dataloader(self):
        # Handle JTT Stage 2
        if self.jtt_error_indices is not None:
            return get_jtt_loader(
                self.train_dataset, self.jtt_error_indices, self.config
            )

        prefetch_factor = 2 if self.config.num_workers > 0 else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            sampler=self.train_sampler,
        )

    def val_dataloader(self):
        prefetch_factor = 2 if self.config.num_workers > 0 else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )

    def test_dataloader(self):
        prefetch_factor = 2 if self.config.num_workers > 0 else None
        loader_aligned = DataLoader(
            self.test_dataset_aligned,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
        loader_misaligned = DataLoader(
            self.test_dataset_misaligned,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
        )
        return [loader_aligned, loader_misaligned]

    def set_jtt_error_indices(self, error_indices):
        self.jtt_error_indices = error_indices
        self.train_sampler = None

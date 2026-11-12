import torch
import torch.nn as nn
from typing import Optional

from .base import BaseMethod
from config import ExperimentConfig

MISSING_GROUP_ID = -1


class DatasetScoreMatchingLoss(nn.Module):
    """
    Dataset-level score matching loss that maintains a persistent buffer of predictions
    across the entire dataset. Computes standard deviation using all available predictions but
    only backpropagates gradients through the current batch.

    Binary (num_classes=2): probs are positive-class probabilities of shape (B,) or (B, 1).
    Multiclass (num_classes=K>2): probs are class probabilities of shape (B, K). For each true
    class c, the std across groups of the mean score vector is averaged over the K outputs.
    """

    score_buffer: torch.Tensor
    label_buffer: torch.Tensor
    group_buffer: torch.Tensor

    def __init__(
        self,
        dataset_size: int,
        min_subgroup_count: int = 10,
        device: torch.device | str = "cuda",
        num_classes: int = 2,
    ):
        super().__init__()
        self.min_subgroup_count = max(1, min_subgroup_count)
        self.num_classes = num_classes
        score_shape = (dataset_size,) if num_classes == 2 else (dataset_size, num_classes)

        # Persistent buffers (not parameters, won't be updated by optimizer)
        self.register_buffer(
            "score_buffer", torch.full(score_shape, float("nan"), device=device)
        )
        self.register_buffer(
            "label_buffer",
            torch.full((dataset_size,), -1, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "group_buffer",
            torch.full(
                (dataset_size,), MISSING_GROUP_ID, dtype=torch.long, device=device
            ),
        )

    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss using full dataset buffer but only backprop through current batch.

        Args:
            probs: (B,) or (B, 1) predicted probabilities for current batch; (B, K) if K>2
            labels: (B,) true class labels (0..num_classes-1) for current batch
            groups: (B,) group indicators for current batch
            indices: (B,) dataset indices for current batch samples

        Returns:
            Scalar loss tensor with gradients only w.r.t. current batch predictions
        """
        # Normalize shapes
        if probs.dim() == 2 and self.num_classes == 2:
            assert probs.shape[1] == 1
            probs = probs.squeeze(1)

        B = probs.shape[0]
        assert probs.shape[1:] == self.score_buffer.shape[1:]
        assert labels.shape == (B,) and groups.shape == (B,) and indices.shape == (B,)

        # Keep assignment dtypes consistent with persistent buffers.
        probs = probs.to(device=self.score_buffer.device, dtype=self.score_buffer.dtype)
        indices = indices.to(device=self.score_buffer.device, dtype=torch.long)
        labels = labels.to(
            device=self.label_buffer.device, dtype=self.label_buffer.dtype
        )
        groups = groups.to(device=self.group_buffer.device, dtype=torch.float32)
        assert not torch.any(groups == float(MISSING_GROUP_ID)), (
            "Incoming groups contain -1, which is reserved as internal missing-group sentinel."
        )
        groups = torch.nan_to_num(groups, nan=float(MISSING_GROUP_ID))
        groups = groups.to(dtype=self.group_buffer.dtype)

        # Start with detached buffer, then inject gradient-enabled batch predictions
        buffer_scores = self.score_buffer.clone()
        buffer_scores[indices] = probs

        buffer_labels = self.label_buffer.clone()
        buffer_labels[indices] = labels

        buffer_groups = self.group_buffer.clone()
        buffer_groups[indices] = groups

        # Check valid entries (all three buffers must be valid) AFTER inserting batch data
        valid_mask = (
            ~torch.isnan(buffer_scores).reshape(len(buffer_scores), -1).any(dim=1)
            & (buffer_labels >= 0)
            & (buffer_groups != MISSING_GROUP_ID)
        )
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype).detach()

        # Extract only valid entries for computation
        buffer_scores = buffer_scores[valid_mask]
        buffer_labels = buffer_labels[valid_mask]
        buffer_groups = buffer_groups[valid_mask]

        # Per true class: std across groups of the group-wise average scores
        unique_groups = buffer_groups.unique()
        class_stds = []

        for c in range(self.num_classes):
            group_avgs = []
            for g in unique_groups:
                mask = (buffer_groups == g) & (buffer_labels == c)
                if mask.sum() >= self.min_subgroup_count:
                    group_avgs.append(buffer_scores[mask].mean(dim=0))
            if len(group_avgs) >= 2:
                class_stds.append(torch.stack(group_avgs).std(dim=0).mean())

        if class_stds:
            return torch.stack(class_stds).mean()
        return torch.tensor(0.0, device=probs.device, dtype=probs.dtype).detach()

    def update(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        Update buffers after optimizer step (since predictions changed).

        Args:
            probs: (B,) or (B, 1) updated predictions for current batch; (B, K) if K>2
            labels: (B,) true class labels (0..num_classes-1) for current batch
            groups: (B,) group indicators for current batch
            indices: (B,) dataset indices for current batch samples
        """
        if probs.dim() == 2 and self.num_classes == 2:
            probs = probs.squeeze(1)

        probs = probs.to(device=self.score_buffer.device, dtype=self.score_buffer.dtype)
        indices = indices.to(device=self.score_buffer.device, dtype=torch.long)
        labels = labels.to(
            device=self.label_buffer.device, dtype=self.label_buffer.dtype
        )
        groups = groups.to(device=self.group_buffer.device, dtype=torch.float32)
        assert not torch.any(groups == float(MISSING_GROUP_ID)), (
            "Incoming groups contain -1, which is reserved as internal missing-group sentinel."
        )
        groups = torch.nan_to_num(groups, nan=float(MISSING_GROUP_ID))
        groups = groups.to(dtype=self.group_buffer.dtype)

        with torch.no_grad():
            self.score_buffer[indices] = probs.detach()
            self.label_buffer[indices] = labels.detach()
            self.group_buffer[indices] = groups.detach()

    def reset_buffer(self) -> None:
        """Reset buffers (e.g., at start of new epoch if desired)."""
        self.score_buffer.fill_(float("nan"))
        self.label_buffer.fill_(-1)
        self.group_buffer.fill_(MISSING_GROUP_ID)


class DatasetScoreMatchingMethod(BaseMethod):
    def __init__(self, config: ExperimentConfig, dataset_size: Optional[int] = None):
        super().__init__(config)
        self.dataset_size = dataset_size
        if dataset_size is not None:
            self.score_matching_loss = DatasetScoreMatchingLoss(
                min_subgroup_count=getattr(
                    config, "dataset_score_matching_min_subgroup_count", 10
                ),
                dataset_size=dataset_size,
                device=config.device,
            )
        else:
            self.score_matching_loss = None

        # Default lambda is 1.0 if not specified in config
        self.lambda_val = getattr(config, "dataset_score_matching_lambda", 1.0)

    def get_model_components(
        self, num_features: int
    ) -> tuple[nn.Module, Optional[nn.Module]]:
        # Score matching only needs a classification head.
        clf = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, 1))
        # We return None for the projection head because we don't use it.
        return clf, None

    def compute_loss(
        self,
        model_output: tuple[torch.Tensor, Optional[torch.Tensor]],
        targets: torch.Tensor,
        extra_info: Optional[dict] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculates Total Loss = BCE + Lambda * ScoreMatchingLoss
        Uses 'extra_info' to access the Drain labels.
        """
        assert extra_info is not None
        assert "drain" in extra_info.keys()
        assert self.score_matching_loss is not None

        logits, _ = model_output

        # 1. Classification Loss (Standard)
        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        cls_loss = bce_loss  # Always use unweighted BCE as the primary optimization objective to match training

        # 2. Score matching loss
        score_matching_val = self.score_matching_loss(
            probs=torch.sigmoid(logits.view(-1)),
            labels=targets,
            groups=extra_info["drain"],
            indices=extra_info["indices"],
        )

        total_loss = cls_loss + self.lambda_val * score_matching_val

        return total_loss, {
            "bce": bce_loss.item(),
            "wbce": wbce_loss.item(),
            "dataset_score_matching_loss": score_matching_val.item(),
        }

    def update_loss(
        self,
        model_output: tuple[torch.Tensor, Optional[torch.Tensor]],
        targets: torch.Tensor,
        extra_info: Optional[dict] = None,
        weight: Optional[torch.Tensor] = None,
    ):

        assert self.score_matching_loss is not None
        assert extra_info is not None
        assert "drain" in extra_info.keys()
        assert "indices" in extra_info.keys()

        logits, _ = model_output

        self.score_matching_loss.update(
            probs=torch.sigmoid(logits.view(-1)),
            labels=targets,
            groups=extra_info["drain"],
            indices=extra_info["indices"],
        )

    def clone(self, dataset_size: Optional[int] = None):
        """
        Creates a copy with a fresh loss instance, optionally with a new dataset size.

        Args:
            dataset_size: New dataset size for the cloned instance.
                         If None, uses the same size as the original.

        Returns:
            DatasetScoreMatchingMethod: A new instance with fresh buffers.
        """
        new_dataset_size = (
            dataset_size if dataset_size is not None else self.dataset_size
        )

        # Create a new instance with potentially different dataset size and newly instantiated loss
        cloned = DatasetScoreMatchingMethod(self.config, new_dataset_size)

        # Copy over lambda_val in case it was modified after initialization
        cloned.lambda_val = self.lambda_val

        return cloned

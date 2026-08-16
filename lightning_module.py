import torch
import torch.optim as optim
import pytorch_lightning as pl
import logging
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torcheval.metrics import BinaryAUROC
from typing import Any, Optional

from config import ExperimentConfig
from model import CXP_Model
from scoring import (
    compute_group_fairness_score,
    compute_saab_robust_auroc,
    compute_worst_group_accuracy,
)


class CXPLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: CXP_Model,
        method,
        config: ExperimentConfig,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "method"])
        self.model: CXP_Model = model
        self.config = config
        # Linear-probe mode (DFR Stage 2): the encoder stays frozen AND in eval()
        # for the whole run, so BatchNorm running stats cannot drift under the head.
        self.freeze_encoder = freeze_encoder

        # Clone method templates for train/val/test to maintain separate loss states.
        # Note: We must pass `dataset_size` during cloning because certain methods
        # (e.g., DatasetScoreMatchingMethod) require it to initialize dataset-sized
        # persistent buffers.
        #
        # This cloning occurs in setup() rather than __init__() because the
        # DataModule (and thus the exact dataset sizes) is not fully attached
        # or initialized until this phase of the PyTorch Lightning lifecycle.

        self.method_template = method
        self.train_method: Optional[Any] = None
        self.val_method: Optional[Any] = None
        self.test_method_aligned: Optional[Any] = None
        self.test_method_misaligned: Optional[Any] = None

        # EMA Model
        self.ema_model = AveragedModel(
            self.model,
            multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay),
            use_buffers=True,
        )

        # Metrics
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.val_wauroc = BinaryAUROC()

        # Test Metrics
        self.test_auroc_aligned = BinaryAUROC()
        self.test_auroc_misaligned = BinaryAUROC()

        # Storage for validation epoch outputs to compute fairness
        self.validation_step_outputs = []

        # Storage for test outputs
        self.test_aligned_outputs = []
        self.test_misaligned_outputs = []
        self._encoder_trainable: Optional[bool] = None

        self.automatic_optimization = True

    def _should_train_encoder(self) -> bool:
        if self.freeze_encoder:
            return False
        return self.config.backbone == "densenet" and self.current_epoch >= 5

    def _sync_encoder_trainability(self) -> None:
        encoder = self.model.encoder
        should_train_encoder = self._should_train_encoder()
        encoder.train(mode=should_train_encoder)

        if self._encoder_trainable != should_train_encoder:
            for param in encoder.parameters():
                param.requires_grad = should_train_encoder
            self._encoder_trainable = should_train_encoder

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and hasattr(self, "model"):
            self._sync_encoder_trainability()
        return self

    def _get_dataset_size(self, dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "dataset"):
            return self._get_dataset_size(dataset.dataset)
        return len(dataset)

    def setup(self, stage: str):
        datamodule = getattr(self.trainer, "datamodule", None)

        # "validate" is included for DFR, whose Stage 2 is fitted outside the
        # Lightning loop and therefore runs validation without ever fitting.
        if stage in ("fit", "validate"):
            train_size = self._get_dataset_size(
                getattr(datamodule, "train_dataset", None)
            )
            val_size = self._get_dataset_size(getattr(datamodule, "val_dataset", None))

            self.train_method = self.method_template.clone(dataset_size=train_size)
            self.val_method = self.method_template.clone(dataset_size=val_size)

        if stage == "test":
            test_aligned_size = self._get_dataset_size(
                getattr(datamodule, "test_dataset_aligned", None)
            )
            test_misaligned_size = self._get_dataset_size(
                getattr(datamodule, "test_dataset_misaligned", None)
            )

            self.test_method_aligned = self.method_template.clone(
                dataset_size=test_aligned_size
            )
            self.test_method_misaligned = self.method_template.clone(
                dataset_size=test_misaligned_size
            )

    def on_train_epoch_start(self):
        self._sync_encoder_trainability()
        for method in [self.train_method, self.val_method, self.method_template]:
            if method is not None and hasattr(method, "set_epoch"):
                method.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx):
        batch_data = batch

        if self.freeze_encoder:
            # Belt and braces: keep BN/dropout in inference mode even if some
            # other hook flipped the encoder back to train().
            self.model.encoder.eval()

        model_output = self.model(batch_data.inputs)
        extra_info = batch_data.extra_info()

        if self.train_method is None:
            self.train_method = self.method_template.clone()

        train_method = self.train_method
        assert train_method is not None

        loss, components = train_method.compute_loss(
            model_output,
            batch_data.labels,
            extra_info=extra_info,
            weight=batch_data.usable_weights,
        )

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce", components.get("bce", 0.0), on_step=False, on_epoch=True)

        logits, _ = model_output
        flat_logits = logits.reshape(-1)
        self.train_auroc.update(flat_logits.detach(), batch_data.labels.detach())

        probs = torch.sigmoid(flat_logits)
        brier = ((probs - batch_data.labels.float()) ** 2).sum()
        self.log(
            "train/brier",
            brier,
            on_step=False,
            on_epoch=True,
            metric_attribute="train_brier",
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Handle method.update_loss if needed (e.g. for dataset_score_matching)
        if self.train_method is None or not hasattr(self.train_method, "update_loss"):
            return

        # Re-run forward pass as done in utils.py
        batch_data = batch

        extra_info = batch_data.extra_info()

        with torch.no_grad():
            model_output = self.model(batch_data.inputs)
            self.train_method.update_loss(
                model_output,
                batch_data.labels,
                extra_info=extra_info,
                weight=batch_data.usable_weights,
            )

    def on_train_epoch_end(self):
        # Update EMA Model
        self.ema_model.update_parameters(self.model)

        # Log Train AUROC
        self.log("train/auroc", self.train_auroc.compute(), on_epoch=True)
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        # Validation uses EMA model
        batch_data = batch

        # Determine method implementation
        method = self.val_method if self.val_method else self.method_template

        # Use EMA model for validation/eval
        self.ema_model.eval()
        logits, projections = self.ema_model(batch_data.inputs)

        extra_info = batch_data.extra_info()

        # Calculate Loss
        loss, components = method.compute_loss(
            (logits, projections),
            batch_data.labels,
            weight=batch_data.usable_weights,
            extra_info=extra_info,
        )

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bce", components.get("bce", 0.0), on_step=False, on_epoch=True)
        self.log("val/wbce", components.get("wbce", 0.0), on_step=False, on_epoch=True)

        # Metrics - Move to CPU for stability on MPS
        self.val_auroc.update(logits.view(-1).cpu(), batch_data.labels.cpu())

        if batch_data.usable_weights is not None:
            self.val_wauroc.update(
                logits.view(-1).cpu(),
                batch_data.labels.cpu(),
                weight=batch_data.usable_weights.cpu(),
            )

        # Store for epoch-end Fairness computation
        self.validation_step_outputs.append(
            {
                "logits": logits.view(-1).detach().cpu(),
                "labels": batch_data.labels.detach().cpu(),
                "drains": batch_data.drains.detach().cpu(),
            }
        )

        return loss

    def on_validation_epoch_end(self):
        # Compute Fairness
        if not self.validation_step_outputs:
            return

        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        all_drains = torch.cat([x["drains"] for x in self.validation_step_outputs])

        # Scope labels make fairness warnings actionable in logs by indicating
        # exactly which stage produced them; they do not affect the metric value.
        fairness_score, fairness_details = compute_group_fairness_score(
            all_logits,
            all_labels,
            all_drains,
            scope="validation_epoch_aggregated",
            power=self.config.fairness_power,
        )

        self.log("val/fairness_score", fairness_score, on_epoch=True)

        if "summary" in fairness_details:
            self.log(
                "val/fairness_mean_pairwise_auroc",
                fairness_details["summary"]["mean_pairwise_auroc"],
                on_epoch=True,
            )
            self.log(
                "val/fairness_min_pairwise_auroc",
                fairness_details["summary"]["min_pairwise_auroc"],
                on_epoch=True,
            )
            self.log(
                "val/fairness_arithmetic_mean",
                fairness_details["summary"].get("arithmetic_mean", 0.0),
                on_epoch=True,
            )
            self.log(
                "val/fairness_geometric_mean",
                fairness_details["summary"].get("geometric_mean", 0.0),
                on_epoch=True,
            )
            self.log(
                "val/fairness_harmonic_mean",
                fairness_details["summary"].get("harmonic_mean", 0.0),
                on_epoch=True,
            )

        if "pairwise" in fairness_details:
            for pair_key, pair_info in fairness_details["pairwise"].items():
                if "auroc" in pair_info:
                    self.log(
                        f"val/pairwise_auroc_{pair_key}",
                        pair_info["auroc"],
                        on_epoch=True,
                    )

        # JTT-style worst-group accuracy: intersectional groups (y, a) encoded as label*2 + drain
        intersectional_groups = all_labels.float() * 2 + all_drains.float()
        worst_acc, _ = compute_worst_group_accuracy(
            all_logits, all_labels, intersectional_groups
        )
        self.log("val/worst_group_accuracy", worst_acc, on_epoch=True)

        self.log("val/auroc", self.val_auroc.compute(), on_epoch=True)

        if self.config.balance_val:
            self.log("val/wauroc", self.val_wauroc.compute(), on_epoch=True)

        self.val_auroc.reset()
        self.val_wauroc.reset()
        self.validation_step_outputs.clear()  # Free memory

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Using EMA model for testing
        self.ema_model.eval()
        batch_data = batch

        logits, projections = self.ema_model(batch_data.inputs)
        extra_info = batch_data.extra_info()

        # Note: We need to differentiate aligned vs misaligned.
        # PL handles multiple test dataloaders by passing dataloader_idx

        if dataloader_idx == 0:
            # Aligned
            method = (
                self.test_method_aligned
                if self.test_method_aligned
                else self.method_template
            )
            loss, _ = method.compute_loss(
                (logits, projections),
                batch_data.labels,
                weight=batch_data.usable_weights,
                extra_info=extra_info,
            )
            self.test_auroc_aligned.update(logits.reshape(-1), batch_data.labels)
            self.log("test/loss_aligned", loss, on_epoch=True, add_dataloader_idx=False)
            self.test_aligned_outputs.append(
                {
                    "label": batch_data.labels.cpu(),
                    "y_prob": torch.sigmoid(logits.reshape(-1)).cpu(),
                    "drain": batch_data.drains.cpu(),
                }
            )
        else:
            # Misaligned
            method = (
                self.test_method_misaligned
                if self.test_method_misaligned
                else self.method_template
            )
            loss, _ = method.compute_loss(
                (logits, projections),
                batch_data.labels,
                weight=batch_data.usable_weights,
                extra_info=extra_info,
            )
            self.test_auroc_misaligned.update(logits.reshape(-1), batch_data.labels)
            self.log(
                "test/loss_misaligned", loss, on_epoch=True, add_dataloader_idx=False
            )
            self.test_misaligned_outputs.append(
                {
                    "label": batch_data.labels.cpu(),
                    "y_prob": torch.sigmoid(logits.reshape(-1)).cpu(),
                    "drain": batch_data.drains.cpu(),
                }
            )

    def on_test_epoch_end(self):
        auroc_aligned = self.test_auroc_aligned.compute()
        auroc_misaligned = self.test_auroc_misaligned.compute()

        self.log("test/auroc_aligned", auroc_aligned)
        self.log("test/auroc_misaligned", auroc_misaligned)

        # AUROC is computed on the merged test set using:
        # (label=1, drain=0) U (label=0, drain=1).
        merged_test_outputs = self.test_aligned_outputs + self.test_misaligned_outputs
        if merged_test_outputs:
            merged_labels = torch.cat([x["label"] for x in merged_test_outputs]).float()
            merged_probs = torch.cat([x["y_prob"] for x in merged_test_outputs])
            merged_drains = torch.cat([x["drain"] for x in merged_test_outputs]).float()

            fairness_score, fairness_details = compute_group_fairness_score(
                merged_probs,
                merged_labels,
                merged_drains,
                scope="test_epoch_aggregated",
                power=self.config.fairness_power,
            )
            self.log("test/fairness_score", fairness_score)
            if "summary" in fairness_details:
                self.log(
                    "test/fairness_mean_pairwise_auroc",
                    fairness_details["summary"]["mean_pairwise_auroc"],
                )
                self.log(
                    "test/fairness_min_pairwise_auroc",
                    fairness_details["summary"]["min_pairwise_auroc"],
                )
                self.log(
                    "test/fairness_arithmetic_mean",
                    fairness_details["summary"].get("arithmetic_mean", 0.0),
                )
                self.log(
                    "test/fairness_geometric_mean",
                    fairness_details["summary"].get("geometric_mean", 0.0),
                )
                self.log(
                    "test/fairness_harmonic_mean",
                    fairness_details["summary"].get("harmonic_mean", 0.0),
                )

            if "pairwise" in fairness_details:
                for pair_key, pair_info in fairness_details["pairwise"].items():
                    if "auroc" in pair_info:
                        self.log(f"test/pairwise_auroc_{pair_key}", pair_info["auroc"])

            robust_auroc, robust_details = compute_saab_robust_auroc(
                preds=merged_probs,
                labels=merged_labels,
                drains=merged_drains,
            )
            if robust_auroc is not None:
                self.log("test/robust_auroc_saab", robust_auroc)
                logging.info(f"test/robust_auroc_saab: {robust_auroc:.4f}")
            else:
                self.log("test/robust_auroc_saab", float("nan"))
                logging.info(
                    "test/robust_auroc_saab: nan (insufficient robust subset diversity)"
                )

            valid_group_mask = ~torch.isnan(merged_drains)
            if valid_group_mask.any():
                merged_labels_valid = merged_labels[valid_group_mask].long()
                merged_probs_valid = merged_probs[valid_group_mask]
                merged_drains_valid = merged_drains[valid_group_mask].long()

                unique_labels = torch.unique(merged_labels_valid, sorted=True).tolist()
                unique_groups = torch.unique(merged_drains_valid, sorted=True).tolist()

                for label_value in unique_labels:
                    for drain_value in unique_groups:
                        subgroup_mask = (merged_labels_valid == label_value) & (
                            merged_drains_valid == drain_value
                        )
                        if subgroup_mask.any():
                            metric_name = f"test/mean_score_y{int(label_value)}_g{int(drain_value)}"
                            self.log(
                                metric_name, merged_probs_valid[subgroup_mask].mean()
                            )

        if self.test_aligned_outputs:
            labels_aligned = torch.cat(
                [x["label"] for x in self.test_aligned_outputs]
            ).float()
            probs_aligned = torch.cat([x["y_prob"] for x in self.test_aligned_outputs])

            brier_aligned = ((probs_aligned - labels_aligned) ** 2).mean()
            calib_dist_aligned = torch.abs(probs_aligned.mean() - labels_aligned.mean())

            self.log("test_brier_aligned", brier_aligned)
            self.log("test_calib_dist_aligned", calib_dist_aligned)

        if self.test_misaligned_outputs:
            labels_misaligned = torch.cat(
                [x["label"] for x in self.test_misaligned_outputs]
            ).float()
            probs_misaligned = torch.cat(
                [x["y_prob"] for x in self.test_misaligned_outputs]
            )

            brier_misaligned = ((probs_misaligned - labels_misaligned) ** 2).mean()
            calib_dist_misaligned = torch.abs(
                probs_misaligned.mean() - labels_misaligned.mean()
            )

            self.log("test_brier_misaligned", brier_misaligned)
            self.log("test_calib_dist_misaligned", calib_dist_misaligned)

        self.test_aligned_outputs.clear()
        self.test_misaligned_outputs.clear()

    def configure_optimizers(self):
        encoder_params = list(filter(lambda p: p.requires_grad, self.model.encoder.parameters()))
        clf_params = list(filter(lambda p: p.requires_grad, self.model.clf.parameters()))

        params = []
        if encoder_params:
            params.append({"params": encoder_params, "lr": self.config.lr / 5})

        clf_weight_decay = self.config.weight_decay
        if self.freeze_encoder and self.config.method == "dfr":
            clf_weight_decay = getattr(
                self.config, "dfr_stage2_weight_decay", self.config.weight_decay
            )

        if clf_params:
            params.append({"params": clf_params, "lr": self.config.lr, "weight_decay": clf_weight_decay})

        if self.model.projection_head is not None:
            proj_params = list(filter(lambda p: p.requires_grad, self.model.projection_head.parameters()))
            if proj_params:
                params.append(
                    {
                        "params": proj_params,
                        "lr": self.config.lr,
                        "weight_decay": self.config.weight_decay, # DFR doesn't use projection, but standard default
                    }
                )

        if torch.cuda.is_available() and torch.__version__ >= "2.0":
            optimizer = optim.AdamW(
                params, weight_decay=self.config.weight_decay, eps=1e-10, fused=True
            )
        else:
            optimizer = optim.AdamW(
                params, weight_decay=self.config.weight_decay, eps=1e-10
            )

        return optimizer

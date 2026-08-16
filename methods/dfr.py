import torch.nn as nn
from .base import BaseMethod


class DFRMethod(BaseMethod):
    """
    Deep Feature Reweighting (DFR) or Last Layer Re-Training.
    Stage 1 uses StandardMethod (ERM).
    Stage 2 discards the Stage-1 head and refits this one on a group-balanced
    subsample of the validation split (see `dfr_stage2.run_dfr_stage2`).

    The head is intentionally identical to StandardMethod's so that DFR is
    compared against the other baselines at matched capacity; the loss below is
    only used for evaluation, since Stage 2 is fitted with sklearn rather than
    by the Lightning training loop.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_model_components(self, num_features: int):
        # Same architecture as StandardMethod -- see class docstring.
        clf = nn.Sequential(nn.Linear(num_features, 512), nn.ReLU(), nn.Linear(512, 1))
        # We return None for the projection head because we don't use it.
        return clf, None

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        logits, _ = model_output

        # DFR uses standard BCE loss. The data sampling handles the balancing.
        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        loss = bce_loss

        return loss, {"bce": bce_loss.item(), "wbce": wbce_loss.item()}

import torch.nn as nn
from .base import BaseMethod


class DFRMethod(BaseMethod):
    """
    Deep Feature Reweighting (DFR) or Last Layer Re-Training.
    Stage 1 uses StandardMethod (ERM).
    Stage 2 uses this method, providing a simple linear layer and standard BCE loss.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_model_components(self, num_features: int):
        # DFR only requires a standard linear classifier
        clf = nn.Linear(num_features, 1)
        # We return None for the projection head because we don't use it.
        return clf, None

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        logits, _ = model_output

        # DFR uses standard BCE loss. The data sampling handles the balancing.
        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)
        loss = bce_loss

        return loss, {"bce": bce_loss.item(), "wbce": wbce_loss.item()}

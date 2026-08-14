from .standard import StandardMethod
from .supcon import SupConMethod
from .mmd import MMDMethod
from .cdan import CDANMethod
from .score_matching import ScoreMatchingMethod
from .score_matching_dataset import DatasetScoreMatchingMethod
from .jtt import JTTMethod
from .soft_equalized_odds import SoftEqualizedOddsMethod
from .dfr import DFRMethod
def get_method(method, config):
    """Factory function to initialize the correct strategy."""
    if method == "standard":
        return StandardMethod(config)
    elif method == "supcon":
        return SupConMethod(config)
    elif method == "mmd":
        return MMDMethod(config)
    elif method == "cdan":
        return CDANMethod(config)
    elif method == "score_matching":
        return ScoreMatchingMethod(config)
    elif method == "dataset_score_matching":
        return DatasetScoreMatchingMethod(config)
    elif method == "jtt":
        return JTTMethod(config)
    elif method == "soft_equalized_odds":
        return SoftEqualizedOddsMethod(config)
    elif method == "dfr":
        return DFRMethod(config)
    else:
        raise ValueError(f"Method '{method}' not implemented.")

from dataclasses import dataclass
from pathlib import Path
import torch
import argparse
from omegaconf import OmegaConf


@dataclass
class ExperimentConfig:
    """
    Central configuration object for the training pipeline.
    """

    method: str = "standard"
    backbone: str = "densenet"
    use_cached_features: bool = False
    select_chkpt_on: str = "fairness"

    lr: float = 0.0001
    weight_decay: float = 0.005
    ema_decay: float = 0.9

    supcon_lambda: float = 0.50
    supcon_temperature: float = 0.10
    mmd_lambda: float = 1.0
    cdan_lambda: float = 1.0
    cdan_entropy: bool = True
    score_matching_lambda: float = 10.0
    score_matching_min_subgroup_count: int = 1
    dataset_score_matching_lambda: float = 10.0
    dataset_score_matching_min_subgroup_count: int = 10
    soft_eo_lambda: float = 1.0
    soft_eo_mode: str = "both"
    soft_eo_tau: float = 0.5
    soft_eo_min_subgroup_count: int = 1
    soft_eo_temp_start: float = 0.0
    soft_eo_temp_end: float = 25.0
    soft_eo_temp_schedule_epochs: int = 25
    jtt_duration: int = 1
    jtt_lambda: float = 4.0
    dfr_stage1_epochs: int = 150
    # Deprecated: Stage 2 is fitted to convergence by sklearn (see dfr_stage2.py)
    # and no longer runs a Lightning training loop, so this has no effect. Kept
    # so existing YAML configs that set it still load.
    dfr_stage2_epochs: int = 30
    # L2 penalty for the Stage-2 head; passed to MLPClassifier(alpha=...).
    dfr_stage2_weight_decay: float = 0.05
    # Stage-2 solver budget. max_iter is an upper bound: fitting stops early once
    # the loss fails to improve by `tol` for `n_iter_no_change` iterations.
    dfr_stage2_max_iter: int = 2000
    dfr_stage2_n_iter_no_change: int = 25
    dfr_stage2_tol: float = 1e-5
    # Fraction of the val split used to retrain the DFR head; the remainder is
    # held back for checkpoint selection so Stage 2 never selects on its own
    # training data.
    dfr_stage2_train_fraction: float = 0.5
    dfr_split_seed: int = 42
    # Stage-1 CSV variant. Kept separate from balance_val so that turning on
    # val balancing does not silently swap the Stage-1 training split.
    dfr_stage1_use_v2_csv: bool = False

    balance_train: bool = False
    balance_val: bool = False
    fairness_power: float = -1.0

    epochs: int = 150
    batch_size: int = 64
    num_runs: int = 1
    num_workers: int = 12
    seed: int = 42

    data_dir: str = "data"
    csv_dir: str = "csv_data"
    out_dir: str = "runs"
    features_dir: str = "features_cache"
    comment: str = "No comment provided"

    @property
    def device(self):
        """Helper to automatically determine device (CUDA -> MPS -> CPU)"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def get_config(args: argparse.Namespace, trial=None) -> ExperimentConfig:
    """
    Constructs an ExperimentConfig object based on YAML configuration,
    command-line overrides, and Optuna trial suggestions.
    """
    # 0. Define the schema
    schema = OmegaConf.structured(ExperimentConfig)

    # 1. Load base configuration
    base_conf = OmegaConf.load(args.config)

    # 2. If debug mode, load and merge debug overrides
    if args.debug:
        debug_conf = OmegaConf.load(Path(__file__).parent / "config" / "debug.yaml")
        base_conf = OmegaConf.merge(base_conf, debug_conf)

    # 3. Merge dynamic command-line overrides (dotlist format)
    cli_conf = OmegaConf.from_cli(args.opts)

    # Merge everything onto the strictly typed schema
    merged_conf = OmegaConf.merge(schema, base_conf, cli_conf)

    # 4. Convert OmegaConf dict back to a standard dataclass instance
    config: ExperimentConfig = OmegaConf.to_object(merged_conf)

    # 5. Optuna Parameter Overrides (takes absolute precedence during hyperparam search)
    if trial is not None:
        if config.method == "supcon":
            config.supcon_lambda = trial.suggest_float(
                "supcon_lambda", 100.0, 400.0, log=True
            )
            config.supcon_temperature = trial.suggest_float(
                "supcon_temperature", 0.05, 0.5
            )
        elif config.method == "mmd":
            config.mmd_lambda = trial.suggest_float("mmd_lambda", 1e-4, 10.0, log=True)
        elif config.method == "cdan":
            config.cdan_lambda = trial.suggest_float(
                "cdan_lambda", 1e-4, 100.0, log=True
            )
            config.cdan_entropy = True
        elif config.method == "score_matching":
            config.score_matching_lambda = trial.suggest_float(
                "score_matching_lambda", 0.1, 1000.0, log=True
            )
        elif config.method == "dataset_score_matching":
            config.dataset_score_matching_lambda = trial.suggest_float(
                "dataset_score_matching_lambda", 1e-1, 1e4, log=True
            )
        elif config.method == "jtt":
            config.jtt_duration = trial.suggest_int(
                "jtt_duration", 1, max(1, config.epochs // 2)
            )
            config.jtt_lambda = trial.suggest_int("jtt_lambda", 2, 100, log=True)
        elif config.method == "soft_equalized_odds":
            config.soft_eo_lambda = trial.suggest_float(
                "soft_eo_lambda", 1e-3, 1e2, log=True
            )
        elif config.method == "dfr":
            # Tuned on the disjoint selection half of the val split; this is the
            # MLPClassifier `alpha` (the DFR "inverse regularization strength").
            config.dfr_stage2_weight_decay = trial.suggest_float(
                "dfr_stage2_weight_decay", 1e-4, 1.0, log=True
            )

    return config

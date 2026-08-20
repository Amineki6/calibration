import argparse
import json
import logging
import pathlib
import pickle
import sys
from pathlib import Path

# Add project root to sys.path so we can import config, model, etc.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from calibration.calibration_metrics import (
    calibration_slope_intercept,
    nan_calibration_result,
)
from calibration.core import (
    apply_group_calibration,
    fit_group_calibrators,
    add_calibration_weights,
)
from calibration.plotting import render_reliability_plots
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import methods
from config import ExperimentConfig
from dataset import CXP_dataset
from lightning_module import CXPLightningModule
from model import CXP_Model
from scoring import compute_group_fairness_score


CALIBRATION_WEIGHT_COL = "calibration_weight"
CALIBRATION_SPLITS = ("aligned", "misaligned")

# Aggregated calibration keys, treated exactly like the bare AUROC split keys:
# one value per checkpoint/trial, summarised as mean +/- std.
CALIBRATION_METRIC_KEYS = tuple(
    f"cal_{stat}_{split}"
    for split in CALIBRATION_SPLITS
    for stat in ("slope", "intercept_joint", "intercept_fixed")
)

# Mapping from the flat per-checkpoint metric names to the keys returned by
# calibration_slope_intercept(). CI bounds are logged and written to CSV, but
# never averaged across trials.
_CALIBRATION_KEY_MAP = (
    ("cal_slope_{split}", "slope"),
    ("cal_slope_ci_low_{split}", "slope_ci_low"),
    ("cal_slope_ci_high_{split}", "slope_ci_high"),
    ("cal_intercept_joint_{split}", "intercept"),
    ("cal_intercept_joint_ci_low_{split}", "intercept_ci_low"),
    ("cal_intercept_joint_ci_high_{split}", "intercept_ci_high"),
    ("cal_intercept_fixed_{split}", "intercept_fixed_slope"),
    ("cal_intercept_fixed_ci_low_{split}", "intercept_fixed_slope_ci_low"),
    ("cal_intercept_fixed_ci_high_{split}", "intercept_fixed_slope_ci_high"),
    ("cal_n_{split}", "n"),
    ("cal_prevalence_{split}", "prevalence"),
    ("cal_converged_{split}", "converged"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of final-eval checkpoints before/after groupwise beta calibration."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base directory containing run directories.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Run directory names relative to --base-path.",
    )
    parser.add_argument(
        "--runs-file",
        type=Path,
        help="Optional text file with one run directory name per line.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory above CheXpert-v1.0-small.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        help="Override the features_dir saved in the checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--beta-params", default="abm")
    parser.add_argument(
        "--calibration-intercept",
        choices=["joint", "fixed"],
        default="joint",
        help=(
            "Intercept variant shown in the summary tables: 'joint' (intercept of the "
            "slope+intercept fit) or 'fixed' (calibration-in-the-large, slope fixed at 1). "
            "Both are always computed, logged and written to CSV."
        ),
    )
    parser.add_argument(
        "--calibration-ci-alpha",
        type=float,
        default=0.05,
        help="Alpha for the Wald CIs of the calibration slope/intercepts (0.05 -> 95%% CI).",
    )
    parser.add_argument(
        "--calibration-val-csv",
        type=Path,
        default=None,
        help="Explicit validation CSV for fitting calibrators; default follows the checkpoint validation split.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="trial_*_best.ckpt",
        help="Checkpoint glob inside final_evaluation/checkpoints.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _windows_safe_torch_load(checkpoint_path: Path) -> dict[str, Any]:
    import os

    torch.serialization.add_safe_globals(
        [
            ExperimentConfig,
            pathlib.PosixPath,
            pathlib.PurePosixPath,
            pathlib.PureWindowsPath,
            pathlib.WindowsPath,
        ]
    )

    original_posix_path = pathlib.PosixPath
    original_windows_path = pathlib.WindowsPath

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path
        pathlib.WindowsPath = original_windows_path

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
    return checkpoint


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_run_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    names.extend(args.runs)
    if args.runs_file is not None:
        names.extend(
            line.strip()
            for line in args.runs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    unique_names = list(dict.fromkeys(names))
    if not unique_names:
        raise ValueError("No runs specified. Use --runs and/or --runs-file.")
    return unique_names


def _build_dataloader(
    config: ExperimentConfig,
    csv_path: Path,
    split_name: str,
) -> DataLoader:
    dataset = CXP_dataset(
        config.data_dir,
        csv_path,
        augment=False,
        compute_sample_weights=False,
        split_name=split_name,
        backbone=config.backbone,
        use_cached_features=config.use_cached_features,
        features_dir=config.features_dir,
    )
    prefetch_factor = 2 if config.num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )


def _run_predictions(
    module: CXPLightningModule,
    dataloader: DataLoader,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    enable_autocast = device.type == "cuda"
    with torch.inference_mode():
        for batch in dataloader:
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                logits_tensor, _ = module.ema_model(
                    batch.inputs.to(device, non_blocking=True)
                )
            # Cast to float32 before sigmoid to prevent fp16 underflow (ties)
            logits_fp32 = logits_tensor.reshape(-1).float()
            probs = torch.sigmoid(logits_fp32).cpu().numpy()
            logits_np = logits_fp32.cpu().numpy()
            labels = batch.labels.cpu().numpy()
            drains = batch.drains.cpu().numpy()

            for label, prob, logit_val, drain in zip(labels, probs, logits_np, drains):
                rows.append(
                    {
                        "split": split_name,
                        "label": int(label),
                        "y_prob": float(prob),
                        "logit": float(logit_val),
                        "drain": float(drain),
                    }
                )
    return pd.DataFrame(rows)


def _safe_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels_array = np.asarray(list(labels))
    scores_array = np.asarray(list(scores))
    if np.unique(labels_array).size < 2:
        return float("nan")
    return float(roc_auc_score(labels_array, scores_array))


def _safe_calibration(
    labels: Iterable[int], probs: Iterable[float], alpha: float = 0.05
) -> dict[str, Any]:
    """NaN-safe logistic recalibration fit; mirrors the _safe_auc contract."""
    labels_array = np.asarray(list(labels), dtype=float)
    probs_array = np.asarray(list(probs), dtype=float)
    try:
        return calibration_slope_intercept(
            labels_array, probs_array, mode="both", alpha=alpha
        )
    except Exception:  # never abort a sweep because one split is unfittable
        logging.warning("Calibration fit failed; reporting NaN.", exc_info=True)
        return nan_calibration_result(
            n=int(labels_array.size), n_pos=int(np.nansum(labels_array))
        )


def _compute_split_metrics(
    test_df: pd.DataFrame,
    score_col: str,
    prob_col: str | None = None,
    ci_alpha: float = 0.05,
) -> dict[str, float]:
    """AUROC/fairness on ``score_col`` plus calibration metrics on ``prob_col``.

    ``prob_col`` must contain probabilities: the pre-calibration AUROC is computed
    on logits (rank-equivalent), which are not valid input for a logistic
    recalibration fit.
    """
    prob_col = score_col if prob_col is None else prob_col
    metrics: dict[str, float] = {}
    for split_name in CALIBRATION_SPLITS:
        split_df = test_df[test_df["split"] == split_name]
        metrics[split_name] = _safe_auc(split_df["label"], split_df[score_col])

        cal = _safe_calibration(split_df["label"], split_df[prob_col], alpha=ci_alpha)
        for out_template, cal_key in _CALIBRATION_KEY_MAP:
            metrics[out_template.format(split=split_name)] = cal[cal_key]

    scores_tensor = torch.tensor(test_df[score_col].values, dtype=torch.float32)
    labels_tensor = torch.tensor(test_df["label"].values, dtype=torch.long)
    groups_tensor = torch.tensor(test_df["drain"].values, dtype=torch.long)
    fairness_score, _ = compute_group_fairness_score(
        logits=scores_tensor,
        labels=labels_tensor,
        groups=groups_tensor,
        scope="test",
        power=-1.0,
    )
    metrics["fairness"] = fairness_score

    return metrics


def _format_summary_block(title: str, rows: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 135)
    lines.append(title)
    lines.append(
        f"{'METHOD':<32} | {'N':<3} | {'ALIGNED AUROC (Mean +/- Std)':<30} | {'MISALIGNED AUROC (Mean +/- Std)':<30} | {'FAIRNESS SCORE (Mean +/- Std)':<30}"
    )
    lines.append("-" * 135)
    for row in rows:
        aligned_mean = (
            float(np.nanmean(row["aligned"])) if row["aligned"] else float("nan")
        )
        aligned_std = (
            float(np.nanstd(row["aligned"])) if row["aligned"] else float("nan")
        )
        misaligned_mean = (
            float(np.nanmean(row["misaligned"])) if row["misaligned"] else float("nan")
        )
        misaligned_std = (
            float(np.nanstd(row["misaligned"])) if row["misaligned"] else float("nan")
        )
        fairness_mean = (
            float(np.nanmean(row["fairness"])) if row.get("fairness") else float("nan")
        )
        fairness_std = (
            float(np.nanstd(row["fairness"])) if row.get("fairness") else float("nan")
        )
        lines.append(
            f"{row['name']:<32} | {len(row['aligned']):<3} | "
            f"{aligned_mean:.4f} +/- {aligned_std:.4f}{'':<14} | "
            f"{misaligned_mean:.4f} +/- {misaligned_std:.4f}{'':<14} | "
            f"{fairness_mean:.4f} +/- {fairness_std:.4f}"
        )
    lines.append("=" * 135)
    return "\n".join(lines)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array))


def _calibration_log_line(metrics: dict[str, float]) -> str:
    parts: list[str] = []
    for split_name in CALIBRATION_SPLITS:
        parts.append(
            f"{split_name}: "
            f"slope={metrics[f'cal_slope_{split_name}']:.4f} "
            f"[{metrics[f'cal_slope_ci_low_{split_name}']:.4f}, "
            f"{metrics[f'cal_slope_ci_high_{split_name}']:.4f}] "
            f"intercept={metrics[f'cal_intercept_joint_{split_name}']:+.4f} "
            f"[{metrics[f'cal_intercept_joint_ci_low_{split_name}']:+.4f}, "
            f"{metrics[f'cal_intercept_joint_ci_high_{split_name}']:+.4f}] "
            f"intercept_b1={metrics[f'cal_intercept_fixed_{split_name}']:+.4f} "
            f"[{metrics[f'cal_intercept_fixed_ci_low_{split_name}']:+.4f}, "
            f"{metrics[f'cal_intercept_fixed_ci_high_{split_name}']:+.4f}] "
            f"(n={metrics[f'cal_n_{split_name}']:.0f}, "
            f"prev={metrics[f'cal_prevalence_{split_name}']:.4f})"
        )
    return " | ".join(parts)


def _format_calibration_summary_block(
    title: str, rows: list[dict[str, Any]], intercept_variant: str = "joint"
) -> str:
    """Calibration counterpart of _format_summary_block (mean +/- std over trials)."""
    intercept_label = "INTERCEPT" if intercept_variant == "joint" else "INTERCEPT(b=1)"
    width = 154
    lines = []
    lines.append("")
    lines.append("=" * width)
    lines.append(f"{title} (Mean +/- Std across trials)")
    lines.append(
        f"{'METHOD':<32} | {'N':<3} | "
        f"{'ALIGNED SLOPE':<26} | {'ALIGNED ' + intercept_label:<26} | "
        f"{'MISALIGNED SLOPE':<26} | {'MISALIGNED ' + intercept_label:<26}"
    )
    lines.append("-" * width)
    for row in rows:
        cells: list[str] = []
        for split_name in CALIBRATION_SPLITS:
            for key in (
                f"cal_slope_{split_name}",
                f"cal_intercept_{intercept_variant}_{split_name}",
            ):
                mean, std = _mean_std(row.get(key, []))
                cells.append(f"{mean:+.4f} +/- {std:.4f}")
        n_fits = len(row.get(f"cal_slope_{CALIBRATION_SPLITS[0]}", []))
        lines.append(
            f"{row['name']:<32} | {n_fits:<3} | "
            + " | ".join(f"{cell:<26}" for cell in cells)
        )
    lines.append("=" * width)
    return "\n".join(lines)


def _configure_file_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _instantiate_module(
    checkpoint: dict[str, Any], config: ExperimentConfig, device: torch.device
) -> CXPLightningModule:
    method = methods.get_method(config.method, config)
    model = CXP_Model(
        method_strategy=method,
        backbone=config.backbone,
        use_cached_features=config.use_cached_features,
    )
    module = CXPLightningModule(model=model, method=method, config=config)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing state_dict")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(
            "Missing state dict keys (%d), sample=%s", len(missing), missing[:5]
        )
    if unexpected:
        logging.warning(
            "Unexpected state dict keys (%d), sample=%s",
            len(unexpected),
            unexpected[:5],
        )
    module.to(device)
    module.eval()
    module.ema_model.eval()
    return module


def _checkpoint_config(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> ExperimentConfig:
    config = ExperimentConfig()
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    ckpt_config = (
        hyper_parameters.get("config") if isinstance(hyper_parameters, dict) else None
    )
    if isinstance(ckpt_config, ExperimentConfig):
        config = ckpt_config

    config.data_dir = args.data_dir
    if getattr(args, "features_dir", None) is not None:
        config.features_dir = args.features_dir
    if config.use_cached_features:
        features_dir = Path(config.features_dir)
        feature_candidates = [features_dir]
        if not features_dir.is_absolute():
            feature_candidates.append(args.data_dir / features_dir)
        feature_candidates.append(args.data_dir / features_dir.name)

        for candidate in feature_candidates:
            if candidate.exists():
                config.features_dir = candidate
                break

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    return config


def _resolve_csv_dir(config: ExperimentConfig, data_dir: Path) -> tuple[Path, str]:
    expected_val_name = (
        "val_drain_shortcut_v2.csv" if config.balance_val else "val_drain_shortcut.csv"
    )

    candidates: list[tuple[Path, str]] = []

    raw_csv_dir = Path(config.csv_dir) if config.csv_dir is not None else None
    if raw_csv_dir is not None:
        candidates.append((raw_csv_dir, "checkpoint_csv_dir_raw"))
        if not raw_csv_dir.is_absolute():
            candidates.append(
                (data_dir / raw_csv_dir, "data_dir_plus_checkpoint_csv_dir")
            )

    candidates.extend(
        [
            (data_dir / "csv_data", "data_dir_csv_data"),
            (data_dir / "csv_data_siim", "data_dir_csv_data_siim"),
            (data_dir, "data_dir_root"),
            (Path("csv_data"), "workspace_csv_data"),
            (Path("csv_data_siim"), "workspace_csv_data_siim"),
        ]
    )

    seen: set[Path] = set()
    deduped_candidates: list[tuple[Path, str]] = []
    for path_candidate, reason in candidates:
        resolved = path_candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_candidates.append((resolved, reason))

    for candidate, reason in deduped_candidates:
        expected_val_path = candidate / expected_val_name
        if expected_val_path.exists():
            return candidate, reason

    attempted = "\n".join([f"  - {p} ({r})" for p, r in deduped_candidates])
    raise FileNotFoundError(
        "Could not resolve CSV directory. None of the candidate directories contained "
        f"{expected_val_name}. Attempted:\n{attempted}"
    )


def _resolve_calibration_val_csv(
    config: ExperimentConfig,
    csv_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, str]:
    if args.calibration_val_csv is not None:
        calibration_val_csv = args.calibration_val_csv
        if not calibration_val_csv.is_absolute():
            calibration_val_csv = csv_dir / calibration_val_csv
        if not calibration_val_csv.exists():
            raise FileNotFoundError(
                f"Explicit calibration validation CSV not found: {calibration_val_csv}"
            )
        return calibration_val_csv, "explicit"

    val_name = (
        "val_drain_shortcut_v2.csv" if config.balance_val else "val_drain_shortcut.csv"
    )
    val_csv = csv_dir / val_name
    if not val_csv.exists():
        raise FileNotFoundError(f"Calibration validation CSV not found: {val_csv}")
    return val_csv, "checkpoint"


def _known_drain_calibration_csv(
    val_csv: Path, cache_dir: Path, logger: logging.Logger
) -> Path:
    val_table = pd.read_csv(val_csv)
    drain_values = pd.to_numeric(val_table["Drain"], errors="coerce")
    known_drain_mask = drain_values.isin([0.0, 1.0])
    if known_drain_mask.all():
        return val_csv
    if not known_drain_mask.any():
        raise RuntimeError(
            f"Calibration validation CSV has no known drain rows: {val_csv}"
        )

    known_drain_csv = cache_dir / f"{val_csv.stem}_known_drain{val_csv.suffix}"
    val_table.loc[known_drain_mask].to_csv(known_drain_csv, index=False)
    logger.info(
        "Calibration validation filtered to known Drain rows: kept=%d total=%d path=%s",
        int(known_drain_mask.sum()),
        len(val_table),
        known_drain_csv,
    )
    return known_drain_csv


def _log_group_prevalence(logger: logging.Logger, name: str, df: pd.DataFrame) -> None:
    logger.info(
        "%s prevalence: overall=%.4f n=%d", name, float(df["label"].mean()), len(df)
    )
    for group in (0.0, 1.0):
        group_df = df[df["drain"] == group]
        if group_df.empty:
            logger.info("%s prevalence: drain=%d n=0", name, int(group))
            continue
        logger.info(
            "%s prevalence: drain=%d n=%d label_mean=%.4f score_mean=%.4f",
            name,
            int(group),
            len(group_df),
            float(group_df["label"].mean()),
            float(group_df["y_prob"].mean()),
        )
        if CALIBRATION_WEIGHT_COL in group_df.columns:
            weights = group_df[CALIBRATION_WEIGHT_COL].to_numpy(dtype=float)
            logger.info(
                "%s weighted prevalence: drain=%d label_mean=%.4f score_mean=%.4f weight_sum=%.4f",
                name,
                int(group),
                float(np.average(group_df["label"], weights=weights)),
                float(np.average(group_df["y_prob"], weights=weights)),
                float(weights.sum()),
            )


def _cache_paths(cache_dir: Path, checkpoint_name: str) -> dict[str, Path]:
    stem = Path(checkpoint_name).stem
    return {
        "meta": cache_dir / f"{stem}_meta_new.json",
        "val_preds": cache_dir / f"{stem}_val_predictions_new.csv",
        "test_preds": cache_dir / f"{stem}_test_predictions_new.csv",
        "calibrators": cache_dir / f"{stem}_beta_calibrators_new.pkl",
    }


def _save_calibrators(calibrators, path: Path, beta_params: str) -> None:
    payload = {
        "beta_params": beta_params,
        "calibration_weighting": "drain_stratified_class_balanced",
        "groups": {group: calibrator for group, calibrator in calibrators.items()},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_calibrators(path: Path) -> tuple[dict, str | None]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"Invalid calibrator cache: {path}")
    beta_params = payload.get("beta_params")
    if beta_params is not None and not isinstance(beta_params, str):
        raise ValueError(f"Invalid beta_params in calibrator cache: {path}")
    return groups, beta_params


def _load_cache_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid cache metadata: {path}")
    return meta


def main() -> None:
    # Set identical PyTorch optimizations as train.py to ensure precision parity
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    run_names = _load_run_names(args)
    device = _resolve_device(args.device)

    aggregate_rows: list[dict[str, Any]] = []
    all_runs_test_dfs: list[pd.DataFrame] = []

    for run_name in run_names:
        run_dir = args.base_path / run_name
        final_eval_dir = run_dir / "final_evaluation"
        checkpoints_dir = final_eval_dir / "checkpoints"
        cache_dir = final_eval_dir / "recalibration_cache_new"
        plots_dir = final_eval_dir / "recalibration_plots_new"
        log_path = final_eval_dir / "recalibration_eval_new.log"
        metric_keys = ("aligned", "misaligned", "fairness") + CALIBRATION_METRIC_KEYS

        if not checkpoints_dir.exists():
            print(f"Skipping {run_name}: Missing checkpoints dir {checkpoints_dir}")
            empty_scores = {key: [] for key in metric_keys}
            summary_rows = [
                {"name": f"{run_name} | original (SKIPPED)", **empty_scores},
                {"name": f"{run_name} | recalibrated (SKIPPED)", **empty_scores},
            ]
            aggregate_rows.extend(summary_rows)
            continue

        cache_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger = _configure_file_logger(log_path)

        checkpoint_paths = sorted(checkpoints_dir.glob(args.checkpoint_glob))
        if not checkpoint_paths:
            logger.warning(f"Skipping {run_name}: No checkpoints found in {checkpoints_dir} matching {args.checkpoint_glob}")
            empty_scores = {key: [] for key in metric_keys}
            summary_rows = [
                {"name": f"{run_name} | original (SKIPPED)", **empty_scores},
                {"name": f"{run_name} | recalibrated (SKIPPED)", **empty_scores},
            ]
            aggregate_rows.extend(summary_rows)
            logger.handlers.clear()
            continue

        logger.info("Run: %s", run_name)
        logger.info("Base path: %s", args.base_path)
        logger.info("Checkpoints dir: %s", checkpoints_dir)
        logger.info("Using device=%s data_dir=%s", device, args.data_dir)

        metric_keys = ("aligned", "misaligned", "fairness") + CALIBRATION_METRIC_KEYS
        original_scores: dict[str, list[float]] = {key: [] for key in metric_keys}
        recalibrated_scores: dict[str, list[float]] = {key: [] for key in metric_keys}
        metrics_rows: list[dict[str, Any]] = []

        for checkpoint_path in checkpoint_paths:
            logger.info("=" * 72)
            logger.info("Checkpoint: %s", checkpoint_path.name)
            cache_paths = _cache_paths(cache_dir, checkpoint_path.name)

            checkpoint = _windows_safe_torch_load(checkpoint_path)
            config = _checkpoint_config(checkpoint, args)
            csv_dir, csv_dir_source = _resolve_csv_dir(config, args.data_dir)
            val_csv, val_csv_source = _resolve_calibration_val_csv(
                config, csv_dir, args
            )
            calibration_inference_csv = _known_drain_calibration_csv(
                val_csv, cache_dir, logger
            )
            aligned_csv = csv_dir / "test_drain_shortcut_aligned.csv"
            misaligned_csv = csv_dir / "test_drain_shortcut_misaligned.csv"

            logger.info("CSV resolution: source=%s csv_dir=%s", csv_dir_source, csv_dir)
            logger.info(
                "Calibration validation CSV: source=%s path=%s", val_csv_source, val_csv
            )

            expected_cache_meta = {
                "checkpoint": str(checkpoint_path),
                "calibration_val_csv": str(val_csv),
                "calibration_inference_csv": str(calibration_inference_csv),
                "aligned_csv": str(aligned_csv),
                "misaligned_csv": str(misaligned_csv),
                "beta_params": args.beta_params,
                "calibration_weighting": "drain_stratified_class_balanced",
            }

            reuse_predictions = (
                not args.force
                and cache_paths["val_preds"].exists()
                and cache_paths["test_preds"].exists()
            )
            reuse_calibrators = (
                reuse_predictions and cache_paths["calibrators"].exists()
            )
            if reuse_calibrators:
                cache_meta = _load_cache_meta(cache_paths["meta"])
                mismatched_meta = {
                    key: (cache_meta.get(key), value)
                    for key, value in expected_cache_meta.items()
                    if cache_meta.get(key) != value
                }
                if mismatched_meta:
                    logger.info(
                        "Cache metadata mismatch for %s: %s",
                        checkpoint_path.name,
                        mismatched_meta,
                    )
                    reuse_calibrators = False

            if reuse_calibrators:
                logger.info(
                    "Loading cached predictions/calibrators for %s",
                    checkpoint_path.name,
                )
                val_df = pd.read_csv(cache_paths["val_preds"])
                test_df = pd.read_csv(cache_paths["test_preds"])
                val_df = add_calibration_weights(val_df)
                calibrators, cached_beta_params = _load_calibrators(
                    cache_paths["calibrators"]
                )
                if cached_beta_params != args.beta_params:
                    logger.info(
                        "Cached beta_params (%s) != requested (%s); recomputing calibrators.",
                        cached_beta_params,
                        args.beta_params,
                    )
                    calibrators = fit_group_calibrators(val_df, args.beta_params)
                    _save_calibrators(
                        calibrators, cache_paths["calibrators"], args.beta_params
                    )
            elif reuse_predictions:
                logger.info(
                    "Loading cached predictions and refitting calibrators for %s",
                    checkpoint_path.name,
                )
                val_df = pd.read_csv(cache_paths["val_preds"])
                test_df = pd.read_csv(cache_paths["test_preds"])
                val_df = add_calibration_weights(val_df)
                calibrators = fit_group_calibrators(val_df, args.beta_params)
                _save_calibrators(
                    calibrators, cache_paths["calibrators"], args.beta_params
                )
                val_df.to_csv(cache_paths["val_preds"], index=False)
            else:
                logger.info(
                    "Running validation/test inference for %s", checkpoint_path.name
                )
                module = _instantiate_module(checkpoint, config, device)

                val_split_name = (
                    "val_v2" if getattr(config, "balance_val", False) else "val"
                )
                val_loader = _build_dataloader(
                    config, calibration_inference_csv, val_split_name
                )
                aligned_loader = _build_dataloader(config, aligned_csv, "test_aligned")
                misaligned_loader = _build_dataloader(
                    config, misaligned_csv, "test_misaligned"
                )

                val_df = _run_predictions(module, val_loader, "val", device)
                test_df = pd.concat(
                    [
                        _run_predictions(module, aligned_loader, "aligned", device),
                        _run_predictions(
                            module, misaligned_loader, "misaligned", device
                        ),
                    ],
                    ignore_index=True,
                )
                val_df = add_calibration_weights(val_df)
                calibrators = fit_group_calibrators(val_df, args.beta_params)
                _save_calibrators(
                    calibrators, cache_paths["calibrators"], args.beta_params
                )
                val_df.to_csv(cache_paths["val_preds"], index=False)

            meta = {
                **expected_cache_meta,
                "csv_dir": str(csv_dir),
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
            }
            cache_paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

            test_df = apply_group_calibration(test_df, calibrators)
            test_df.to_csv(cache_paths["test_preds"], index=False)
            all_runs_test_dfs.append(test_df)

            _log_group_prevalence(logger, "calibration_val", val_df)
            _log_group_prevalence(logger, "merged_test", test_df)

            score_col_original = "logit" if "logit" in test_df.columns else "y_prob"
            # AUROC/fairness stay on the original score column (logits are
            # rank-equivalent); calibration always needs probabilities.
            original_metrics = _compute_split_metrics(
                test_df,
                score_col_original,
                prob_col="y_prob",
                ci_alpha=args.calibration_ci_alpha,
            )
            recalibrated_metrics = _compute_split_metrics(
                test_df,
                "y_prob_calibrated",
                ci_alpha=args.calibration_ci_alpha,
            )
            for key in metric_keys:
                original_scores[key].append(original_metrics[key])
                recalibrated_scores[key].append(recalibrated_metrics[key])
            for variant, variant_metrics in (
                ("original", original_metrics),
                ("recalibrated", recalibrated_metrics),
            ):
                metrics_rows.append(
                    {
                        "run": run_name,
                        "checkpoint": checkpoint_path.name,
                        "variant": variant,
                        **variant_metrics,
                    }
                )

            logger.info(
                "Original      | aligned=%.4f | misaligned=%.4f | fairness=%.4f",
                original_metrics["aligned"],
                original_metrics["misaligned"],
                original_metrics["fairness"],
            )
            logger.info(
                "Recalibrated  | aligned=%.4f | misaligned=%.4f | fairness=%.4f",
                recalibrated_metrics["aligned"],
                recalibrated_metrics["misaligned"],
                recalibrated_metrics["fairness"],
            )
            logger.info(
                "Original      | calibration | %s",
                _calibration_log_line(original_metrics),
            )
            logger.info(
                "Recalibrated  | calibration | %s",
                _calibration_log_line(recalibrated_metrics),
            )

            plot_prefix = checkpoint_path.stem
            render_reliability_plots(test_df, plots_dir, plot_prefix)

        summary_rows = [
            {"name": f"{run_name} | original", **original_scores},
            {"name": f"{run_name} | recalibrated", **recalibrated_scores},
        ]
        logger.info(_format_summary_block("Per-run summary", summary_rows))
        logger.info(
            _format_calibration_summary_block(
                "Per-run calibration summary",
                summary_rows,
                args.calibration_intercept,
            )
        )

        if metrics_rows:
            metrics_csv = final_eval_dir / "recalibration_metrics_new.csv"
            pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
            logger.info("Wrote per-checkpoint metrics: %s", metrics_csv)

        aggregate_rows.extend(summary_rows)
        logger.handlers.clear()

    if all_runs_test_dfs:
        overall_test_df = pd.concat(all_runs_test_dfs, ignore_index=True)
        overall_plots_dir = args.base_path / "recalibration_plots_overall_new"
        overall_plots_dir.mkdir(parents=True, exist_ok=True)
        render_reliability_plots(overall_test_df, overall_plots_dir, "overall_runs")

    aggregate_log_path = args.base_path / "recalibration_eval_summary_new.log"
    aggregate_logger = _configure_file_logger(aggregate_log_path)
    aggregate_logger.info(
        _format_summary_block("Aggregated summary across runs", aggregate_rows)
    )
    aggregate_logger.info(
        _format_calibration_summary_block(
            "Aggregated calibration summary across runs",
            aggregate_rows,
            args.calibration_intercept,
        )
    )
    aggregate_logger.handlers.clear()


if __name__ == "__main__":
    main()
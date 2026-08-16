import os
import sys
import logging
from pathlib import Path
import coolname

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# W&B Cluster Timeouts
os.environ["WANDB_HTTP_TIMEOUT"] = "120"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
os.environ["WANDB__SERVICE_WAIT"] = "300"

# Local imports
from config import ExperimentConfig, get_config
from model import CXP_Model
import methods
from utils import run_jtt_stage1, run_dfr_stage1, TeeStream, _load_dfr_stage1_encoder
from dfr_stage2 import is_backbone_frozen, run_dfr_stage2
from lightning_module import CXPLightningModule
from lightning_datamodule import CXPDataModule

from args import parse_args
from optimization import run_optimization_study


# Register safe globals for checkpoint loading.
import pathlib

torch.serialization.add_safe_globals(
    [
        ExperimentConfig,
        pathlib.PosixPath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.WindowsPath,
    ]
)


def _infer_default_out_dir(data_dir: Path) -> Path:
    """Infer the run output root from the dataset root."""
    return data_dir / "runs"


def setup_logging(root_dir):
    log_path = root_dir / "optuna_training.log"

    # Create or append to the log file
    log_file = open(log_path, "a", encoding="utf-8")

    # Overwrite stdout and stderr
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    # Use only StreamHandler, since Tee will capture its stderr output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = exception_handler



def _stage_banner(title: str) -> None:
    logging.info("=" * 72)
    logging.info(title)
    logging.info("=" * 72)


def _get_method_lambda_items(
    config: ExperimentConfig,
) -> list[tuple[str, float | int | str]]:
    if config.method == "supcon":
        return [
            ("supcon_lambda", config.supcon_lambda),
            ("supcon_temperature", config.supcon_temperature),
        ]
    if config.method == "mmd":
        return [("mmd_lambda", config.mmd_lambda)]
    if config.method == "cdan":
        return [
            ("cdan_lambda", config.cdan_lambda),
            ("cdan_entropy", int(config.cdan_entropy)),
        ]
    if config.method == "score_matching":
        return [
            ("score_matching_lambda", config.score_matching_lambda),
            (
                "score_matching_min_subgroup_count",
                config.score_matching_min_subgroup_count,
            ),
        ]
    if config.method == "dataset_score_matching":
        return [
            ("dataset_score_matching_lambda", config.dataset_score_matching_lambda),
            (
                "dataset_score_matching_min_subgroup_count",
                config.dataset_score_matching_min_subgroup_count,
            ),
        ]
    if config.method == "jtt":
        return [
            ("jtt_lambda", config.jtt_lambda),
            ("jtt_duration", config.jtt_duration),
        ]
    if config.method == "soft_equalized_odds":
        return [
            ("soft_eo_lambda", config.soft_eo_lambda),
            ("soft_eo_mode", config.soft_eo_mode),
            ("soft_eo_tau", config.soft_eo_tau),
            ("soft_eo_min_subgroup_count", config.soft_eo_min_subgroup_count),
            ("soft_eo_temp_start", config.soft_eo_temp_start),
            ("soft_eo_temp_end", config.soft_eo_temp_end),
            ("soft_eo_temp_schedule_epochs", config.soft_eo_temp_schedule_epochs),
        ]
    return []


def run_lightning_training(
    args,
    config,
    trial_number,
    study_root,
    trial=None,
    wandb_group=None,
    run_name=None,
    run_phase: str = "unspecified",
):
    """
    Helper function to run the Lightning training loop.
    Used for both Optimization Objective and Final Evaluation.
    """

    logging.info(
        "[PHASE=%s] Starting run: trial_number=%s, run_name=%s, method=%s, monitor=%s",
        run_phase,
        trial_number,
        run_name if run_name is not None else f"trial_{trial_number}",
        config.method,
        config.select_chkpt_on,
    )
    lambda_items = _get_method_lambda_items(config)
    if lambda_items:
        logging.info(
            "[PHASE=%s] Method coefficients: %s",
            run_phase,
            ", ".join([f"{k}={v}" for k, v in lambda_items]),
        )

    # --- SETUP DATA ---
    datamodule = CXPDataModule(config, debug=args.debug)

    # --- METHOD ---
    method = methods.get_method(config.method, config)

    frozen_backbone = is_backbone_frozen(config)

    # --- JTT STAGE 1 ---
    # NOT skipped for frozen backbones: JTT Stage 1 exists to produce the error
    # set, and the head still trains under a frozen encoder. Skipping it would
    # leave jtt_error_indices=None and silently turn JTT into plain ERM.
    if config.method == "jtt":
        run_jtt_stage1(
            config,
            method,
            datamodule,
            study_root,
            trial_number,
            run_name,
            wandb_group=wandb_group,
            debug=args.debug,
            no_wandb=args.no_wandb,
        )

    # --- DFR STAGE 1 ---
    dfr_best_model_path = None
    if config.method == "dfr":
        if frozen_backbone:
            # ERM cannot move a single weight of a frozen encoder, and the
            # Stage-1 head is discarded by Stage 2, so Stage 1 has no effect
            # whatsoever on the result -- it is pure wasted compute here.
            logging.info(
                "DFR Stage 1 skipped: backbone=%s is frozen, so ERM cannot change "
                "the features Stage 2 is fitted on.",
                config.backbone,
            )
        else:
            dfr_best_model_path = run_dfr_stage1(
                config,
                datamodule,
                study_root,
                trial_number,
                run_name,
                wandb_group=wandb_group,
                debug=args.debug,
                no_wandb=args.no_wandb,
            )

    # --- MAIN TRAINING ---

    # Init Model
    model = CXP_Model(
        method, backbone=config.backbone, use_cached_features=config.use_cached_features
    )

    # DFR Stage 2 surgery must happen BEFORE CXPLightningModule is constructed:
    # the module deep-copies `model` into its EMA wrapper, and validation/test
    # forward through that copy. Fitting the head afterwards would evaluate a
    # model that never saw either the Stage-1 encoder or the Stage-2 head.
    freeze_encoder = config.method == "dfr"
    dfr_stage2_info = None
    if freeze_encoder:
        if dfr_best_model_path:
            _load_dfr_stage1_encoder(model, dfr_best_model_path)

        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()

        # Stage 2 needs datamodule.train_dataset (the head-training half of the
        # val split); Lightning has not called setup() yet at this point.
        datamodule.setup("fit")
        dfr_stage2_info = run_dfr_stage2(config, model, datamodule)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(
            "DFR Stage 2: encoder frozen and in eval(); head fitted by sklearn and "
            "transplanted. Trainable params in CXP_Model: %d",
            n_trainable,
        )

    pl_module = CXPLightningModule(
        model, method, config, freeze_encoder=freeze_encoder
    )

    # Logger
    final_run_name = run_name if run_name else f"{args.study_name}_trial_{trial_number}"
    wandb_logger = WandbLogger(
        project="cxr_optuna_study",
        group=wandb_group,
        name=final_run_name,
        config=config.__dict__,
        save_dir=str(study_root),
        id=f"{final_run_name}_{coolname.generate_slug(2)}",
        mode="disabled" if args.no_wandb else "online",
    )

    if dfr_stage2_info is not None:
        wandb_logger.log_hyperparams(
            {f"dfr_stage2/{k}": v for k, v in dfr_stage2_info.items()}
        )

    # Checkpointing Logic
    monitor_metric = ""
    mode = ""

    if config.select_chkpt_on.upper() == "AUROC":
        monitor_metric = "val/auroc"
        mode = "max"
    elif config.select_chkpt_on.upper() == "WAUROC":
        monitor_metric = "val/wauroc"
        mode = "max"
    elif config.select_chkpt_on.upper() == "FAIRNESS":
        monitor_metric = "val/fairness_score"
        mode = "max"
    elif config.select_chkpt_on.upper() == "BCE":
        monitor_metric = "val/bce"
        mode = "min"
    elif config.select_chkpt_on.upper() == "WBCE":
        monitor_metric = "val/wbce"
        mode = "min"
    elif config.select_chkpt_on.upper() == "WORST_GROUP":
        monitor_metric = "val/worst_group_accuracy"
        mode = "max"

    checkpoint_filename = f"trial_{trial_number}_best"

    checkpoint_callback = ModelCheckpoint(
        dirpath=study_root / "checkpoints",
        filename=checkpoint_filename,
        monitor=monitor_metric,
        mode=mode,
        save_top_k=1,
        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        precision="16-mixed",
        deterministic=True,  # Enforce reproducibility
    )

    if config.method == "dfr":
        # The head is already fitted to convergence, so there is nothing left for
        # gradient descent to do and no checkpoint to select over epochs. Run the
        # validation pass once to produce the monitored metric (which Optuna and
        # the selection half consume), then checkpoint the fitted weights.
        logging.info(
            "[PHASE=%s] DFR: skipping the training loop; running a single "
            "validation pass on the held-out selection half.",
            run_phase,
        )
        trainer.validate(pl_module, datamodule=datamodule)

        metric_value = trainer.callback_metrics.get(monitor_metric)
        if metric_value is None:
            logging.warning(
                "[PHASE=%s] Monitored metric %s absent after validation; "
                "reporting the neutral default.",
                run_phase,
                monitor_metric,
            )
            best_score = 0.0 if mode == "max" else float("inf")
        else:
            best_score = float(metric_value)

        dfr_ckpt_path = study_root / "checkpoints" / f"{checkpoint_filename}.ckpt"
        dfr_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(dfr_ckpt_path)
        logging.info(
            "[PHASE=%s] DFR: saved fitted head to %s (val %s=%s).",
            run_phase,
            dfr_ckpt_path,
            monitor_metric,
            best_score,
        )
    else:
        logging.info("[PHASE=%s] Entering train/validation loop.", run_phase)

        # Fit
        trainer.fit(pl_module, datamodule=datamodule)

        # Get Best Metric Value
        best_score = checkpoint_callback.best_model_score
        if best_score is not None:
            best_score = best_score.item()
        else:
            best_score = 0.0 if mode == "max" else float("inf")

    # --- TESTING ---
    test_results = None
    if run_name is not None:
        logging.info(
            "[PHASE=%s] Entering test loop (best-checkpoint evaluation).", run_phase
        )
        # DFR has no epoch sweep to select from: the in-memory weights ARE the
        # fitted head, so test them directly rather than reloading a checkpoint.
        best_model_path = (
            "" if config.method == "dfr" else checkpoint_callback.best_model_path
        )
        if best_model_path:
            test_results = trainer.test(
                pl_module, datamodule=datamodule, ckpt_path=best_model_path
            )
        else:
            if config.method != "dfr":
                logging.warning(
                    "[PHASE=%s] No best checkpoint found; testing current in-memory weights.",
                    run_phase,
                )
            test_results = trainer.test(pl_module, datamodule=datamodule)

    wandb_logger.experiment.finish()

    # Cleanup
    del trainer, pl_module, model, datamodule
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("[PHASE=%s] Run complete. best_score=%s", run_phase, best_score)
    return best_score, test_results


def run_final_evaluation_runs(args, best_params, study_root):
    final_config = get_config(args)

    # Apply Best Params
    for k, v in best_params:
        if hasattr(final_config, k):
            setattr(final_config, k, v)

    if args.debug:
        final_config.epochs = 2
        final_config.batch_size = 4
        final_config.num_workers = 0

    eval_root = study_root / "final_evaluation"
    eval_root.mkdir(exist_ok=True)

    lambda_items = _get_method_lambda_items(final_config)
    if lambda_items:
        logging.info(
            "Final-eval method coefficients: %s",
            ", ".join([f"{k}={v}" for k, v in lambda_items]),
        )

    import numpy as np
    from collections import defaultdict

    all_test_metrics = defaultdict(list)

    for i in range(args.n_eval_runs):
        run_name = f"{args.study_name}_final_run_{i}"
        _, test_results = run_lightning_training(
            args=args,
            config=final_config,
            trial_number=i,
            study_root=eval_root,
            wandb_group=f"{args.study_name}_final",
            run_name=run_name,
            run_phase=f"final_eval_run_{i}",
        )

        if test_results is not None and len(test_results) > 0:
            res_dict = test_results[0]
            for k, v in res_dict.items():
                if k.startswith("test/") or k.startswith("test_"):
                    all_test_metrics[k].append(v)

    if all_test_metrics:
        logging.info("===== FINAL RESULTS ACROSS RUNS =====")
        if lambda_items:
            for k, v in lambda_items:
                logging.info("estimated/%s: %s", k, v)
        lambda_from_best_params = [(k, v) for (k, v) in best_params if "lambda" in k]
        if lambda_from_best_params:
            for k, v in lambda_from_best_params:
                logging.info("estimated/%s (from best_params): %s", k, v)
        for k, v_list in all_test_metrics.items():
            mean_val = np.mean(v_list)
            std_val = np.std(v_list)
            logging.info(f"{k}: {mean_val:.4f} +/- {std_val:.4f}")
        logging.info("=====================================")


def main():
    args = parse_args()

    # Enforce Reproducibility
    pl.seed_everything(42, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    config = get_config(args)
    data_dir = Path(config.data_dir)

    if config.out_dir == "runs" or config.out_dir is None:
        out_dir = _infer_default_out_dir(data_dir)
    else:
        out_dir = Path(config.out_dir)

    if args.study_name is None:
        args.study_name = (
            coolname.generate_slug(2) + "_" + config.backbone + "_" + config.method
        )

    study_root = out_dir / args.study_name
    study_root.mkdir(parents=True, exist_ok=True)

    (study_root / "checkpoints").mkdir(exist_ok=True)
    (study_root / "predictions").mkdir(exist_ok=True)

    setup_logging(study_root)
    _stage_banner(
        f"RUN START: study={args.study_name}, method={config.method}, n_trials={args.n_trials}, n_eval_runs={args.n_eval_runs}"
    )

    # Optuna Setup
    if args.n_trials == 0 or config.method == "standard":
        optimize = False
        logging.info(
            f"Skipping optimization stage: n_trials={args.n_trials}, method={config.method}"
        )
    else:
        optimize = True

    best_params = []

    if optimize:
        _stage_banner("STAGE 1/3: HYPERPARAMETER OPTIMIZATION (OPTUNA)")

        def objective(trial):
            config = get_config(args, trial)
            best_score, _ = run_lightning_training(
                args=args,
                config=config,
                trial_number=trial.number,
                study_root=study_root,
                trial=trial,
                wandb_group=args.study_name,
                run_phase=f"optuna_trial_{trial.number}",
            )
            chkpt_path = study_root / "checkpoints" / f"trial_{trial.number}_best.ckpt"
            if chkpt_path.exists():
                chkpt_path.unlink()
                
            dfr_chkpt_path = study_root / "checkpoints" / f"trial_{trial.number}_dfr_stage1_best.ckpt"
            if dfr_chkpt_path.exists():
                dfr_chkpt_path.unlink()
                
            return best_score

        best_params = run_optimization_study(args, study_root, objective)

    else:
        _stage_banner("STAGE 1/3: INITIAL TRAIN+EVAL (NO OPTIMIZATION)")

        if args.n_eval_runs > 0:
            logging.info(
                "Skipping initial run because n_eval_runs > 0. We will proceed directly to final evaluation runs."
            )
        else:
            config = get_config(args)
            best_score, _ = run_lightning_training(
                args=args,
                config=config,
                trial_number=0,
                study_root=study_root,
                wandb_group=args.study_name,
                run_name=f"{args.study_name}_initial",
                run_phase="initial_eval",
            )
            logging.info(f"Run Completed. Best Score: {best_score}")

    # ========================== FINAL EVALUATION RUNS ==========================================
    _stage_banner("STAGE 2/3: TRANSITION TO FINAL EVALUATION")
    if args.n_eval_runs > 0:
        _stage_banner(f"STAGE 3/3: FINAL EVALUATION RUNS ({args.n_eval_runs} seeds)")
        run_final_evaluation_runs(args, best_params, study_root)
    else:
        _stage_banner("STAGE 3/3: FINAL EVALUATION RUNS (SKIPPED)")
        logging.info("No final evaluation runs requested (n_eval_runs == 0).")

    _stage_banner("RUN COMPLETE")


if __name__ == "__main__":
    main()

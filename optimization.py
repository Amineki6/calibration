import json
import logging
import datetime
import optuna
from optuna.samplers import GPSampler
import torch
from wandb.errors import CommError


def run_optimization_study(args, study_root, objective_fn, select_chkpt_on: str):
    """
    Sets up and runs the Optuna hyperparameter optimization study.
    """
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    config_dict = vars(args).copy()
    config_dict["start_time"] = start_time_str

    # Save experiment config
    with open(study_root / "experiment_config.json", "w") as f:
        json.dump(config_dict, f, indent=4, default=str)

    logging.info(f"Starting Study: {args.study_name}")
    logging.info(
        f"Device: {'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'}"
    )

    if "AUROC" in select_chkpt_on.upper() or select_chkpt_on.upper() in [
        "FAIRNESS",
        "WORST_GROUP",
    ]:
        direction = "maximize"
    else:
        direction = "minimize"

    study = optuna.create_study(
        sampler=GPSampler(),
        direction=direction,
        study_name=args.study_name,
        load_if_exists=True,
    )

    logging.info(
        "========================== RUN OPTIMIZATION =========================="
    )

    study.optimize(
        objective_fn,
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(CommError, TimeoutError, ConnectionError),
    )

    best_params = list(study.best_trial.params.items())

    logging.info("===== STUDY COMPLETED =====")
    logging.info(f"Best Trial Number: {study.best_trial.number}")
    logging.info(f"Best Value ({select_chkpt_on}): {study.best_trial.value}")
    logging.info("Best Params:")
    for k, v in best_params:
        logging.info(f"  {k}: {v}")

    return best_params

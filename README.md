# Prevalence calibration as shortcut mitigation 

<p align="center">
  <img src="images/thumbnail.jpg" alt="Thumbnail">
</p>

Code repository supporting the paper "Prevalence calibration as shortcut mitigation" 

**Mohamed Amine Kina, Eike Petersen**



## Installation

1. Clone the repository:
```bash
git clone https://github.com/Amineki6/calibration.git
cd calibration
```

2. Create a virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `train.py`: The main entry point for running training, Optuna sweeps, and final evaluations.
- `config/`: Directory containing YAML configuration files (e.g., `default.yaml`).
- `args.py` / `config.py`: Centralized configuration loading using `omegaconf`.
- `dataset.py` / `lightning_datamodule.py`: Handling of data loading, caching, and batching.
- `model.py` / `lightning_module.py`: PyTorch Lightning model definitions and backbone integrations.
- `methods/`: Directory containing all shortcut mitigation algorithms (e.g., `cdan.py`, `mmd.py`, `jtt.py`).
- `calibration/`: Directory containing all scripts for groupwise recalibration and reliability evaluation.
- `extract_features.py`: Script to extract and cache foundation model features.

## Implemented Methods

You can select a shortcut mitigation algorithm by setting the `method` configuration parameter (e.g., `method=supcon`). Supported methods include:
- `standard`: Standard Empirical Risk Minimization (ERM)
- `supcon`: Supervised Contrastive Learning
- `mmd`: Conditional Maximum Mean Discrepancy
- `cdan`: Conditional Domain Adversarial Networks (CDAN+E)
- `score_matching` & `dataset_score_matching`: Conditional Score Matching (CSM) objectives
- `soft_equalized_odds`: Soft Equalized Odds penalty
- `jtt`: Just Train Twice
- `logit_adjustment`: Group-conditional logit adjustment (prior-corrected BCE); see below
- `logit_adjustment_score_matching` / `logit_adjustment_dataset_score_matching`: logit adjustment combined with the batch- / dataset-level score matching penalty

## Usage

### Basic Training

To run a standard baseline with no hyperparameter optimization (using default config parameters):

```bash
python train.py \
    method=standard \
    backbone=densenet \
    data_dir=/path/to/data \
    csv_dir=/path/to/csv \
    epochs=50 \
    batch_size=64
```

### Hyperparameter Optimization

The framework is highly optimized for running Optuna trials to find the best mitigation hyperparameters. You can specify the number of trials and the evaluation metric for model selection. 

```bash
python train.py \
    --n_trials 50 \
    method=score_matching \
    backbone=medsiglip \
    select_chkpt_on=fairness \
    use_cached_features=true
```
*Tip: When using foundation models, pass `use_cached_features=true` to load pre-extracted features instead of running images through the heavy backbone. Make sure you run the `extract_features.py` script first.*

### Logit Adjustment (in-training prevalence calibration)

`methods/logit_adjustment.py` trains with BCE on shifted logits,
`BCE(z + tau * (logit(pi_g) - logit(pi_ref)), y)`, where `pi_g = P(y=1 | drain=g)` is estimated from the training split and `pi_ref` is a common reference prevalence. At inference the raw score `sigmoid(z)` is used. No group labels are needed at test time.

Under the assumption that the drain affects the image only through a label-independent signature plus the label prior, the population optimum of this loss no longer depends on the drain. Every group is then calibrated at `pi_ref`. With `pi_ref = 0.5` (the default) this is the in-training counterpart of the groupwise prevalence calibration below. The module docstring has the derivation, the relation to score matching, caveats and references.

```bash
# a) standalone: no hyperparameters by default, so Optuna is skipped automatically
python train.py --n_eval_runs 5 method=logit_adjustment backbone=medsiglip use_cached_features=true

# b) combined with score matching: Optuna tunes the score-matching lambda
#    (same upper bound as score_matching-only; lower bound 1e-3 so it can reach standalone LA)
python train.py --n_trials 50 --n_eval_runs 5 method=logit_adjustment_score_matching backbone=medsiglip use_cached_features=true
python train.py --n_trials 50 --n_eval_runs 5 method=logit_adjustment_dataset_score_matching backbone=medsiglip use_cached_features=true
```

Options (all methods above):
- `logit_adjustment_reference_prior`: `uniform` (0.5, default), `pooled` (training marginal) or a number in (0, 1). This does not affect AUROCs; it affects Brier score and calibration-in-the-large.
- `logit_adjustment_tau`: offset scale (1.0 = exact prior correction, 0.0 = plain BCE). Set `logit_adjustment_tune_tau=true` to let Optuna search it in [0, 2].
- `logit_adjustment_missing_group`: how to treat rows with `Drain=NaN` (most of the SIIM training split): `own_group` (default, own estimated prevalence), `no_offset` (plain BCE) or `exclude`.
- With `balance_train=true`, prevalences are estimated under the sampler weights. The repo's (label, drain)-balanced sampler makes both groups 50/50, so the offsets become ~0.

### Recalibration

Post-hoc recalibration of trained models, utilizing **groupwise beta calibration**. This technique (Prevalence-aware calibration) adjusts the model's predicted probabilities across different subpopulation groups to mitigate shortcut learning.

You can run batch evaluation of your final checkpoints before and after recalibration using `calibration/eval_recalibrated_runs.py`:

```bash
python calibration/eval_recalibrated_runs.py \
    --base-path /path/to/runs_directory \
    --runs run_1 run_2 \
    --data-dir /path/to/data \
    --beta-params abm
```

**Key Arguments:**
- `--base-path`: The root directory where your run outputs are stored.
- `--runs`: Specific run directory names to evaluate. You can alternatively provide a text file using `--runs-file`.
- `--calibration-val-csv`: Provide an explicit validation CSV file used for fitting the beta calibrators. If omitted, it defaults to the checkpoint's validation split.

## Configuration

Default hyperparameters and training parameters are managed in `config.py` using `ExperimentConfig`. You can override most of these from the command line.

For example, to manually override method-specific hyperparams instead of using Optuna:
```bash
python train.py method=mmd mmd_lambda=2.5 epochs=50
```

## Logging and Tracking

The project natively integrates with **Weights & Biases (W&B)**. 
- If you prefer to disable W&B tracking entirely, simply pass the `--no_wandb` flag to your `train.py` command.
- A comprehensive `optuna_training.log` is also generated automatically inside the output directory.

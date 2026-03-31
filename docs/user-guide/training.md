# Training

This page covers the training workflow, configuration options, and running models on HPC clusters.

## General Training Workflow

Training a model in Neural-LAM generally follows three distinct steps:
1. **Pre-processing:** Generating pre-calculated data and standardisation stats for your [Datastore](datastores.md).
2. **Graph Generation:** Creating the specific mesh graph used by the model (see [Graph Generation](graph-generation.md)).
3. **Training:** Executing the training loop using the `train_model` module.

## Training Commands

You execute training via `neural_lam.train_model` with a configuration path.

### Graph-LAM
The base encode-process-decode architecture on a single mesh grid. Used for both L1-LAM and GC-LAM.
```bash
# For 1L-LAM
python -m neural_lam.train_model --model graph_lam --graph 1level --config_path <config_path>

# For GC-LAM
python -m neural_lam.train_model --model graph_lam --graph multiscale --config_path <config_path>
```

### Hi-LAM
Uses a hierarchical mesh graph and performs sequential message passing up and down the hierarchy.
```bash
python -m neural_lam.train_model --model hi_lam --graph hierarchical --config_path <config_path>
```

### Hi-LAM-Parallel
A variant routing message passing in parallel (up, down, inter-level).
```bash
python -m neural_lam.train_model --model hi_lam_parallel --graph hierarchical --config_path <config_path>
```

## Key Training Flags

A few of the key options to configure the training loop include:
* `--config_path`: Path to the configuration file (e.g., `data/myexperiment/config.yaml`).
* `--model`: Which model to train.
* `--graph`: Which graph structure to use.
* `--epochs`: Number of epochs to train for.
* `--processor_layers`: Number of GNN layers to use in the latent processor part of the model.
* `--ar_steps_train`: Number of autoregressive time steps to unroll for when making predictions and computing the loss.
* `--ar_steps_eval`: Number of time steps to unroll for during validation steps.

Checkpoints of trained models are automatically saved inside a `saved_models` directory.

## Logging

### Weights & Biases (W&B)
Neural-LAM integrates with Weights & Biases for live, interactive metric visualization. W&B is on by default and logs to the `neural-lam` project.
* To turn it on/login: `wandb login`
* To run locally without syncing: `wandb off` (saves dryruns locally to `wandb/`).

### MLFlow
You can alternatively log to MLFlow by adding `--logger mlflow` to your command. You must provide the tracking URI through environment variables:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python -m neural_lam.train_model --config_path <config_path> --logger mlflow
```

## High Performance Computing (HPC) & SLURM

The training script scales natively to multi-GPU clusters using PyTorch Lightning's `DDP` backend. Set the `--num_nodes` flag if running across multiple machines. On a standard machine, you can pick specific GPUs using `--devices 0 1`.

When utilizing a SLURM scheduling system, the task can be distributed via a shell execution script:
```bash
#!/bin/bash -l
#SBATCH --job-name=Neural-LAM
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres:gpu=4
#SBATCH --partition=normal
#SBATCH --mem=444G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=lightning_logs/neurallam_out_%j.log
#SBATCH --error=lightning_logs/neurallam_err_%j.log

# Load necessary modules or activate environment
conda activate neural-lam

srun -ul python -m neural_lam.train_model \
    --config_path /path/to/config.yaml \
    --num_nodes $SLURM_JOB_NUM_NODES
```

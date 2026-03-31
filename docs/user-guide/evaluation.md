# Evaluation

This page details how to run evaluation loops on trained models and discusses testing constraints.

## Running Evaluation

Evaluation reuses the training module but specifies the `--eval` flag to indicate evaluation mode.

To evaluate on the validation set:
```bash
python -m neural_lam.train_model --config_path <config-path> --eval val
```

To evaluate on the test set:
```bash
python -m neural_lam.train_model --config_path <config-path> --eval test
```

## Key Evaluation Flags

Most training options overlap into evaluation, but a few become specifically crucial:
* `--load`: Path to the local model checkpoint (`.ckpt`) to load parameters from.
* `--n_example_pred`: The number of example predictions the module should visually plot out.
* `--ar_steps_eval`: The number of autoregressive time steps to iteratively unroll out to during the evaluation loop.

## Multi-GPU Limitations

:::{warning}
While it is technically possible to use multiple GPUs for running evaluation, it is strongly discouraged.

If using multiple devices, PyTorch Lightning's `DistributedSampler` will replicate some samples across ranks to guarantee that all devices process exactly identical batch sizes. Because of these replicated samples, test metrics and visualisations will become inaccurate and skewed.
:::

A possible workaround if you suspect an extreme bottleneck is to strictly use `--batch-size 1` during multi-GPU evaluation. This behavior stems directly from PyTorch Lightning design choices.

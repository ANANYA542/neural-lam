# What Neural-LAM Actually Requires

:::{important}
Neural-LAM is a research framework, not a standalone forecasting tool.
The code implements graph-based neural weather prediction models, but
producing useful forecasts requires pairing it with a large regional
dataset and substantial compute resources.
:::

## Framework, Not Product

Neural-LAM provides the model architectures, training pipeline, and
evaluation tools. It does not include production-scale training data
or the compute to train on it. Think of it as a well-tested engine
that needs fuel (data) and a vehicle (compute infrastructure) to go
anywhere.

## What a Real Training Dataset Looks Like

The models were developed using datasets like DANRA (Danish Reanalysis)
and MEPS (MetCoOp Ensemble Prediction System):

| Property | Typical Value |
|----------|--------------|
| Spatial coverage | Bounded region, ~1000x1000 km |
| Spatial resolution | 2.5 km grid spacing (~400x400 grid points) |
| Temporal coverage | 20-30 years |
| Temporal resolution | 6-hourly |
| Variables | 10-20 atmospheric fields |
| Raw data size | 500 GB - 2 TB |
| Processed Zarr size | 200 GB - 800 GB |

DANRA is a reanalysis product from the Danish Meteorological Institute
covering Denmark at 2.5 km resolution from 1990 to present. Access
typically requires a data-sharing agreement with the producing agency.

## Compute Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1x A100 (40 GB) | 4x A100 (80 GB) |
| System RAM | 64 GB | 128 GB |
| Training time | ~3-5 days (1 GPU) | ~1-2 days (4 GPU DDP) |
| Disk | 500 GB | 2 TB |

Training on a consumer GPU (e.g. RTX 3090, 24 GB VRAM) is possible for
smaller regions or reduced resolutions, with longer training times and
potential memory constraints.

## What the Example Data Is For

The repository includes a small example dataset (~50 MB, auto-downloaded
via [pooch](https://www.fatiando.org/pooch/) when you run tests). It
exists for one purpose: **verifying that the code runs correctly**.

It is a tiny spatial domain with very few timesteps. A model trained on
this data will produce meaningless predictions. That is expected.

The example data lets you:
- Confirm your installation works
- Verify training, graph generation, and evaluation all function
- Develop and debug code changes

It does not let you evaluate forecast quality or compare architectures.

## What the Hello World Example Shows

Running through the quickstart — graph generation, training, evaluation —
demonstrates **the workflow**, not the capability. The output shows how
data flows through the system, how training metrics are logged, and how
autoregressive rollout works step by step. For actual forecast skill, see
the [publications](https://github.com/mllam/neural-lam#publications)
where models trained on full datasets demonstrate competitive skill
against operational NWP systems.

## What You Need to Plan For

To use Neural-LAM for real weather forecasting research:

1. **A regional weather dataset** — existing reanalysis (DANRA, MEPS,
   ERA5 subset) or your own data, 10+ years at <=10 km resolution
2. **Storage** — 1-2 TB of fast disk for raw and processed data
3. **GPU compute** — at least one A100-class GPU for several days.
   University HPC, cloud, or national compute grants
4. **Domain expertise** — which variables, what domain, how to validate
5. **The mllam-data-prep pipeline** — see
   [Preparing a Custom Dataset](../user-guide/custom-dataset.md)

Every serious ML project requires data and compute. Neural-LAM is
transparent about this so you can plan accordingly. The
[mllam Slack](https://join.slack.com/t/ml-lam/shared_invite/zt-3jyw20n4g-ESRxMPPSijiZ2ZA6Nh8XhA)
is a good place to discuss data sources and compute options.

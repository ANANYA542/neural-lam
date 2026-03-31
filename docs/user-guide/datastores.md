# Datastores

This page details how data is handled, processed, and loaded into Neural-LAM using Datastores.

## Introduction to Datastores

A datastore is essentially the data-loading backbone of Neural-LAM. Weather data comes in various complex formats (NetCDF, Zarr, GRIB, Numpy) and is often distributed across many files with missing or irregular values. The datastore's job is to abstract away all this complexity. It acts as an adapter that knows exactly how to read a specific format of raw weather data from your disk and translate it into a unified representation that the rest of the Neural-LAM system can understand.

Without datastores, the core machine learning models would have to be hardcoded to understand specific file formats or dataset quirks. By using datastores, Neural-LAM remains modular: if you want to train on a completely new weather dataset, you don't need to rewrite the graph generation or the training loop—you simply write a new datastore recipe that bridges your raw data to Neural-LAM's standard format.

## How Data Flows

The data flow in Neural-LAM is modularised into distinct steps:

1. **Raw Data:** Weather data resides on disk (e.g., Zarr or Numpy arrays).
2. **DataStore:** A subclass of `BaseDataStore` loads this data, flattens spatial coordinates into a 1D grid index, stacks variables across a feature dimension, calculates normalisation statistics, and handles splitting (train/val/test). It outputs an `xarray.DataArray`.
3. **WeatherDataset:** A `pytorch.Dataset` class samples the datastore's arrays over time, normalises the values, and turns them into sequential `torch.Tensor` objects.
4. **Model:** The graph neural network consumes these tensors to perform autoregressive training and rollout.

## Configuration Structure

Any command you run in neural-lam includes the path to a configuration file to be used (usually called `config.yaml`). This defines the path to the datastore configuration you wish to use. The root directory relative to which all other paths are resolved is the parent directory of this configuration file.

Example directory structure (assuming `config.yaml` is in `data/`):
```text
data/
├── config.yaml           - Configuration file for neural-lam
├── danra.datastore.yaml  - Configuration file for the datastore, referred to from config.yaml
└── graphs/               - Directory containing graphs for training
```

Example `config.yaml`:
```yaml
datastore:
  kind: mdp
  config_path: danra.datastore.yaml
training:
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    weights:
      u100m: 1.0
      v100m: 1.0
      t2m: 1.0
      r2m: 1.0
  output_clamping:
    lower:
      t2m: 0.0
      r2m: 0
    upper:
      r2m: 1.0
```

## Available Datastores

### MDP (mllam-data-prep) Datastore
**Kind ID:** `mdp`

For `MDPDatastore`, the selection, transformation, and pre-calculation steps to go from gridded weather data to an ML-ready format are done by a separate package called `mllam-data-prep`. The datastore configuration specifies source datasets, variables, dimensions, statistics for normalisation, and data splits.

Once configured, Neural-LAM writes the transformed dataset in Zarr format to disk when first initiated. You can also run the pre-processing directly:
```bash
python -m mllam_data_prep --config data/danra.datastore.yaml
```
For large datasets (>10GB), you can distribute the processing locally:
```bash
python -m mllam_data_prep --config data/danra.datastore.yaml --dask-distributed-local-core-fraction 0.5
```

### NpyFiles MEPS Datastore
**Kind ID:** `npyfilesmeps`

Designed to read MEPS data directly from `.npy` files as introduced in Neural-LAM `v0.1.0`. While heavily tied to the MEPS dataset (downloadable from MEPS archives), it acts as an example for processing numpy-based analyses.

Example datastore config snippet (`meps.datastore.yaml`):
```yaml
dataset:
  name: meps_example
  num_forcing_features: 16
  var_longnames:
  - pres_heightAboveGround_0_instant
  # ... (omitted for brevity)
  num_timesteps: 65
  num_ensemble_members: 2
  step_length: 3
  remove_state_features_with_index: [15]
grid_shape_state:
- 268
- 238
projection:
  class_name: LambertConformal
  kwargs:
    central_latitude: 63.3
    central_longitude: 15.0
    standard_parallels:
    - 63.3
    - 63.3
```

For npy-file based datastores, you must manually pre-compute standardization stats:
```bash
python -m neural_lam.datastore.npyfilesmeps.compute_standardization_stats <path-to-datastore-config>
```

## Creating Your Own Datastore

If none of the above fit your needs, you can create a Custom Datastore. To do this, subclass the `neural_lam.datastore.BaseDataStore` class, or `neural_lam.datastore.BaseRegularGridDatastore` if your data rests on a uniform regular grid, and implement the requisite abstract methods for data ingestion and metadata handling.

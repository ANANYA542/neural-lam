# Preparing a Custom Dataset

:::{important}
This guide covers getting your own weather data into Neural-LAM.
To verify your installation, use the built-in example data — it
downloads automatically when you run the tests. This page is for
when you are ready to work with your own data.
:::

## What is mllam-data-prep?

Neural-LAM does not read raw weather files directly. Weather data comes
in many formats with different conventions for variable naming, coordinate
ordering, and missing values.

[mllam-data-prep](https://github.com/mllam/mllam-data-prep) handles the
conversion. You describe your data sources in a YAML config, and
mllam-data-prep reads, selects, regrids, and writes everything into a
standardised Zarr store. Neural-LAM reads only from that Zarr store
through its datastore abstraction. You never need to modify model code
to support a new dataset.

## Step 1: Write a Datastore Config

The config tells mllam-data-prep where your raw data lives, which
variables to extract, how to split train/val/test, and how to compute
normalisation statistics.

Minimal working example:

```yaml
schema_version: v0.5.0
dataset_version: v0.1.0

output:
  variables:
    state:
      values:
        t2m:
          source: era5
          path: "data/era5/*.nc"
          dims: [time, y, x]
        u10m:
          source: era5
          path: "data/era5/*.nc"
          dims: [time, y, x]
      dim_mapping:
        time: time
        grid_index: [y, x]

    forcing:
      values:
        toa_radiation:
          source: era5
          path: "data/era5_forcing/*.nc"
          dims: [time, y, x]
      dim_mapping:
        time: time
        grid_index: [y, x]

    static:
      values:
        lsm:
          source: era5
          path: "data/era5_static/lsm.nc"
          dims: [y, x]
      dim_mapping:
        grid_index: [y, x]

  coord_ranges:
    time:
      start: "1990-01-01"
      end: "2020-12-31"
      step: "PT6H"

  splitting:
    dim: time
    splits:
      train:
        start: "1990-01-01"
        end: "2017-12-31"
      val:
        start: "2018-01-01"
        end: "2019-12-31"
      test:
        start: "2020-01-01"
        end: "2020-12-31"
```

:::{tip}
Check the [mllam-data-prep documentation](https://github.com/mllam/mllam-data-prep)
for the latest `schema_version`.
:::

## Step 2: Run mllam-data-prep

```bash
python -m mllam_data_prep --config data/my_dataset.datastore.yaml
```

This reads source files, computes normalisation statistics over the
training split, and writes a Zarr store.

For large datasets (>10 GB), distribute the processing:

```bash
python -m mllam_data_prep \
    --config data/my_dataset.datastore.yaml \
    --dask-distributed-local-core-fraction 0.5
```

## Step 3: Verify the Output

```python
import xarray as xr

ds = xr.open_zarr("data/my_dataset.zarr")
print(ds)
print(ds["state"])       # (n_times, n_grid_points, n_state_features)
print(ds["forcing"])     # (n_times, n_grid_points, n_forcing_features)
print(ds["static"])      # (n_grid_points, n_static_features)
```

Check that:
- Time dimension spans the expected date range
- Grid point count matches your spatial domain
- All requested variables appear as features
- No unexpected NaN values in the training split

## Step 4: Point Neural-LAM at Your Dataset

Create or update `config.yaml`:

```yaml
datastore:
  kind: mdp
  config_path: my_dataset.datastore.yaml

training:
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    weights:
      t2m: 1.0
      u10m: 1.0
  output_clamping: {}
```

`config_path` is relative to this file's location. You can now
[generate graphs](graph-generation.md) and [train](training.md).

## Supported Data Formats

mllam-data-prep reads any format supported by
[xarray](https://docs.xarray.dev/):

| Format | Extension | Notes |
|--------|-----------|-------|
| NetCDF | `.nc` | Most common for climate/weather data |
| Zarr | `.zarr` | Cloud-optimised, lazy loading |
| GRIB | `.grib`, `.grib2` | Requires `cfgrib` |
| HDF5 | `.h5`, `.hdf5` | Via `h5netcdf` backend |

For NumPy `.npy` files, use the `npyfilesmeps` datastore instead —
see [Datastores](datastores.md).

## Compute Requirements

:::{warning}
Processing large weather datasets is resource-intensive.

| Dataset Size | RAM | Time | Zarr Output |
|-------------|-----|------|-------------|
| 1 GB raw | 4 GB | ~5 min | ~1 GB |
| 10 GB raw | 16 GB | ~30 min | ~8 GB |
| 100 GB raw | 32-64 GB | ~4 hours | ~60 GB |
| 1 TB raw (full DANRA) | 64-128 GB | ~24 hours | ~500 GB |

For datasets above 10 GB, always use `--dask-distributed-local-core-fraction`.
For TB-scale datasets, run on a compute cluster.
:::

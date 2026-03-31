# Overview

Neural-LAM is a framework for graph-based neural weather prediction over
bounded geographic regions. It provides implementations of three GNN
architectures, a modular data-loading layer, and an autoregressive training
pipeline built on PyTorch Lightning.

:::{important}
Neural-LAM is a research framework. The example data in this repository
verifies your installation — it does not produce useful forecasts.
Real weather forecasting requires large regional datasets and substantial
compute. See [What Neural-LAM Actually Requires](../contributing/what-neural-lam-needs.md).
:::

## What Problem It Solves

Most deep learning weather models forecast globally at 25-50 km resolution.
Neural-LAM focuses on **Limited Area Modeling (LAM)**: forecasting over a
specific region at 2.5 km resolution or finer. By concentrating computation
on a bounded domain, it resolves fine-scale phenomena — thunderstorms, sea
breezes, orographic effects — that global models miss.

The key design choice is representing the atmosphere as a **graph** rather
than a regular grid. This allows flexible spatial resolutions, non-rectangular
domains, and hierarchical multi-scale representations.

## Core Concepts

### Limited Area Modeling

A limited area model forecasts over a bounded region (e.g. Scandinavia,
Denmark). The boundary receives conditions from a global model, but
high-resolution prediction happens only within the area of interest. This
makes it feasible to run at 2.5 km grid spacing — roughly 16x finer than
typical global models.

### Graph-Based Prediction

Neural-LAM builds a graph where:

- **Nodes** are grid points carrying atmospheric state variables
  (temperature, wind, pressure, humidity)
- **Edges** represent spatial relationships used for message passing
- **Message passing** propagates information between nodes, simulating
  how weather conditions at one location influence neighbours

The graph is split into a fine **grid** (where data lives) and a coarser
**mesh** (where message passing happens). The model encodes grid features
onto the mesh, processes on the mesh, then decodes predictions back to
the grid.

### Datastores

Raw weather data comes in many formats (NetCDF, GRIB, Zarr, NumPy).
Datastores abstract this away: each datastore reads a specific format
and presents normalised arrays with consistent dimensions. Swapping
data sources means changing a config file, not model code.
See [Datastores](../user-guide/datastores.md).

### Autoregressive Rollout

The model predicts one timestep ahead, feeds that prediction back as
input, and repeats. A 48-hour forecast at 6-hour steps requires 8
sequential predictions. During training, the model unrolls for multiple
steps and accumulates loss at each step — this teaches it to produce
predictions that stay stable when fed back as input.

## Models

### Graph-LAM

Uses a flat, single-level mesh graph. The encoder maps grid features
onto mesh nodes, the processor runs message passing on the mesh, and
the decoder maps results back to the grid. Corresponds to L1-LAM
(one mesh level) and GC-LAM (multiscale edges) from the original papers.
Suitable as a baseline or for smaller domains where one mesh resolution
suffices.

### Hi-LAM

Extends Graph-LAM with a hierarchical, multi-level mesh. Message passing
travels sequentially up the hierarchy (fine to coarse), across each level,
and back down. Coarse levels capture synoptic-scale patterns (fronts, jet
streams); fine levels preserve local detail (convection, coastal effects).
Suited for larger domains where a single mesh resolution cannot capture
both scales.

### Hi-LAM-Parallel

Same architecture as Hi-LAM, but all hierarchical message passing runs
simultaneously rather than sequentially. Explores whether parallel
information flow through the hierarchy is more effective.

## Next Steps

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Install
:link: installation
Set up your environment and verify with the example data.
:::

:::{grid-item-card} Train
:link: ../user-guide/training
Data, graph generation, training, and evaluation.
:::

:::{grid-item-card} Extend
:link: ../contributing/architecture
Understand the codebase and where to make changes.
:::

::::

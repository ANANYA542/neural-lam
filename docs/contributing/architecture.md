# neural-lam · Architecture

:::{note}
Tensor shapes follow the convention `(B, T, N_grid, d_h)` — B = batch,
T = timesteps, N_grid = flattened grid nodes, N_mesh = mesh nodes,
d_h = hidden width, d_state = state variables, d_forcing = forcing variables.
:::

Neural-LAM maps gridded weather analyses onto a graph and steps
autoregressively through time using encode-process-decode message passing.

---

## 01 · System Overview

Data flows through four layers during training: raw weather files are read
by a datastore, wrapped into a PyTorch dataset, batched by a DataLoader,
and consumed by the autoregressive model.

```{mermaid}
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8'}}}%%
graph TB
    raw["Raw Data<br/>Zarr / NpyFiles"] --> base["BaseDataStore<br/>datastore/base.py"]
    base --> impl["MDPDatastore / NpyFilesDatastoreMEPS"]
    impl --> wds["WeatherDataset<br/>weather_dataset.py"]
    wds --> dl["PyTorch DataLoader"]
    dl --> ar["ARModel<br/>models/ar_model.py"]
    ar --> bgm["BaseGraphModel<br/>models/base_graph_model.py"]
    bgm --> variants["GraphLAM / HiLAM / HiLAMParallel"]
    variants --> pred["Predictions"]
    pred --> metrics["Metrics + Visualisation"]

    classDef data fill:#1e3a5f,stroke:#38bdf8,color:#bae6fd;
    classDef ds fill:#2e1065,stroke:#a78bfa,color:#e9d5ff;
    classDef model fill:#431407,stroke:#fb923c,color:#fed7aa;
    classDef output fill:#052e16,stroke:#34d399,color:#a7f3d0;
    class raw,base,impl data;
    class wds,dl ds;
    class ar,bgm,variants model;
    class pred,metrics output;
```

Each layer has a single responsibility. The datastore reads files but knows
nothing about batching. The dataset handles sampling and normalisation but
nothing about file formats. The AR model handles time-stepping but delegates
spatial prediction to the graph model. You can swap any layer independently.

| Tensor | Shape | Description |
|--------|-------|-------------|
| `init_states` | `(B, 2, N_grid, d_state)` | Two consecutive normalised states to warm-start each rollout |
| `forcing` | `(B, T+2, N_grid, d_forcing)` | Forcing slices with static covariates concatenated |
| `target_states` | `(B, T, N_grid, d_state)` | Future trajectory for loss computation |
| `target_times` | `(B, T)` | Datetime timestamps per target step |

:::{note}
Static covariates are concatenated into `forcing` inside `WeatherDataset`.
The batch contract is the 4-tuple `(init_states, target_states, forcing, target_times)`.
:::

---

## 02 · Datastore Class Hierarchy

Datastores let you swap data sources without touching model code. Each
datastore reads a specific raw format and presents normalised arrays with
consistent dimensions. Switching from MEPS NumPy files to DANRA Zarr
via mllam-data-prep means changing a config file, not the training loop.

```{mermaid}
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8'}}}%%
classDiagram
    BaseDataStore <|-- BaseRegularGridDatastore
    BaseRegularGridDatastore <|-- MDPDatastore
    BaseRegularGridDatastore <|-- NpyFilesDatastoreMEPS

    class BaseDataStore {
        +root_path()
        +config()
        +step_length()
        +get_dataarray(category, split)
        +boundary_mask
        +expected_dim_order()
    }

    class BaseRegularGridDatastore {
        +grid_shape_state
        +get_xy()
        +stack_grid_coords()
        +unstack_grid_coords()
        +num_grid_points()
    }

    class MDPDatastore {
        +__init__()
        +get_dataarray()
        +get_standardization_dataarray()
        +boundary_mask
        +coords_projection()
    }

    class NpyFilesDatastoreMEPS {
        +__init__()
        +get_dataarray()
        +_get_single_timeseries_dataarray()
        +_calc_datetime_forcing_features()
        +get_standardization_dataarray()
        +boundary_mask
    }
```

`BaseDataStore` handles any spatial layout including unstructured grids.
`BaseRegularGridDatastore` adds coordinate flattening and grid-shape
utilities for regularly-gridded data. Subclass whichever matches your data.

| Tensor | Shape | Description |
|--------|-------|-------------|
| `state_cube` | `(T, N_grid, d_state)` | Stacked state fields from `get_dataarray(category="state")` |
| `forcing_cube` | `(T, N_grid, d_forcing)` | Forcing arrays including datetime encodings |
| `static_covariates` | `(N_grid, d_static)` | Time-invariant grid features, concatenated into forcing by `WeatherDataset` |

---

## 03 · Autoregressive Unrolling

During training the model unrolls for multiple autoregressive steps,
accumulating loss at each step. This matters because at inference the
model's own predictions become input for subsequent steps. Training on
single-step accuracy alone would cause error accumulation and divergence
over multi-step forecasts.

```{mermaid}
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8','activationBkgColor':'#1e3a5f','activationBorderColor':'#38bdf8'}}}%%
sequenceDiagram
    participant WD as WeatherDataset
    participant DL as DataLoader
    participant AR as ARModel
    participant BG as BaseGraphModel
    participant MT as Metrics

    WD->>DL: init_states, target_states, forcing, target_times
    DL->>AR: batched (init_states, target_states, forcing, target_times)

    loop t = 1 .. T
        AR->>BG: current_state + forcing_t
        BG-->>AR: delta (residual update)
        AR->>AR: next_state = current_state + delta
        AR->>MT: loss(next_state, target_state_t)
    end

    MT-->>AR: total_loss (wMSE)
    AR->>AR: optimizer.step()
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `init_states` | `(B, 2, N_grid, d_state)` | Two consecutive timesteps to warm-start the first AR step |
| `target_states` | `(B, T, N_grid, d_state)` | Future trajectory the loss compares against |
| `forcing` | `(B, T+2, N_grid, d_forcing)` | 1 past + T current + 1 future step, static covariates included |
| `current_state` | `(B, N_grid, d_state)` | Latest prediction fed into the next step |
| `delta` | `(B, N_grid, d_state)` | Residual output — added to `current_state` to produce `next_state` |

:::{note}
The model predicts a residual delta, not the full next state.
`next_state = current_state + delta` keeps the learning task small and
stabilises training during long rollouts.
:::

---

## 04 · Encode → Process → Decode

The grid where data lives (high-resolution, potentially irregular) differs
from the representation where dynamics should be learned (a smaller mesh
where message passing is tractable). The encoder projects grid features onto
mesh nodes, the processor runs message passing on the mesh, and the decoder
maps results back to the grid. Changing your spatial domain does not require
changes to the processor.

```{mermaid}
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#94a3b8'}}}%%
flowchart LR
    classDef stateNode fill:#78350f,stroke:#fbbf24,color:#fef3c7,font-weight:bold;
    classDef encNode fill:#1e3a5f,stroke:#38bdf8,color:#bae6fd,font-weight:bold;
    classDef procNode fill:#052e16,stroke:#34d399,color:#a7f3d0,font-weight:bold;
    classDef decNode fill:#431407,stroke:#fb923c,color:#fed7aa,font-weight:bold;

    prevState["prev_state<br/>(B, N_grid, d_state)"]

    subgraph ENCODE["ENCODE"]
        grid["Grid nodes<br/>(B, N_grid, d_h)"]
        g2m["g2m edges"]
        meshIn["Mesh nodes<br/>(B, N_mesh, d_h)"]
    end

    subgraph PROCESS["PROCESS x N layers"]
        meshLoop["Mesh nodes"]
        m2m["m2m edges"]
        meshOut["Updated mesh<br/>(B, N_mesh, d_h)"]
    end

    subgraph DECODE["DECODE"]
        meshDec["Mesh nodes"]
        m2g["m2g edges"]
        gridOut["Grid nodes<br/>(B, N_grid, d_h)"]
        delta["Residual delta<br/>(B, N_grid, d_state)"]
    end

    nextState["next_state = prev + delta<br/>(B, N_grid, d_state)"]

    prevState --> grid
    grid --> g2m --> meshIn --> meshLoop
    meshLoop --> m2m --> meshOut --> meshLoop
    meshOut --> meshDec
    meshDec --> m2g --> gridOut --> delta
    delta --> nextState
    prevState --> nextState

    class prevState,nextState stateNode;
    class grid,g2m,meshIn encNode;
    class meshLoop,m2m,meshOut procNode;
    class meshDec,m2g,gridOut,delta decNode;
```

- **Encode:** `grid_embedder` projects atmospheric state into hidden space.
  `g2m_gnn` propagates embeddings from grid nodes to mesh nodes.
- **Process:** `m2m_gnn` runs N iterations of message passing across the
  mesh. This is where the model learns atmospheric dynamics.
- **Decode:** `m2g_gnn` propagates processed features back to the grid.
  A final MLP produces the residual delta.

| Tensor | Shape | Description |
|--------|-------|-------------|
| `grid_embed` | `(B, N_grid, d_h)` | State + forcing projected by `grid_embedder` |
| `mesh_latent` | `(B, N_mesh, d_h)` | Mesh features after `g2m_gnn` (encode) |
| `mesh_updated` | `(B, N_mesh, d_h)` | Mesh features after N rounds of `m2m_gnn` (process) |
| `grid_decoded` | `(B, N_grid, d_h)` | Grid features after `m2g_gnn` (decode) |
| `delta` | `(B, N_grid, d_state)` | Residual added to `prev_state` to form `next_state` |

---

## 05 · HiLAM — Hierarchical Processing

A flat single-level mesh (GraphLAM) works for small domains. For larger
areas the model needs to capture both local dynamics (convection, coastal
effects) and synoptic-scale patterns (fronts, jet streams). HiLAM uses a
multi-level mesh: message passing travels upward from fine to coarse,
processes at the coarsest level, then travels back down. Coarser levels
have fewer nodes but wider spatial reach, capturing large-scale correlations
without prohibitively long-range edges at the fine level.

```{mermaid}
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#94a3b8'}}}%%
flowchart LR
    classDef gridNode fill:#1e3a5f,stroke:#38bdf8,color:#bae6fd,font-weight:bold;
    classDef meshNode fill:#2e1065,stroke:#a78bfa,color:#e9d5ff,font-weight:bold;
    classDef opNode fill:#431407,stroke:#fb923c,color:#fed7aa,font-weight:bold;
    classDef procNode fill:#052e16,stroke:#34d399,color:#a7f3d0,font-weight:bold;

    subgraph UPSWEEP["Up Sweep — finest to coarsest"]
        direction LR
        GIN["Grid"]
        ENC["Encode<br/>g2m edges"]
        L0U["Mesh L0<br/>finest"]
        MU0["mesh_up<br/>edge_index"]
        L1U["Mesh L1"]
        MU1["mesh_up<br/>edge_index"]
        L2U["Mesh L2<br/>coarsest"]
        GIN --> ENC --> L0U --> MU0 --> L1U --> MU1 --> L2U
    end

    PROC["Process<br/>m2m edges<br/>at top level"]

    subgraph DOWNSWEEP["Down Sweep — coarsest to finest"]
        direction LR
        L2D["Mesh L2<br/>coarsest"]
        MD1["mesh_down<br/>edge_index"]
        L1D["Mesh L1"]
        MD2["mesh_down<br/>edge_index"]
        L0D["Mesh L0<br/>finest"]
        DEC["Decode<br/>m2g edges"]
        GOUT["Grid<br/>+ residual"]
        L2D --> MD1 --> L1D --> MD2 --> L0D --> DEC --> GOUT
    end

    L2U --> PROC --> L2D

    class GIN,GOUT gridNode;
    class L0U,L1U,L2U,L2D,L1D,L0D meshNode;
    class ENC,MU0,MU1,MD1,MD2,DEC opNode;
    class PROC procNode;
```

At L0 each mesh node covers a small area and captures local weather. At L2
each node represents a large region, and message passing at that level lets
the model learn how distant weather systems interact. The down-sweep carries
that global context back to fine resolution.

| Tensor | Shape | Description |
|--------|-------|-------------|
| `mesh_L0` | `(B, N_mesh_L0, d_h)` | Finest mesh activations from the encoder (g2m) |
| `mesh_L1` | `(B, N_mesh_L1, d_h)` | Intermediate features after the first up-step |
| `mesh_L2` | `(B, N_mesh_L2, d_h)` | Coarsest representation, iterated by the process block |
| `grid_out` | `(B, N_grid, d_state)` | Final decoded residual after full down-sweep and m2g |

:::{note}
HiLAMParallel uses the same graph files but runs the up-sweep, process,
and down-sweep in parallel rather than sequentially. Architecture is
identical; execution order is the only difference.
:::

---

## 06 · Extension Points

| What to add | Where to look | Key base class |
|-------------|---------------|----------------|
| Alternative file-backed datastore (e.g. NetCDF) | `datastore/base.py`, mirror `datastore/mdp.py` | `BaseDataStore` or `BaseRegularGridDatastore` |
| Custom sampling strategy | `weather_dataset.py` | `torch.utils.data.Dataset` |
| New graph encoder/decoder | `models/base_graph_model.py` | `BaseGraphModel` |
| Additional hierarchical variant | `models/hi_lam.py`, `hi_lam_parallel.py` | `BaseHiGraphModel` + `InteractionNet` |
| New loss or metric | `metrics.py`, `loss_weighting.py` | `metrics.get_metric` helpers |
| New graph topology | [weather-model-graphs](https://github.com/mllam/weather-model-graphs) | — |

---

## 07 · File Map

| File | Description |
|------|-------------|
| `neural_lam/__init__.py` | Package marker, version metadata |
| `neural_lam/config.py` | Typed Pydantic config objects (`NeuralLAMConfig`) |
| `neural_lam/create_graph.py` | CLI to build grid/mesh graphs from a datastore |
| `neural_lam/custom_loggers.py` | W&B and MLflow Lightning logger adapters |
| `neural_lam/interaction_net.py` | Edge-conditioned `InteractionNet` message-passing module |
| `neural_lam/loss_weighting.py` | Per-variable and spatial loss-weight utilities |
| `neural_lam/metrics.py` | Metric factory — MSE, MAE, wMSE, NLL, CRPS |
| `neural_lam/plot_graph.py` | Visualise grid/mesh connectivity from `graphs/` |
| `neural_lam/train_model.py` | Lightning training entry point |
| `neural_lam/utils.py` | MLP builders, graph loading, rank-zero printing |
| `neural_lam/vis.py` | Prediction and diagnostic visualisation |
| `neural_lam/weather_dataset.py` | `WeatherDataset` — bridges datastores and AR rollouts |
| `neural_lam/datastore/base.py` | `BaseDataStore` and `BaseRegularGridDatastore` interfaces |
| `neural_lam/datastore/mdp.py` | `MDPDatastore` — mllam-data-prep Zarr wrapper |
| `neural_lam/datastore/plot_example.py` | Quick-look plotting for datastore samples |
| `neural_lam/datastore/npyfilesmeps/__init__.py` | MEPS numpy-file datastore entry points |
| `neural_lam/datastore/npyfilesmeps/config.py` | Dataclass schema for MEPS file layout |
| `neural_lam/datastore/npyfilesmeps/store.py` | `NpyFilesDatastoreMEPS` with dask-backed loading |
| `neural_lam/datastore/npyfilesmeps/compute_standardization_stats.py` | Precompute MEPS normalisation statistics |
| `neural_lam/models/ar_model.py` | `ARModel` — `LightningModule` for autoregressive training |
| `neural_lam/models/base_graph_model.py` | Encode-process-decode scaffold and output clamping |
| `neural_lam/models/base_hi_graph_model.py` | Base utilities for hierarchical mesh processing |
| `neural_lam/models/graph_lam.py` | Single-level `GraphLAM` |
| `neural_lam/models/hi_lam.py` | `HiLAM` — sequential multilevel message passing |
| `neural_lam/models/hi_lam_parallel.py` | `HiLAMParallel` — parallelised hierarchical variant |

---

## Module Dependencies

```{mermaid}
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#1e3a5f','primaryTextColor':'#e2e8f0','lineColor':'#38bdf8'}}}%%
graph LR
    config["config.py"] --> datastore["datastore/"]
    config --> create_graph["create_graph.py"]
    datastore --> weather_dataset["weather_dataset.py"]
    create_graph --> models["models/"]
    weather_dataset --> models
    models --> train_model["train_model.py"]
    models --> metrics["metrics.py"]
    models --> vis["vis.py"]
```

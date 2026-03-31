# Graph Generation

## Understanding the Graph

Neural-LAM represents the atmosphere as a graph because weather is a
continuous physical field where every location influences its neighbours
in distance-dependent ways. A graph captures this naturally.

### What a Node Represents

Each **node** in the graph corresponds to a physical grid point in your
limited area domain. At that grid point, the atmospheric state is described
by a vector of features: temperature, wind components, pressure, humidity,
and other meteorological variables at that location and time.

The grid nodes form the "ground truth" layer — they are where your input
data lives and where the model's predictions are evaluated. Every grid point
in your spatial domain becomes one node.

### What an Edge Represents

Each **edge** represents a relationship between two grid points that the
model uses for **message passing**. During a forward pass, information flows
along edges: a node aggregates messages from its neighbours, updates its own
state, and sends updated messages outward. This is how the GNN simulates
atmospheric dynamics — weather conditions propagate from location to location
through these learned edge connections.

There are three types of edges:

| Edge Type | Direction | Purpose |
|-----------|-----------|---------|
| **g2m** (grid-to-mesh) | Grid nodes → Mesh nodes | Encode atmospheric state onto the processing mesh |
| **m2m** (mesh-to-mesh) | Mesh nodes → Mesh nodes | Process and propagate information across the domain |
| **m2g** (mesh-to-grid) | Mesh nodes → Grid nodes | Decode processed features back to prediction grid |

### Why a Separate Mesh?

The key insight is that message passing directly between all grid nodes would
be prohibitively expensive (a 400×400 grid has 160,000 nodes). Instead, the
grid is connected to a much smaller **mesh graph** — a coarser set of nodes
covering the same domain. The model encodes grid information onto the mesh,
does its heavy processing on the smaller mesh, then decodes predictions back
to the grid.

### GraphLAM vs. HiLAM: Flat vs. Hierarchical

**GraphLAM** uses a **flat, single-level mesh**. There is one layer of mesh
nodes at one spatial resolution. All mesh-to-mesh message passing happens at
this single scale. This is simpler, faster to train, and sufficient when the
domain is small or when a single resolution captures the relevant weather
dynamics. This corresponds to L1-LAM (one level) and GC-LAM (multiscale
edges on one level).

**HiLAM** uses a **hierarchical, multi-level mesh**. Multiple mesh layers
exist at progressively coarser spatial resolutions. Message passing flows
upward from fine to coarse levels, across each level, and back down from
coarse to fine. Coarser levels capture large-scale weather patterns (synoptic
fronts, jet streams), while finer levels preserve local detail (convection,
coastal effects). This is more powerful for large domains but adds
computational cost and architectural complexity.

### The Full Graph Processing Pipeline

```{mermaid}
graph LR
    A["Grid Points<br/>(atmospheric state)"] --> B["Graph<br/>Construction"]
    B --> C["g2m Edges<br/>(encode)"]
    C --> D["Mesh Nodes<br/>(processing layer)"]
    D --> E["m2m Edges<br/>(message passing)"]
    E --> F["m2g Edges<br/>(decode)"]
    F --> G["Prediction<br/>(forecast at grid)"]
    G -->|"Autoregressive<br/>feedback"| A
```

In a hierarchical graph (HiLAM), the "Mesh Nodes" box expands into multiple
levels with additional upward and downward edges between levels.

## Graph Creation Commands

Graphs are specific to the datastore (the spatial grid of your data). To
create a graph, run `python -m neural_lam.create_graph`. Below are the
commands for each graph type:

**GC-LAM (Multiscale)**
```bash
python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name multiscale
```

**Hi-LAM / Hi-LAM-Parallel (Hierarchical)**
```bash
python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name hierarchical --hierarchical
```

**L1-LAM (1-Level)**
```bash
python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name 1level --levels 1
```

## Graph Directory Format

Generated graph structures are saved within a `graphs` directory located
relative to the `config.yaml` file:

```text
graphs
├── graph1                                  - Directory with a graph definition
│   ├── m2m_edge_index.pt                   - Edges in mesh graph
│   ├── g2m_edge_index.pt                   - Edges from grid to mesh
│   ├── m2g_edge_index.pt                   - Edges from mesh to grid
│   ├── m2m_features.pt                     - Static features of mesh edges
│   ├── g2m_features.pt                     - Static features of grid to mesh edges
│   ├── m2g_features.pt                     - Static features of mesh to grid edges
│   └── mesh_features.pt                    - Static features of mesh nodes
├── graph2
└── ...
```

## Mesh Hierarchy Format

For hierarchical mesh graphs with `L` layers, the files `m2m_edge_index.pt`,
`m2m_features.pt`, and `mesh_features.pt` contain Python lists of length `L`.
For non-hierarchical (L1) graphs, `L == 1` and these are single-entry lists.
Index 0 corresponds to the lowest/finest spatial level.

Hierarchical graphs (`L > 1`) include additional files for inter-level edges:

```text
├── graph1
│   ├── ...
│   ├── mesh_down_edge_index.pt             - Downward edges in mesh graph
│   ├── mesh_up_edge_index.pt               - Upward edges in mesh graph
│   ├── mesh_down_features.pt               - Static features of downward mesh edges
│   ├── mesh_up_features.pt                 - Static features of upward mesh edges
│   ├── ...
```

These files have length `L-1` because they represent connections **between**
adjacent layers. Index 0 characterises the links between Level 1 and Level 2.

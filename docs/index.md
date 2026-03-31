# Neural-LAM

**Graph-based neural weather prediction for Limited Area Modeling.**

:::{important}
Neural-LAM is a research framework. The example data in this repository
is designed to verify your installation works — not to train useful models.
Real weather forecasting requires large regional datasets and substantial
compute resources. See [What Neural-LAM Actually Requires](contributing/what-neural-lam-needs.md)
before planning your project.
:::

Neural-LAM is an open-source repository of graph-based deep learning models
for regional weather forecasting. It represents atmospheric states as graphs
and predicts their evolution over time using message-passing neural networks.

```{admonition} Quick links
:class: tip

- [GitHub Repository](https://github.com/mllam/neural-lam)
- [PyPI Package](https://pypi.org/project/neural-lam/)
- [Slack Community](https://join.slack.com/t/ml-lam/shared_invite/zt-3jyw20n4g-ESRxMPPSijiZ2ZA6Nh8XhA)
```

## Key Features

- **Limited Area Modeling** — high-resolution forecasting over bounded
  geographic regions
- **Graph-based architecture** — atmosphere as a graph, flexible spatial
  resolutions, message-passing dynamics
- **Three model variants** — Graph-LAM, Hi-LAM, Hi-LAM-Parallel
- **Pluggable datastores** — swap data sources without changing model code
- **Research-ready** — built on PyTorch Lightning, supports W&B and MLflow

## Getting Started

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: getting-started/overview
Core concepts, models, and what Neural-LAM does.
:::

:::{grid-item-card} Installation
:link: getting-started/installation
Set up your environment and verify your install.
:::

:::{grid-item-card} User Guide
:link: user-guide/datastores
Datastores, graph generation, training, evaluation.
:::

:::{grid-item-card} Contributing
:link: contributing/architecture
Architecture, development setup, how to contribute.
:::

::::

## Pipeline

```{mermaid}
graph LR
    A[Weather Data] --> B[Datastore]
    B --> C[Graph Construction]
    C --> D[GNN Encoder]
    D --> E[Mesh Processing]
    E --> F[GNN Decoder]
    F --> G[Forecast]
    G -->|Autoregressive| D
```

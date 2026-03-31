[![slack](https://img.shields.io/badge/slack-join-brightgreen.svg?logo=slack)](https://kutt.to/mllam)
[![Linting](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/pre-commit.yml)
[![CPU+GPU testing](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml/badge.svg?branch=main)](https://github.com/mllam/neural-lam/actions/workflows/install-and-test.yml)

<p align="middle">
    <img src="https://raw.githubusercontent.com/mllam/neural-lam/main/figures/neural_lam_header.png" width="700">
</p>

Neural-LAM is a repository of graph-based neural weather prediction models designed for high-resolution Limited Area Modeling (LAM). The codebase is built on [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/pytorch-lightning), utilizing [PyG](https://pyg.org/) for graph neural network implementations. It provides a modular framework for training and evaluating state-of-the-art autoregressive regional weather forecasts.

## Quick Install

```bash
python -m pip install neural_lam
```

For advanced installation options, including `uv`, developer tools, and exact PyTorch CUDA variants, see the [full Installation Guide](docs/getting-started/installation.md).

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](docs/getting-started/overview.md) | Overview and [installation](docs/getting-started/installation.md) |
| [Datastores](docs/user-guide/datastores.md) | Working with data |
| [Graph Generation](docs/user-guide/graph-generation.md) | Creating graph structures |
| [Training](docs/user-guide/training.md) | Training models |
| [Evaluation](docs/user-guide/evaluation.md) | Evaluating models |
| [Contributing](docs/contributing/contributing.md) | How to contribute and [architecture details](docs/contributing/architecture.md) |

## Development and Contributing

See the [Contributing Guide](docs/contributing/contributing.md) for information on how to set up a development environment, run tests, and open pull requests.

## Publications
For a more in-depth scientific introduction to machine learning for LAM weather forecasting see the publications listed here.
As the code in the repository is continuously evolving, the latest version might feature some small differences to what was used for these publications.
We retain some paper-specific branches for reproducibility purposes.


*If you use Neural-LAM in your work, please cite the relevant paper(s)*.

#### [Graph-based Neural Weather Prediction for Limited Area Modeling](https://arxiv.org/abs/2309.17370)
```
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning},
    year={2023}
}
```
See the branch [`ccai_paper_2023`](https://github.com/joeloskarsson/neural-lam/tree/ccai_paper_2023) for a revision of the code that reproduces this workshop paper.

#### [Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks](https://arxiv.org/abs/2406.04759)
```
@inproceedings{oskarsson2024probabilistic,
  title = {Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks},
  author = {Oskarsson, Joel and Landelius, Tomas and Deisenroth, Marc Peter and Lindsten, Fredrik},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024},
}
```
See the branches [`prob_model_lam`](https://github.com/mllam/neural-lam/tree/prob_model_lam) and [`prob_model_global`](https://github.com/mllam/neural-lam/tree/prob_model_global) for revisions of the code that reproduces this paper.
The global and probabilistic models from this paper are not yet fully merged with `main` (see issues [62](https://github.com/mllam/neural-lam/issues/62) and [63](https://github.com/mllam/neural-lam/issues/63)).

## Contact
If you are interested in machine learning models for LAM, have questions about the implementation or ideas for extending it, feel free to get in touch.
There is an open [mllam slack channel](https://join.slack.com/t/ml-lam/shared_invite/zt-2t112zvm8-Vt6aBvhX7nYa6Kbj_LkCBQ) that anyone can join (after following the link you have to request to join, this is to avoid spam bots).
You can also open a github issue on this page.

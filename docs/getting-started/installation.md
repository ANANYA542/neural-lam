# Installation

:::{important}
Neural-LAM is a research framework. The example data in this repository
is designed to verify your installation works — not to train useful models.
Real weather forecasting requires large regional datasets and substantial
compute resources. See [What Neural-LAM Actually Requires](../contributing/what-neural-lam-needs.md)
before planning your project.
:::

This page covers how to install Neural-LAM and set up your environment.

## Prerequisites
* Python $\ge$ 3.10
* Git
* `uv` (recommended) or `pip`

## Installing Neural-LAM

When installing `neural-lam` you have a choice of either installing directly with `pip` or using the `uv` package manager.
We recommend using `uv` as it makes it easy to add/remove packages while keeping versions consistent (it automatically updates the `pyproject.toml` file), makes it easy to handle virtual environments and includes the development toolchain packages installation too.

:::{note}
**Regarding `torch` installation:** Because `torch` creates different package variants for different CUDA versions and CPU-only support, you will need to install `torch` separately if you don't want the most recent GPU variant that also expects the most recent version of CUDA on your system.
:::

For reference on CI/CD pipelines and GPU environments, see our GitHub Actions setup in `.github/workflows/`.

### From source using uv

1. Clone this repository and navigate to the root directory.
2. Install `uv` if you don't have it installed on your system (either with `pip install uv` or following the `uv` install instructions).

:::{note}
If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system), you can skip step 4.
:::

3. Create a virtual environment for uv to use with:
   ```bash
   uv venv --no-project
   ```
4. Install a specific version of `torch`:
   For a CPU-only version:
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   For CUDA 11.1 support:
   ```bash
   uv pip install torch --index-url https://download.pytorch.org/whl/cu111
   ```
   (Find the correct URL for your variant on the PyTorch webpage).

5. Install the dependencies:
   ```bash
   uv pip install .
   ```
   If you will be developing `neural-lam`, we recommend installing the development dependencies in editable mode:
   ```bash
   uv pip install --group dev -e .
   ```

### From source using pip

1. Clone this repository and navigate to the root directory.

:::{note}
If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system), you can skip step 2.
:::

2. Install a specific version of `torch`:
   For CPU-only:
   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   For CUDA 11.1:
   ```bash
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu111
   ```

3. Install the dependencies:
   ```bash
   python -m pip install .
   ```
   If you will be developing `neural-lam`, we recommend installing in editable mode:
   ```bash
   python -m pip install --group dev -e .
   ```

## Verify Your Setup

Once installed, verify your environment is working properly by running the tests (requires installing with `--group dev`):
```bash
pytest -vv -s --doctest-modules
```
*(The first run will attempt to download ~50MB of example data via pooch. If the tests pass, your setup is complete.)*

# Development Environment Setup

Complete setup for contributing to Neural-LAM. Follow every step —
especially Step 3.

## Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| **Python** | 3.10 or newer | Neural-LAM's minimum supported version |
| **Git** | Any recent version | Version control |
| **PDM** | 2.x | Package manager — this is what CI uses |

### Why PDM?

CI uses PDM. Using the same tool locally means your environment matches
CI exactly — if pre-commit and tests pass locally with PDM, they will
pass in CI.

Install PDM if you don't have it:

```bash
pip install pdm
# or
brew install pdm          # macOS with Homebrew
```

## Step 1: Fork and Clone

1. Fork the repository on GitHub: click **Fork** at
   [github.com/mllam/neural-lam](https://github.com/mllam/neural-lam)

2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/neural-lam.git
cd neural-lam
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/mllam/neural-lam.git
```

## Step 2: Install Dependencies

Install all dependencies including development tools:

```bash
pdm install --dev
```

This creates a virtual environment, installs Neural-LAM in editable mode,
and installs all development dependencies (pytest, pre-commit, mypy, etc.).

## Step 3: Install Pre-Commit Hooks

:::{caution}
**This is the step everyone misses.** Without this command, the pre-commit
hooks do **nothing**. You can commit code with style violations, type
errors, and spelling mistakes — and only discover the problems when CI
fails on your pull request.

Run this command now:

```bash
pdm run pre-commit install
```

This installs Git hooks that automatically run all 13 quality checks
every time you run `git commit`. If any check fails, the commit is
blocked until you fix the issue.
:::

## Step 4: Verify Pre-Commit Works

Run all hooks against the entire codebase to confirm everything is set
up correctly:

```bash
pdm run pre-commit run --all-files
```

On a clean checkout of `main`, every hook should pass (green). If any
hook fails, it usually means a dependency is missing — re-run
`pdm install --dev` and try again.

:::{note}
The first run downloads hook environments and may take 1–2 minutes.
Subsequent runs are much faster because the environments are cached.
:::

## Step 5: Run the Tests

```bash
pdm run pytest -vv -s --doctest-modules
```

:::{note}
The first test run downloads approximately 50 MB of example data via
[pooch](https://www.fatiando.org/pooch/). This is a one-time download
cached locally.
:::

All tests should pass on a clean checkout. If they don't, check that
you have PyTorch installed (PDM should handle this, but GPU-specific
variants may need manual installation — see
[Installation](../getting-started/installation.md)).

## Pre-Commit Hooks Reference

All 13 hooks and what they enforce:

| # | Hook | What It Enforces |
|---|------|-----------------|
| 1 | `check-ast` | Validates Python files parse without syntax errors |
| 2 | `check-case-conflict` | Catches filenames that differ only by case (breaks on case-insensitive filesystems) |
| 3 | `check-docstring-first` | Ensures docstrings appear before any code in modules |
| 4 | `check-symlinks` | Detects broken symbolic links |
| 5 | `check-toml` | Validates TOML file syntax (e.g. `pyproject.toml`) |
| 6 | `check-yaml` | Validates YAML file syntax (e.g. CI configs) |
| 7 | `debug-statements` | Catches leftover `import pdb; pdb.set_trace()` and `breakpoint()` calls |
| 8 | `end-of-file-fixer` | Ensures every file ends with exactly one newline |
| 9 | `trailing-whitespace` | Removes trailing whitespace from all lines |
| 10 | `codespell` | Catches common spelling mistakes in code and comments |
| 11 | `black` | Formats Python code to a consistent style (88 char line length) |
| 12 | `isort` | Sorts and groups Python imports consistently |
| 13 | `flake8` | Checks Python code for correctness, style, and best practices |

There is also an optional **mypy** hook for static type checking, which
runs separately since it requires additional type stubs.

## macOS Notes

:::{tip}
**Apple Silicon (M1/M2/M3):** Neural-LAM works natively on Apple Silicon.
PyTorch supports MPS (Metal Performance Shaders) acceleration, though
training is typically done on NVIDIA GPUs via HPC clusters.

**Graphviz for docs builds:** If you plan to build the documentation
locally, you may need Graphviz:

```bash
brew install graphviz
```

This is only needed if sphinx-autoapi generates inheritance diagrams.
The Jupyter Book build itself does not require Graphviz.
:::

## Verify Before PR Checklist

Before pushing your branch and opening a pull request, run these two
commands. **Both must pass** — they are exactly what CI runs:

```bash
# 1. Code quality checks
pdm run pre-commit run --all-files

# 2. Test suite
pdm run pytest -vv -s --doctest-modules
```

If both commands complete with no errors, your PR is ready for review.

:::{seealso}
For the full contribution workflow — finding issues, writing PRs, and
the review process — see [Contributing to Neural-LAM](contributing.md).
:::

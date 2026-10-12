# Kaggle Credit Default Risk

## Environment setup

Install the project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates or updates `.venv` from `pyproject.toml` and `uv.lock`.

Run project tools inside the managed environment without activating it:

```bash
uv run jupyter lab
uv run pytest
```

To activate the environment in your shell instead:

```bash
source .venv/bin/activate
```

Add runtime dependencies with `uv add <package>` and development tools with
`uv add --dev <package>`.
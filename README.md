# options-market-maker

Options market-making research simulator on recorded Deribit data, built on
[pricing-core](https://github.com/divvijchandna/pricing-core).

> Status: pre-release scaffold. This README lists only what exists.

## Install

Requires Python 3.11+ and a sibling checkout of `pricing-core`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../pricing-core
pip install -e ".[dev]"
```

## Development

```bash
ruff check .
ruff format --check .
pytest -m "not slow"
pytest
```

## Data

Market-data recordings are not committed; only a small sample under `data/sample/` is.

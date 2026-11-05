# Data Pipeline (Kedro)

A small [Kedro](https://kedro.org) pipeline that turns the raw tactical tracking CSV into a processed, feature-enriched CSV. It is meant for data scientists to add their own **feature engineering** steps.

## Pipeline overview

```
raw_tactical ──ingest──▶ ingested ──clean──▶ cleaned ──build_features──▶ features ──export──▶ final
 (CSV, input)                                   ▲                                        (CSV, output)
                                  params:features + match_info
```

| Stage | File | Purpose |
|-------|------|---------|
| ingest | [stages/ingest.py](stages/ingest.py) | Typing and sorting of raw frames |
| clean | [stages/clean.py](stages/clean.py) | Drop duplicates and empty columns |
| features | [stages/features.py](stages/features.py) | Derive per-frame features |
| export | [stages/export.py](stages/export.py) | Final shaping of the output |

Everything is wired together in [main.py](main.py).

## Setup and run

From the repository root:

```bash
pip install -r data-pipeline/requirements.txt
python data-pipeline/main.py
```

Output: `data-pipeline/output/tactical_processed.csv`.

## Key concepts

- **Node** – a plain Python function plus its input and output dataset names.
- **Dataset name** – a string (e.g. `"cleaned"`). A node's output name is another node's input name; that is how Kedro knows the execution order.
- **DataCatalog** – registry mapping every dataset name to where its data lives (a CSV file, or in memory). Every name used by a node must exist in the catalog.
- `MemoryDataset` – intermediate data held in RAM, not written to disk.
- `CSVDataset` – data read from or saved to a CSV file.

## Adding a new feature node, step by step

Example: add a node that computes ball acceleration from `ball_speed`.

### Step 1 – Write the function

Create a new file, e.g. `data-pipeline/stages/acceleration.py`.

```python
"""Feature: ball acceleration."""
import pandas as pd


def add_ball_acceleration(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()  # never mutate the input
    dt = out["seconds"].diff()
    out["ball_acceleration"] = out["ball_speed"].diff() / dt
    return out
```

Rules of thumb:
- Take dataframes (and/or dicts of parameters) as arguments, and return the dataframe.
- Do not read or write files inside the function; the catalog handles I/O.
- Return a new dataframe (`df.copy()`) instead of modifying the input.
- The function arguments are positional and follow the order of the node's inputs list (Step 3).

> Alternative: if the feature is small, add it to `build_features` in [stages/features.py](stages/features.py) and skip Steps 2–5. No pipeline or catalog change is needed.

### Step 2 – Import it in `main.py`

```python
from stages.acceleration import add_ball_acceleration
```

### Step 3 – Register the node in `build_pipeline()`

Add a `node(...)` entry in the `pipeline([...])` list. Syntax:

```python
node(function, inputs, outputs, name="unique_name")
```

Insert it where its input already exists. Here we consume `"features"` (produced by the `features` node) and produce a new dataset `"features_accel"`, so it goes after the `features` node:

```python
node(build_features, ["cleaned", "params:features", "match_info"], "features", name="features"),
node(add_ball_acceleration, "features", "features_accel", name="acceleration"),   # NEW
node(export, "features_accel", "final", name="export"),                           # input changed
```

Important:
- Because the new node sits between `features` and `export`, change `export`'s input from `"features"` to `"features_accel"`. Otherwise your node's output is ignored.
- `name` must be unique.
- Inputs may be a string (one input) or a list (several inputs, passed to the function in that order).
- The list order is not what determines execution; dataset names do. Still, keep it in logical order.

### Step 4 – Update the `DataCatalog` (only if needed)

Open `build_catalog()` in [main.py](main.py). Update it **only** in these cases:

| Case | Action |
|------|--------|
| Your node produces a new intermediate dataset (like `features_accel`) | Add `"features_accel": MemoryDataset()` |
| Your node needs a new input file | Add `"my_input": CSVDataset(filepath=...)` |
| Your node needs new parameters | Add `"params:my_feature": MemoryDataset({...})` |
| You want to save the node's output to disk | Use `CSVDataset(filepath=..., save_args={"index": False})` |
| Your node only reuses existing names (e.g. `cleaned`, `match_info`) and writes to an existing one | No change |

For this example:

```python
"features": MemoryDataset(),
"features_accel": MemoryDataset(),   # NEW
```

Using parameters? Define them next to `PARAMS`, then register:

```python
PARAMS = {"smoothing_window": 5}
ACCEL_PARAMS = {"min_dt": 0.04}                      # NEW

"params:accel": MemoryDataset(ACCEL_PARAMS),         # NEW, in the catalog
node(add_ball_acceleration, ["features", "params:accel"], "features_accel", name="acceleration")
```

Use the `params:<name>` naming convention so parameters are easy to spot.

### Step 5 – Run and check

```bash
python data-pipeline/main.py
```

Open `data-pipeline/output/tactical_processed.csv` and confirm the new column (`ball_acceleration`) exists.

## Common errors

| Error | Cause / fix |
|-------|-------------|
| `DatasetNotFoundError` / dataset not found in catalog | A name used in a node isn't registered in `build_catalog()` (Step 4), or has a typo. |
| New column missing from output | `export` (or the next node) still reads the old dataset name; point it at your node's output (Step 3). |
| `Pipeline does not contain...` / cycle error | Two nodes produce the same output name, or inputs/outputs form a loop. Each output name must be unique. |
| `TypeError` about arguments | Number/order of inputs in the node doesn't match the function signature. |
| `ModuleNotFoundError: stages` | Run from the repo root as `python data-pipeline/main.py`, and keep `stages/__init__.py`. |
| `KeyError` on a column | The column was removed upstream (e.g. `clean` drops fully empty columns) or doesn't exist yet; check the order of nodes. |

## Checklist

- [ ] Function written in `stages/` (takes and returns a dataframe, no I/O)
- [ ] Imported in `main.py`
- [ ] Added to `build_pipeline()` with a unique `name`
- [ ] Downstream node (e.g. `export`) reads your output name
- [ ] New dataset names / parameters registered in `build_catalog()`
- [ ] `python data-pipeline/main.py` runs and the output CSV has your feature

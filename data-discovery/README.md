# Data Discovery

## Prerequisites

You need `data/tactical/BAR-ATH_tactical_ds.csv`. If it does not exist, run
[`convert_atd_to_csv.ipynb`](../starter/convert_atd_to_csv.ipynb) first.

## Run

Serve the **repo root** (not `frame-seeker/`), because the page loads the video from `../data/`. The server must support HTTP range requests so the browser can fetch only the parts of the video needed when seeking to a frame, rather than downloading the whole file. Python's `http.server` does not.

```
npx http-server . -p 8000 -c-1
```

If the server is already running, skip the command. Then open
<http://localhost:8000/data-discovery/>.

`-c-1` disables caching so edits show up on a normal reload.

## Usage

The query engine is DuckDB, so queries must use SQL supported by DuckDB. See the
[DuckDB documentation](https://duckdb.org/docs/) to learn more.

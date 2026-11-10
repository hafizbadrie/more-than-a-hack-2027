# Data Pipeline (Kedro)

A small [Kedro](https://kedro.org) pipeline that turns the raw tactical tracking CSV into a processed, feature-enriched CSV. It is meant for data scientists to add their own **feature engineering** steps.

## Pipeline overview

```
raw_tactical ──ingest──▶ clean ──▶ ball, tag and team features ──▶ HMM ──export──▶ final
 (CSV, input)                        ▲                                              (CSV, output)
                            smart tags + parameters
```

| Stage | File | Purpose |
|-------|------|---------|
| ingest | [stages/ingest.py](stages/ingest.py) | Typing and sorting of raw frames |
| clean | [stages/clean.py](stages/clean.py) | Drop duplicates and empty columns |
| features | [stages/ball_focused_features_extraction.py](stages/ball_focused_features_extraction.py), [stages/smart_tags.py](stages/smart_tags.py), [stages/team_focused_features_extraction.py](stages/team_focused_features_extraction.py) | Derive per-frame ball, tag and team features |
| HMM | [stages/hmm_possession.py](stages/hmm_possession.py) | Fit a three-state HMM and add provisional possession estimates |
| export | [stages/export.py](stages/export.py) | Final shaping of the output |

Everything is wired together in [main.py](main.py).

## Setup and run

From the repository root:

```bash
pip install -r data-pipeline/requirements.txt
python data-pipeline/main.py
```

Output: `data-pipeline/output/tactical_processed.csv`.

## HMM possession stage

The stage trains one diagonal-Gaussian HMM across the available input frames and adds
`hmm_possession`: `H` for Barcelona, `A` for Atletic Club, `D` for an estimated stoppage,
and `X` for an observation on which the model abstains. These are **heuristic model outputs**,
not corrected labels; in particular, a set-piece tag is a broad event window, not proof that
the ball is dead. This column is not a submission: the pipeline only outputs input frames,
and a submission must separately cover every video frame.

The emissions use `log1p` of each team's nearest-player ball distance and each team's centroid
speed, standardized on complete observations. Duplicated distance features, tracking-slot IDs,
the mixed-unit team-ball relative positions and `ball_speed` are excluded. Although the
dictionary describes `ball_speed` as metres per second, its current implementation calculates
displacement from normalized coordinates without pitch scaling. The centroid speeds are based
on metre coordinates. The first team-speed observation can be unreliable because the existing
team feature stage treats missing velocity components as zero.

For the supplied 7,501-frame sample, 971 frames (12.9%) lack both ball distances; among the
6,530 frames with both distances, Team A is closer in 61.4% and both teams are over 10 m away
in only 1.3%. In the overlapping public ground-truth window (7,415 sample frames), median
centroid speeds for H, A and D were respectively approximately (1.36, 1.98), (2.05, 1.57) and
(0.55, 0.54) m/s for Barcelona and Atletic Club. The ball was missing on 33.6% of D frames
but also 15.0% of A frames, so missing tracking is not a dead-ball label. These public-window
statistics informed feature choice, not parameter fitting; they are not independent validation.

Initial H and A emission means are medians of the top and bottom 30% of the observed distance
advantage outside set-piece windows. The D mean uses frames in set-piece windows with average
centroid speed in the bottom 40%. These are weak anchors, not labels. Covariances start from
the pooled standardized variance, and the initial diagonal transition probability is 0.997
(roughly 13.3 seconds expected duration at 25 fps); EM can update all model parameters. After
training, states are assigned H/A/D by the minimum total distance to those three initial means.
The mapping can be wrong if learned clusters do not correspond to the anchors. Parameters,
including minimum sample and anchor counts, are in `HMM_PARAMS` in [main.py](main.py); training
fails explicitly if there is insufficient evidence, including when no suitable set-piece
observations exist.

Training breaks sequences at missing feature rows, skipped frame IDs and period boundaries.
Decoding also resets at skipped frames and period boundaries, but bridges missing observations
with neutral emission likelihoods, then reports `X` on those rows. This avoids treating an
untracked ball as dead or falsely joining nonconsecutive frames. The model is fitted and decoded
on the same input; the observed sample is not a held-out test. For that sample, its agreement
with the 7,415 overlapping public labeled frames is about 47.5% when `X` counts as incorrect;
do not interpret this as an estimate of out-of-sample performance.

Run the stage tests with:

```bash
python -m unittest discover -s data-pipeline/tests -v
```

## Output data dictionary

The supplied `output/tactical_processed.csv` contains 288 columns. Repeated player fields are
documented by their naming pattern and the complete list of track slots below; for example,
`2_x`, `2_y`, and `2_s` are the fields for track slot 2. Track-slot numbers are tracker IDs, not
player identities. Positions without a `_m` suffix are normalised to the pitch (0–1); `_x_m`,
`_y_m`, and distance fields are in metres. Missing tracking values remain empty/NaN. The clean
stage removes columns that are entirely empty, so a regenerated output can omit columns that have
no observations.

### Frame and raw tracking columns

| Columns | Definition |
|---------|------------|
| `period_id` | Period/half identifier from the tactical tracking data. |
| `timestamp` | Time of the frame, parsed as a pandas duration. |
| `frame_id` | Tactical video frame number; nominal frame rate is 25 frames per second. |
| `ball_x`, `ball_y` | Ball pitch coordinates on the normalised 0–1 x and y axes. |
| For each slot `2, 3, 4, 6, 9, 11, 12, 13, 15, 16, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 35, 37, 41, 43, 45, 46`: `<slot>_x` | Tracked object's x position, normalised to the pitch (0–1). |
| Same slots as `<slot>_x`: `<slot>_y` | Tracked object's y position, normalised to the pitch (0–1). |
| Same slots as `<slot>_x`: `<slot>_s` | Tracked object's speed in metres per second. |

### Player-slot enrichment columns

| Columns | Definition |
|---------|------------|
| `1_team` through `46_team` (one column per integer slot) | Team assigned to that tracking slot (in this match, `Barcelona` or `Atletic Club`); empty when unavailable. The slot is not a persistent player identity. |
| For each slot `2, 3, 4, 6, 9, 11, 12, 13, 15, 16, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 35, 37, 41, 43, 45, 46`: `<slot>_x_m` | Slot's x position converted from normalised pitch coordinates to metres using the 105.3 m pitch length. |
| Same slots as `<slot>_x_m`: `<slot>_y_m` | Slot's y position converted from normalised pitch coordinates to metres using the 68.0 m pitch width. |
| For each slot `2, 3, 4, 6, 9, 11, 12, 15, 16, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 35, 37, 41, 43, 45, 46`: `<slot>_ball_dist_m` | Euclidean distance in metres from the tracked slot to the ball. This output has no distance column for slot 13. |

### Ball, proximity, and motion features

| Column | Definition |
|--------|------------|
| `hmm_possession` | Heuristic HMM state `H`, `A` or `D`, or `X` when one or more HMM input features are missing. |
| `min_team_a_dist` | Distance in metres from the ball to the nearest tracked Team A player (`Barcelona`) in the frame. |
| `min_team_b_dist` | Distance in metres from the ball to the nearest tracked Team B player (`Atletic Club`) in the frame. |
| `nearest_player_dist` | Smaller of the two teams' nearest-player distances, in metres. |
| `nearest_player_team` | Team of the closest player; ties are assigned to Team A. |
| `possession_advantage` | `min_team_b_dist - min_team_a_dist` in metres; positive means Team A is closer, negative means Team B is closer. This is a distance comparison, not a possession label. |
| `team_a_players_1m`, `team_a_players_2m`, `team_a_players_5m` | Number of Team A players within (or exactly at) 1, 2, or 5 metres of the ball, respectively. |
| `team_b_players_1m`, `team_b_players_2m`, `team_b_players_5m` | Number of Team B players within (or exactly at) 1, 2, or 5 metres of the ball, respectively. |
| `ball_dx`, `ball_dy` | Ball x and y displacement in normalized pitch coordinates since the preceding valid ball observation. |
| `ball_displacement` | Euclidean displacement in normalized coordinates between valid observations; the two axes have different physical scales. |
| `ball_speed` | Normalized-coordinate displacement per second between valid observations, using elapsed frame time; not metres per second. |
| `ball_acceleration` | Change in normalized-coordinate speed per second, not metres per second squared. |
| `ball_direction_deg` | Ball movement direction calculated from normalized coordinate differences (`atan2(ball_dy, ball_dx)`), not a physical pitch angle. |
| `ball_direction_change_deg` | Signed angular change in direction from the preceding valid movement vector, in degrees (−180 to 180). |
| `ball_direction_change_abs_deg` | Absolute angular change from the preceding valid movement vector, in degrees (0 to 180). |
| `team_a_nearest_player_ball_relative_velocity` | Rate of decrease of the nearest Team A player's ball distance, in metres per second; positive means approaching the ball. |
| `team_b_nearest_player_ball_relative_velocity` | Rate of decrease of the nearest Team B player's ball distance, in metres per second; positive means approaching the ball. |
| `nearest_player_ball_relative_velocity` | Relative velocity for the nearest player on either team, in metres per second; positive means approaching the ball. |
| `nearest_player_ball_relative_velocity_team` | Team of the nearest player used for `nearest_player_ball_relative_velocity`; ties go to Team A. |
| `ball_velocity_x` | Horizontal component of ball velocity, in normalized pitch-length units per second. |
| `ball_velocity_toward_team_a_goal` | Ball x velocity projected onto Team A's configured attacking direction; positive means toward that goal. |
| `ball_velocity_toward_team_b_goal` | Ball x velocity projected onto Team B's configured attacking direction; positive means toward that goal. |
| `ball_moving_toward_team_a_goal` | Whether the ball is moving toward Team A's configured goal (`True`/1); missing when velocity is unavailable. |
| `ball_moving_toward_team_b_goal` | Whether the ball is moving toward Team B's configured goal (`True`/1); missing when velocity is unavailable. |

Ball motion features use available valid observations only; missing coordinates are not interpolated.
The first observation or motion interval without a previous valid observation has undefined (NaN)
motion values.

### Smart-tag and match-phase columns

Signed team values use `+1` for the home team (Barcelona), `−1` for the away team (Atletic Club),
and `0` for no recognised team or no signed tag, as applicable. Smart tags are automatic event
intervals, not ground-truth possession labels.

| Column | Definition |
|--------|------------|
| `tag_att_signed` | Signed active attacking-tag code(s), summed and clipped to −1, 0, or +1. Overlapping opposing tags can cancel to 0. |
| `tag_att_trans_signed` | Signed active Attacking Transition tag, with the same home/away convention. |
| `tag_def_signed` | Signed active defending/Defensive Transition tag(s), summed and clipped to −1, 0, or +1. The source does not establish whether the tag's team is the defender or possessor. |
| `tag_n_active` | Number of Smart Tagging intervals active in the frame, including unrecognised tag codes. |
| `tag_box_players` | Maximum active-tag “Max Players in the box” value for the frame (`7+` is represented as 7); 0 when none is provided. |
| `sp_kind` | Active set-piece type: `throw_in`, `corner_kick`, `goal_kick`, `free_kick`, `kick_off_start`, or `kick_off_after_goal`; missing when no recognised set piece is active. |
| `sp_team` | Signed team restarting from the active set-piece tag; 0 when the team is unknown/not applicable. |
| `sp_since` | Seconds since the active set-piece tagging interval began; missing outside such an interval. |
| `evt_t_to_restart` | Seconds until the confirmed restart (negative before, zero at, positive after); missing when a restart time could not be confirmed. |
| `home_attacking` | 1 when an attacking Smart Tag interval assigned to the home team covers this frame; otherwise 0. |
| `home_defending` | 1 when a defending Smart Tag interval assigned to the home team covers this frame; otherwise 0. |
| `away_attacking` | 1 when an attacking Smart Tag interval assigned to the away team covers this frame; otherwise 0. |
| `away_defending` | 1 when a defending Smart Tag interval assigned to the away team covers this frame; otherwise 0. |

### Team shape columns

| Column | Definition |
|--------|------------|
| `team_centroid_x_all_players`, `team_centroid_y_all_players` | Mean x and y coordinates in metres of all tracked players with valid converted coordinates in the frame. |
| `team_centroid_x_barcelona`, `team_centroid_y_barcelona` | Barcelona players' mean x and y coordinates in metres for the frame. |
| `team_centroid_x_atletic_club`, `team_centroid_y_atletic_club` | Atletic Club players' mean x and y coordinates in metres for the frame. |
| `team_dispersion_barcelona` | Mean distance in metres of Barcelona players from Barcelona's frame centroid. |
| `team_dispersion_atletic_club` | Mean distance in metres of Atletic Club players from Atletic Club's frame centroid. |
| `team_centroid_velocity_x_<team>`, `team_centroid_velocity_y_<team>` | Team-centroid velocity components along the pitch x and y axes, in metres per second; `<team>` is `all_players`, `barcelona`, or `atletic_club`. |
| `team_centroid_speed_<team>` | Magnitude of the team's centroid velocity, in metres per second; `<team>` is `all_players`, `barcelona`, or `atletic_club`. |
| `team_average_radial_movement_<team>` | Mean player velocity projected toward the ball for the team, in metres per second. Positive means moving toward the ball; negative means moving away. |
| `team_width_<team>`, `team_depth_<team>` | Team's player-coordinate span in metres: width is max y minus min y, and depth is max x minus min x. |
| `team_tracked_players_<team>` | Number of players assigned to the team with valid x and y coordinates in the frame. |
| `team_movement_coherence_<team>` | Mean-resultant-length measure of the team's moving-player directions, from 0 (directions cancel) to 1 (same direction). Stationary players are excluded. |
| `team_movement_coherence_players_<team>` | Number of moving players with valid directions used to calculate the team's movement coherence. |
| `opponent_team_centroid_distance` | Euclidean distance in metres between the two team centroids. |
| `opponent_team_closing_speed` | Rate of change of the inter-centroid distance in metres per second; positive means the teams are approaching, negative means they are separating. |
| `team_ball_longitudinal_distance_<team>`, `team_ball_lateral_distance_<team>` | Ball position relative to the team's centroid along the team's attacking x direction and the positive y direction, respectively. In the current output, normalized ball coordinates are subtracted from metre-based centroid coordinates, so these values have mixed units and must not be interpreted as physical distances. `<team>` is `barcelona` or `atletic_club`. |

Team velocity, radial movement, and closing-speed features are based on successive valid
observations at the configured 25 frames per second; their first observation in a period, or an
interval without a prior valid observation, has no velocity value.

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

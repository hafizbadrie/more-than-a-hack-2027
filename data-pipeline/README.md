# Data Pipeline (Kedro)

A small [Kedro](https://kedro.org) pipeline that turns the raw tactical tracking CSV into a processed, feature-enriched CSV. It is meant for data scientists to add their own **feature engineering** steps.

## Pipeline overview

```
raw_tactical ──ingest──▶ clean ──▶ ball, tag and team features ──▶ tactical_processed.csv
                                      ▲
                             smart tags + parameters

without_hmm.csv ──▶ HMM ──▶ hybrid possession ──▶ hybrid_results.csv
```

| Stage | File | Purpose |
|-------|------|---------|
| ingest | [stages/ingest.py](stages/ingest.py) | Typing and sorting of raw frames |
| clean | [stages/clean.py](stages/clean.py) | Drop duplicates and empty columns |
| features | [stages/ball_focused_features_extraction.py](stages/ball_focused_features_extraction.py), [stages/smart_tags.py](stages/smart_tags.py), [stages/team_focused_features_extraction.py](stages/team_focused_features_extraction.py) | Derive per-frame ball, tag and team features |
| HMM | [stages/hmm_possession.py](stages/hmm_possession.py) | Fit a three-state HMM and add provisional possession estimates |
| export | [stages/export.py](stages/export.py) | Final shaping of the output |

The cleansing pipeline and catalog are defined in [main.py](main.py); the HMM and
hybrid nodes are defined in [hmm_pipeline.py](hmm_pipeline.py).

The ingest stage exposes `video_frame` as a zero-based column. For the existing tactical CSV,
it converts the one-based source frame counter to `video_frame`; inputs already using
`video_frame` are left unchanged.

## Setup and run

From the repository root:

```bash
pip install -r data-pipeline/requirements.txt
python data-pipeline/main.py
```

Choose either pipeline independently:

```bash
python data-pipeline/main.py --pipeline cleansing
python data-pipeline/main.py --pipeline hmm
```

The default (`--pipeline all`) runs both branches. Cleansing reads the tactical and
smart-tag CSVs and writes `output/tactical_processed.csv`. HMM reads the existing
`output/without_hmm.csv` cache and writes `output/hybrid_results.csv` and the HMM
diagnostic reports. It does not write a separate HMM prediction CSV. These outputs
are overwritten when their respective pipeline runs.

The cache is deliberately separate from the cleansing output: neither cleansing nor
`--pipeline all` updates `without_hmm.csv`. To run HMM on freshly generated features,
after saving any cache you wish to keep, explicitly copy `tactical_processed.csv` to
`without_hmm.csv`, then run `--pipeline hmm`. This can change the HMM baseline.

The player-density extractor now writes missing values when a team has no observed
player-ball distances, instead of treating missing distances as zero players. Existing
cached features retain their old values. Set `"correct_cached_density": True` in
`HMM_PARAMS` to mask those cached density values before fitting; `False` preserves the
existing cached-data baseline. On this match the corrected-data fit substantially changed
the HMM states and degraded its public-window score, so it remains opt-in pending further
validation. Regenerating the feature cache using the extraction stages will also incorporate
the corrected values and change the baseline.

## HMM possession stage

The stage trains one diagonal-Gaussian HMM across the available input frames and adds
`hmm_possession`: `H` for Barcelona, `A` for Atletic Club, `D` for an estimated stoppage,
and `X` for an observation on which the model abstains. These are **heuristic model outputs**,
not corrected labels; in particular, a set-piece tag is a broad event window, not proof that
the ball is dead. This column is not a submission: the pipeline only outputs input frames,
and a submission must separately cover every video frame.

The emissions use the explicitly selected numeric features in `FEATURES` in
[stages/hmm_possession.py](stages/hmm_possession.py). The commented-out motion and velocity
features are not HMM inputs, although they may still be present in the exported dataset.
Other selected features, such as ball displacement and radial movement, do use earlier
tracking observations. Signed values use a signed `log1p` transform before standardization.
Missing values are mean-imputed for EM training and marginalized during decoding; the model
abstains with `X` only when every model feature is missing for a frame.

For the supplied 7,501-frame sample, 971 frames (12.9%) lack both ball distances; among the
6,530 frames with both distances, Team A is closer in 61.4% and both teams are over 10 m away
in only 1.3%. In the overlapping public ground-truth window (7,415 sample frames), median
centroid speeds for H, A and D were respectively approximately (1.36, 1.98), (2.05, 1.57) and
(0.55, 0.54) m/s for Barcelona and Atletic Club. The ball was missing on 33.6% of D frames
but also 15.0% of A frames, so missing tracking is not a dead-ball label. These public-window
statistics informed feature choice, not parameter fitting; they are not independent validation.

Initial H and A emission means are medians of the top and bottom 30% of the observed distance
advantage outside set-piece windows. The D mean uses set-piece-window frames with at least one
observed model feature. These are weak anchors, not labels, and set-piece windows may include
live play. Covariances start from
the pooled standardized variance, and the initial diagonal transition probability is 0.997
(roughly 13.3 seconds expected duration at 25 fps); EM can update all model parameters. After
training, states are assigned H/A/D by the minimum total distance to those three initial means.
The mapping can be wrong if learned clusters do not correspond to the anchors. Parameters,
including minimum sample and anchor counts, are in `HMM_PARAMS` in [main.py](main.py); training
fails explicitly if there is insufficient evidence, including when no suitable set-piece
observations exist.

Training breaks sequences at frames with no observed model features, skipped frame IDs and
period boundaries. Decoding also resets at skipped frames and period boundaries, but marginalizes
missing feature dimensions and reports `X` only for frames with no observed model features. This
avoids treating an untracked ball as dead or falsely joining nonconsecutive frames. The model is
fitted and decoded on the same input; the observed sample is not a held-out test. Previously
reported sample agreement used a different feature set and does not measure this model.

### Controlled HMM experiments

The existing model remains the default. Add any of these optional entries to `HMM_PARAMS`
in [main.py](main.py), changing one at a time and saving the previous outputs first:

| Parameter | Default | Experiment |
|---|---|---|
| `features` | `FEATURES` | A unique list of selected names from `FEATURES`, including `min_team_a_dist` and `min_team_b_dist`. For a geometry ablation, start with those two, then add density or tag features. |
| `min_feature_coverage` | `0.0` | Drop selected features whose observed-frame fraction is below this value, between 0 and 1. The stage fails if an anchor distance is dropped. A fully unobserved feature is always dropped. |
| `training_missing` | `"mean"` | `"complete"` fits EM only on frames with all retained features observed, breaking training sequences around excluded frames. Decoding still marginalizes missing dimensions. This is complete-case training, **not** missing-data EM, and can introduce selection bias or leave too few training/anchor frames. |
| `dead_anchor_max_displacement` | unset | Require observed `ball_displacement` at or below this nonnegative threshold in metres for set-piece dead anchors. It uses the previous valid ball observation, which can be several frames earlier. Missing displacement never qualifies. Choose a threshold based on tracking diagnostics, not the public score. |
| `dead_anchor_max_since` | unset | Also require observed `sp_since` between zero and this many seconds. Set-piece timestamps are approximate. These anchor filters are heuristic, not verified dead-ball labels. |
| `state_mapping` | `"anchors"` | `"distance"` assigns D to the state nearest the dead anchor, then H/A by learned team-distance advantage. This may select the wrong dead state; inspect the state diagnostics. |
| `freeze_transitions` | `False` | `True` prevents EM from changing the initial transition matrix, allowing controlled tests of persistence. |
| `smoothing_half_window` | `0` | A positive integer applies a centred majority vote within each uninterrupted period, frame sequence and non-`X` run. Ties keep the original label. It uses future frames and can erase real short possessions or shift transitions. |

For example, to test smoothing alone, add `"smoothing_half_window": 50` to `HMM_PARAMS`.
At 25 fps this looks two seconds in each direction. This is an experiment, not a recommended
production threshold. Reducing the feature set can increase `X` because a frame is considered
unobserved when all **selected** features are missing.

Each run also writes `output/hmm_states.csv`: assigned state labels, transformed mean
distance advantage, distances to the weak anchors and the learned transition probabilities.
The advantage is the difference of the mean signed-log-transformed distances, not metres.
For comparing fits, inspect this file and `output/hmm_feature_importance.csv` alongside the
metrics; feature importance is emission separation, not causal importance. Do not compare
training likelihoods across different feature sets as though they were the same objective.
All selected features still use Gaussian emissions; the options do not implement a
mixed categorical/count-emission model or a separate dead-ball detector.

### Geometry-first hybrid experiment

The HMM pipeline writes `output/hybrid_results.csv`. It uses the cached frames and
in-memory HMM predictions, and never modifies the HMM label.
The hybrid combines the HMM fallback with the geometry, tag, and smoothing approach from the
standalone [smart-tags submission](../smart-tags/combined_submission.py). The implementation
is adapted to the pipeline's input frame table rather than importing the standalone runner,
which assumes a complete 199,319-frame match. It retains this pipeline's more conservative,
set-piece-gated still/slow/missing-ball stoppage rule. Its rules in
[stages/hybrid_possession.py](stages/hybrid_possession.py) are:

1. Smooth nearest-team geometry labels over `smoothing_half_window` frames. By default, a player
   within `clear_distance_m` makes geometry take precedence; otherwise, an exclusive tag
   signal takes precedence. Set `"geometry_first": True` in `HYBRID_PARAMS` to prefer geometry
   whenever the ball is in bounds and both team distances are observed, even when both players
   are farther than `clear_distance_m`. The tag signal treats a team's attacking phase or the
   opposing team's defending phase as evidence for possession. Where geometry and tags cannot
   decide, an H/A HMM label is the fallback; the HMM's D state is not used as an ownership
   fallback. The result is X when all three sources are inconclusive.
2. Independently detect D for a ball observed outside the pitch for at least one second, or
   during an early configured set-piece window when the ball is missing or still, or tracked
   players are slow. Stillness uses a centred two-second window, scaled to the pitch in metres,
   with 80% ball observations. Player slow uses the centred median of reported player speeds,
   requiring `min_tracked_players` observations per frame.
3. Merge dead-ball gaps up to `dead_merge_gap_s`, then discard dead runs shorter than
   `dead_min_duration_s`. Finally, smooth H/A labels with a centred majority vote and merge
   possessions shorter than `min_possession_s`. D and X labels are preserved; smoothing does
   not cross dead-ball, period, or missing-frame boundaries, and ties keep the original label.

`HYBRID_PARAMS` in [main.py](main.py) contains the match-specific pitch, frame rate,
thresholds, set-piece cutoffs, and minimum-run settings. The cutoff values are based on the
tracking analysis in [../smart-tags/TECHNICAL.md](../smart-tags/TECHNICAL.md), but are not
independently validated for this implementation. Goal kicks and other unlisted set-piece kinds
have no early-window dead detector; the out-of-pitch signal still applies. Missing ball tracking
alone is not evidence of a stoppage outside a configured set-piece window. The standalone
submission's fixed D interval for the documented half-time block is intentionally not copied:
it is tied to absolute frame numbers. These rules remain heuristics, and the centred windows use
future frames, so this is not a real-time model.

The geometry-first option is an experiment, not the default: an ablation on the public window
improved spell-F1 from 0.400 to 0.494, but the window has only 17 possessions and was used to
identify the change. Validate it on independent matches or held-out clips before selecting it.
It does not change dead-ball detection, and distant geometry can be unreliable.

Compare saved complete prediction exports with
`python data-pipeline/compare_hmm_runs.py /path/to/first.csv /path/to/second.csv`.
Inspect class-level and spell metrics, especially dead-ball recall and abstentions. Neither
the new hybrid nor the density-corrected HMM is guaranteed to outperform the baseline on
the full match; the public window is too small for model selection.

### Diagnosing and comparing HMM runs

The pipeline writes `output/hmm_convergence.csv` on each run. Its rows contain the EM
iteration, recorded training log-likelihood, change from the previous iteration, final
status, tolerance, and iteration limit. The first delta is blank. The logged likelihood is
the value recorded by `hmmlearn` before that iteration's parameter update, not a held-out
score. `converged` means a nonnegative final improvement below the tolerance;
`likelihood_decreased` means a decrease greater than the library's numerical warning
threshold; `iteration_limit` means the limit was reached without either condition.
`not_converged` covers other stops, including tiny negative changes. A warning does not
prevent output, so inspect it before relying on predictions.

To compare two runs, save each run's `output/hybrid_results.csv` and
`output/hmm_convergence.csv` under distinct names before running again. The default paths
are overwritten on each pipeline run. Keep the input and ground truth the same, and record
the `HMM_PARAMS` used for each run. Then run, for example:

```bash
python data-pipeline/compare_hmm_runs.py /path/to/first.csv /path/to/second.csv
```

Both predictions must be complete, ordered `frame,label` CSVs matching the ground truth's
frame count. Use `--gt /path/to/ground_truth.csv` to select another complete ground truth.
The comparison reports prediction disagreements and uses the existing evaluator's accuracy,
macro-F1, and possession-spell F1 on graded frames. Predictions of `X` count against
frame-level scores; the evaluator fills abstentions for spell scoring. Check the graded
abstention count as well. The public labels cover only five minutes and are not an
independent held-out evaluation of a model trained on the same match; small metric differences
may be noise. Changing `random_state` alone does not provide a meaningful initialization
comparison here because the HMM's starting parameters are explicitly supplied.

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
| `video_frame` | Tactical video frame number; nominal frame rate is 25 frames per second. |
| `ball_x`, `ball_y` | Ball pitch coordinates on the normalised 0–1 x and y axes. |
| For each slot `2, 3, 4, 6, 9, 11, 12, 13, 15, 16, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 35, 37, 41, 43, 45, 46`: `<slot>_x` | Tracked object's x position, normalised to the pitch (0–1). |
| Same slots as `<slot>_x`: `<slot>_y` | Tracked object's y position, normalised to the pitch (0–1). |
| Same slots as `<slot>_x`: `<slot>_s` | Tracked object's speed in metres per second. |

### Player-slot enrichment columns

| Columns | Definition |
|---------|------------|
| `1_team` through `46_team` (one column per integer slot) | Team assigned to that tracking slot (in this match, `Barcelona` or `Atletic Club`); empty when unavailable. The slot is not a persistent player identity. |
| For each slot `2, 3, 4, 6, 9, 11, 12, 13, 15, 16, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 35, 37, 41, 43, 45, 46`: `<slot>_x_m` | Slot's x position converted from normalised pitch coordinates to metres; the supplied output uses a 105 m scale. |
| Same slots as `<slot>_x_m`: `<slot>_y_m` | Slot's y position converted from normalised pitch coordinates to metres using the 68.0 m pitch width. |
| For each slot `2, 3, 4, 6, 9, 11, 12, 15, 16, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 35, 37, 41, 43, 45, 46`: `<slot>_ball_dist_m` | Euclidean distance in metres from the tracked slot to the ball. This output has no distance column for slot 13. |

### Ball, proximity, and motion features

| Column | Definition |
|--------|------------|
| `hmm_possession` | Heuristic HMM state `H`, `A` or `D`, or `X` when all HMM input features are missing. |
| `min_team_a_dist` | Distance in metres from the ball to the nearest tracked Team A player (`Barcelona`) in the frame. |
| `min_team_b_dist` | Distance in metres from the ball to the nearest tracked Team B player (`Atletic Club`) in the frame. |
| `nearest_player_dist` | Smaller of the two teams' nearest-player distances, in metres. |
| `nearest_player_team` | Team of the closest player; ties are assigned to Team A. |
| `possession_advantage` | `min_team_b_dist - min_team_a_dist` in metres; positive means Team A is closer, negative means Team B is closer. This is a distance comparison, not a possession label. |
| `team_a_players_1m`, `team_a_players_2m`, `team_a_players_5m` | Number of Team A players within (or exactly at) 1, 2, or 5 metres of the ball, respectively. |
| `team_b_players_1m`, `team_b_players_2m`, `team_b_players_5m` | Number of Team B players within (or exactly at) 1, 2, or 5 metres of the ball, respectively. |
| `ball_dx`, `ball_dy` | Ball x and y displacement in metres since the preceding valid ball observation. |
| `ball_displacement` | Euclidean displacement in metres between valid observations. |
| `ball_speed` | Original dataset ball speed in metres per second, when supplied. |
| `ball_speed_calc` | Calculated average ball speed in metres per second between valid observations, using elapsed frame time. |
| `ball_acceleration` | Change in calculated ball speed per second, in metres per second squared. |
| `ball_direction_deg` | Ball movement direction in degrees calculated from metre-coordinate differences (`atan2(ball_dy, ball_dx)`). |
| `ball_direction_change_deg` | Signed angular change in direction from the preceding valid movement vector, in degrees (−180 to 180). |
| `ball_direction_change_abs_deg` | Absolute angular change from the preceding valid movement vector, in degrees (0 to 180). |
| `team_a_nearest_player_ball_relative_velocity` | Rate of decrease of the nearest Team A player's ball distance, in metres per second; positive means approaching the ball. |
| `team_b_nearest_player_ball_relative_velocity` | Rate of decrease of the nearest Team B player's ball distance, in metres per second; positive means approaching the ball. |
| `nearest_player_ball_relative_velocity` | Relative velocity for the nearest player on either team, in metres per second; positive means approaching the ball. |
| `nearest_player_ball_relative_velocity_team` | Team of the nearest player used for `nearest_player_ball_relative_velocity`; ties go to Team A. |
| `ball_velocity_x` | Horizontal component of calculated ball velocity in metres per second. |
| `ball_velocity_toward_team_a_goal` | Ball x velocity projected onto Team A's configured attacking direction; positive means toward that goal. |
| `ball_velocity_toward_team_b_goal` | Ball x velocity projected onto Team B's configured attacking direction; positive means toward that goal. |
| `ball_moving_toward_team_a_goal` | Whether the ball is moving toward Team A's configured goal (`True`/1); missing when velocity is unavailable. |
| `ball_moving_toward_team_b_goal` | Whether the ball is moving toward Team B's configured goal (`True`/1); missing when velocity is unavailable. |

Ball motion features use available valid observations only; missing coordinates are not interpolated.
The first observation or motion interval without a previous valid observation has undefined (NaN)
motion values. Ball coordinates are converted to metres using the affine mapping fitted from
normalized and metre-valued player coordinate pairs. The HMM uses `ball_speed_calc`, not the
original `ball_speed`. Rebuild older feature caches with `--pipeline cleansing` or `--pipeline all`
before running `--pipeline hmm`; earlier caches have different motion units and lack `ball_speed_calc`.

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
| `team_ball_longitudinal_distance_<team>`, `team_ball_lateral_distance_<team>` | Ball position relative to the team's centroid along the team's attacking x direction and the positive y direction, respectively, in metres. `<team>` is `barcelona` or `atletic_club`. |

For radial movement and team-ball distances, normalized ball coordinates are converted
to the player coordinate system using an affine fit of the available player normalized
and metre coordinates for each axis. Each axis needs at least two finite pairs with
distinct normalized values; otherwise the stage raises an error. Regenerate cached
cleansing output to incorporate changes to these features.

Attack-facing features use `period_direction_multipliers` in the ball- and
team-focused parameters. It maps each `period_id` to `+1` or `-1`; the configured
team directions are multiplied by that value. The match defaults use `{1: 1, 2: -1}`
to account for the halftime switch. If a configured period is absent from the mapping,
the feature stage raises an error rather than applying a potentially incorrect direction.

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

Example: add a node that computes the absolute difference between the two teams'
nearest-player distances.

### Step 1 – Write the function

Create a new file, e.g. `data-pipeline/stages/distance_gap.py`.

```python
"""Feature: nearest-player distance gap."""
import pandas as pd


def add_distance_gap(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["distance_gap"] = (out["min_team_a_dist"] - out["min_team_b_dist"]).abs()
    return out
```

Rules of thumb:
- Take dataframes (and/or dicts of parameters) as arguments, and return the dataframe.
- Do not read or write files inside the function; the catalog handles I/O.
- Return a new dataframe (`df.copy()`) instead of modifying the input.
- The function arguments are positional and follow the order of the node's inputs list (Step 3).

### Step 2 – Import it in `main.py`

```python
from stages.distance_gap import add_distance_gap
```

### Step 3 – Register the node in `build_cleansing_pipeline()`

Add a `node(...)` entry in the `pipeline([...])` list. Syntax:

```python
node(function, inputs, outputs, name="unique_name")
```

Insert it where its input already exists. Here we consume
`"with_team_position_relative_to_ball"` and produce a new dataset `"features_gap"`,
immediately before export:

```python
node(add_distance_gap, "with_team_position_relative_to_ball", "features_gap", name="distance_gap"),
node(export, "features_gap", "final", name="export"),
```

Important:
- Change `export`'s input to `"features_gap"`; otherwise your node's output is ignored.
- `name` must be unique.
- Inputs may be a string (one input) or a list (several inputs, passed to the function in that order).
- The list order is not what determines execution; dataset names do. Still, keep it in logical order.

### Step 4 – Update the `DataCatalog` (only if needed)

Open `build_catalog()` in [main.py](main.py). Update it **only** in these cases:

| Case | Action |
|------|--------|
| Your node produces a new intermediate dataset (like `features_gap`) | Add `"features_gap": MemoryDataset()` |
| Your node needs a new input file | Add `"my_input": CSVDataset(filepath=...)` |
| Your node needs new parameters | Add `"params:my_feature": MemoryDataset({...})` |
| You want to save the node's output to disk | Use `CSVDataset(filepath=..., save_args={"index": False})` |
| Your node only reuses existing names (e.g. `cleaned`, `match_info`) and writes to an existing one | No change |

For this example:

```python
"features_gap": MemoryDataset(),
```

Using parameters? Define them next to `PARAMS`, then register:

```python
PARAMS = {"smoothing_window": 5}
MY_FEATURE_PARAMS = {"threshold": 0.04}              # NEW

"params:my_feature": MemoryDataset(MY_FEATURE_PARAMS),  # NEW, in the catalog
node(my_feature, ["with_team_position_relative_to_ball", "params:my_feature"], "features_gap", name="my_feature")
```

Use the `params:<name>` naming convention so parameters are easy to spot.

### Step 5 – Run and check

```bash
python data-pipeline/main.py --pipeline cleansing
```

Open `data-pipeline/output/tactical_processed.csv` and confirm the new column (`distance_gap`) exists.

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
- [ ] Added to `build_cleansing_pipeline()` with a unique `name`
- [ ] Downstream node (e.g. `export`) reads your output name
- [ ] New dataset names / parameters registered in `build_catalog()`
- [ ] `python data-pipeline/main.py --pipeline cleansing` runs and the output CSV has your feature

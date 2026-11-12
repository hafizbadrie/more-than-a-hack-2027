# Hybrid possession: reference

`hybrid_possession.py` (the module `hybrid_pipeline.py`) turns four sources of evidence into one label per frame. The method is configurable in three parts:

1. **Components**: one feature per approach (`hmm`, `geometry`, `clear_geometry`, `tags`, `dead_ball`).
2. **Selection rules**: an ordered list that decides which component wins on each frame.
3. **Smoothing steps**: an ordered list applied to the selected labels.

Output: a copy of the input with one extra column, `hybrid_possession`, holding `H`, `A`, `D` or `X`.

| Label | Meaning |
|---|---|
| `H` | home team (team A) has the ball |
| `A` | away team (team B) has the ball |
| `D` | dead ball: play is stopped |
| `X` | unknown: no source could decide |

The scorer treats the labels differently: `X` is carried forward from the previous label for the spell metric and counts as a miss for macro F1; `D` ends a possession.

## 1. Configuration (`HYBRID_PARAMS` in `main.py`)

```python
HYBRID_PARAMS = {
    "fps": 25,
    "pitch_m": (105.3, 68.0),
    "components": {                     # a component is ON when it is listed; delete it to remove it
        "hmm": {},
        "geometry": {"smoothing_half_window": 50},
        "clear_geometry": {"smoothing_half_window": 50, "clear_distance_m": 3.0},
        "tags": {},
        "dead_ball": {
            "still_distance_m": 0.5,
            "slow_speed_mps": 0.7,
            "dead_min_duration_s": 2.0,
            "dead_merge_gap_s": 2.0,
            "min_tracked_players": 5,
            "set_piece_cutoffs_s": {
                "corner_kick": 5.6, "free_kick": 5.2, "throw_in": 5.8, "kick_off_after_goal": 4.0,
            },
        },
    },
    "selection": {
        "rules": [                      # first match wins, top to bottom
            {"use": "dead_ball"},
            {"use": "clear_geometry"},
            {"use": "geometry"},        # geometry above tags = geometry_first: True
            {"use": "tags"},
            {"use": "hmm"},
        ]
    },
    "smoothing": {
        "dead_handling": "gap",         # "split" = each side of a stoppage is smoothed on its own
        "steps": [
            {"type": "majority_vote", "half_window": 50},
            {"type": "merge_short", "min_possession_s": 3.0},
        ],
    },
}
```

The set-piece cutoffs were measured from tracking around tagged set pieces; the other values are judgement values. The keys of `set_piece_cutoffs_s` must equal the values of the `sp_kind` column from the Smart Tag features, or that set piece never triggers dead ball (check with `df["sp_kind"].dropna().unique()`).

The old flat parameter dict (`clear_distance_m`, `geometry_first`, `min_possession_s`, ...) is still accepted and converted (`config_from_legacy_params`), with identical labels. `DEFAULT_CONFIG` inside the module is only the reference used by the tests; the pipeline uses the config it is given.

### Global keys

| Key | Meaning |
|---|---|
| `fps` | frames per second (int > 0) |
| `pitch_m` | pitch length and width in metres (turns the 0-1 ball position into metres) |

### Component parameters

| Component | Parameter | Meaning | Effect of raising it |
|---|---|---|---|
| `hmm` | none | the HMM label where it says H or A (its D and X are ignored) | |
| `geometry` | `smoothing_half_window` | frames each side for the majority vote of the nearest-team label (50 = 2 s; 0 = off) | smoother, slower to react |
| `clear_geometry` | `smoothing_half_window` | same as above | |
| | `clear_distance_m` | a player this close (m) to the ball makes geometry decisive | decides more often |
| `tags` | none | Smart Tag attacking / defending flags | |
| `dead_ball` | `still_distance_m` | ball moved less than this (m) within +-1 s means "still" | more dead ball |
| | `slow_speed_mps` | median player speed below this within +-1 s means "players slow" | more dead ball |
| | `dead_min_duration_s` | dead stretches shorter than this are dropped | fewer, longer stretches |
| | `dead_merge_gap_s` | gaps this short between dead stretches are closed | stretches join up |
| | `min_tracked_players` | the per-frame median speed needs at least this many tracked players | stricter |
| | `set_piece_cutoffs_s` | `{kind: s}`: dead ball is only allowed in the first `s` seconds of that kind of set piece | longer window |

### Smoothing steps

| Step | Parameter | Meaning |
|---|---|---|
| `majority_vote` | `half_window` | every H/A frame takes the majority H/A label within +-`half_window` frames; X frames abstain; ties keep the frame's label |
| `merge_short` | `min_possession_s` | while a possession is shorter than this, the shortest is merged into a neighbour (its only neighbour at a block edge, otherwise the longer one; a tie goes to the earlier) |

Steps run in the order listed, one block at a time (see `dead_handling` below). Set `half_window` or `min_possession_s` to 0 to switch a step off.

## 2. Inputs

One row per frame, any order. Rows are sorted internally and the original order is restored at the end. Only the columns of the components you list are required.

| Column | Needed by | Meaning |
|---|---|---|
| `period_id`, `video_frame` | always | integer frame number, unique within a period. Gaps are allowed. |
| `ball_x`, `ball_y` | geometry, clear_geometry, dead_ball | ball position, normalised 0-1 over the pitch. NaN = not tracked. |
| `min_team_a_dist`, `min_team_b_dist` | geometry, clear_geometry | distance (m) from each team's nearest player to the ball |
| `{n}_s` | dead_ball | player speed (m/s), one column per slot. Every column named digits + `_s` is used. |
| `sp_kind`, `sp_since` | dead_ball | set piece in progress and seconds since it started |
| `home_attacking`, `away_attacking`, `home_defending`, `away_defending` | tags | 0/1 phase flags |
| `hmm_possession` | hmm | H / A / D / X per frame |

A `ValueError` is raised for missing columns, missing or non-integer frame numbers, duplicate `(period_id, video_frame)` pairs, negative distances or speeds, tag flags other than 0/1, HMM labels other than H/A/D/X, an unknown component or smoothing step, a selection rule that uses a component that is not listed, and invalid parameters.

## 3. The stages

`run(df, config)` runs the stages in order and keeps every intermediate result.

**Stage 0, `build_context`.** Checks the input and the config. Sorts by period and frame. Splits the rows into *sequences*: runs of consecutive frames inside one period. No vote, window or smoothing crosses a gap or a period change.

**Stage 1, `compute_components`.** Each listed component returns one label per frame: `H` or `A` where it has an opinion, `X` where it abstains (`D` or `X` for `dead_ball`).

| Component | Definition |
|---|---|
| `hmm` | the HMM label where it is H or A |
| `geometry` | the team with the nearer player to the ball (ties go to H) where the ball is inside the pitch and both distances exist, then majority-smoothed |
| `clear_geometry` | the geometry label, only where the nearest player is within `clear_distance_m` |
| `tags` | H when home is attacking or away is defending; A when away is attacking or home is defending; X if both or neither hold |
| `dead_ball` | `D` where `outside OR (event_window AND (still OR slow OR ball not seen))`, cleaned with `dead_merge_gap_s` and `dead_min_duration_s` |

Signals behind `dead_ball`: `outside` is the ball seen but outside the pitch box for at least 1 s; `still` is at least 80% of the +-1 s window tracked and the ball moved less than `still_distance_m` in both x and y; `slow` is the rolling median over +-1 s of the per-frame median player speed below `slow_speed_mps`; `event_window` is `sp_kind` a key of `set_piece_cutoffs_s` and `0 <= sp_since <= cutoff`. They are kept in `result.ctx.signals`.

**Stage 2, `select_labels`.** For every frame the first rule whose component has an opinion decides. A rule is `{"use": name}` and may add `"when_known": [names that must have an opinion]` and `"when_unknown": [names that must not]`. A component has an opinion when its label is not `X`. Frames that no rule labels stay `X` with source `none`.

With the rules above this is the old order of priority: dead ball, then clear geometry, then geometry, then tags, then the HMM. `geometry_first: False` of the old code is `tags` above `geometry`.

**Stage 3, `smooth_labels`.** Applies the smoothing steps to the selected labels, block by block, then writes `D` back on the dead frames.

- `dead_handling: "gap"` (default): dead frames are treated as `X` while smoothing. The smoothing never sees the stoppage, so a short possession after a stoppage can be merged into the one before it.
- `dead_handling: "split"`: dead frames cut the labels into blocks, each smoothed on its own, so nothing is merged across a stoppage. It only affects frames next to dead-ball stretches.

## 4. Looking at what happened to a frame

```python
result = run(df, HYBRID_PARAMS)
result.table().head(300)        # one row per frame: every component, the deciding rule, selected / smoothed / final
result.components_table()       # one column per component (hp_<name>), in the ORIGINAL row order
result.source_share()           # share of frames decided by each rule
result.ctx.signals["still"]     # dead-ball signals: still, slow, outside, event_window, dead_raw, clear
features(df, HYBRID_PARAMS)     # just the component features, for Kedro
```

| Question | Where |
|---|---|
| Why is this frame H when the nearest player is away? | `source`: if it is `hmm` or `tags`, geometry was overruled; look at `clear_geometry`. |
| Why is there no D around this set piece? | the `dead_ball` column and `result.ctx.signals`; check the `sp_kind` names. |
| Why did a short possession disappear? | compare `selected` with `smoothed`: `merge_short` merged it. |
| Many X frames? | `source == "none"`: no rule had an opinion. |

## 5. Changing the method

- **Remove an approach:** delete it from `components` and its rule from `selection`. Its input columns are then no longer required.
- **Re-rank approaches:** reorder the rules. `{"use": "tags", "when_unknown": ["geometry"]}` uses the tags only where geometry abstains.
- **Add an approach:** write a function and register it, then list it under `components` and add a rule:
  ```python
  @component("my_feature", requires=["some_column"])
  def my_feature(ctx, params):
      return labels            # array of "H" / "A" / "X", one per row of ctx.ordered
  ```
- **Add a smoothing step:** `@smoother("my_step")` with `def my_step(local_labels, step, ctx)`, then list `{"type": "my_step", ...}` under `steps`.
- **Change the smoothing:** reorder, remove or re-parameterise the steps.

## 6. Behaviours worth knowing

- **HMM D and X are ignored.** Only the HMM's H and A labels are used. D comes from `dead_ball` only.
- **Ties go to H** in the nearest-player comparison.
- **Unknown frames abstain** in every vote.
- **Sequences:** gaps in `video_frame` and period changes split the data, and nothing is smoothed across them.
- **Where `dead_ball` ranks matters:** it is a normal rule. Put it first (as above) and dead frames are always `D`; put it lower and an earlier rule can take the frame.

## 7. Tests

`test_hybrid_pipeline.py` checks the configurable code against the previous single-function version (`hybrid_possession.py` before this change, kept as `orig_hybrid.py` lineage) on random data: 48 parameter combinations (shuffled rows, a frame gap, missing values, ball outside the pitch, `split`), plus the config switches (remove, reorder, conditions, smoothing steps, registering a new component and smoother) and the input errors.

```bash
python test_hybrid_pipeline.py     # ends with: ALL TESTS PASSED
```

The test imports `hybrid_possession` (the previous version) as the reference, so run it in the folder that holds both files.
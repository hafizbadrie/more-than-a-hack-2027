# Hybrid possession: reference

`hybrid_possession.py` (the module `hybrid_pipeline.py`) turns four sources of evidence into one label per frame. The method is configurable in three parts:

1. **Components**: one feature per approach (`hmm`, `geometry`, `clear_geometry`, `tags`, `dead_ball`; optional: `ball_lost`, `physical_dead`, `touches`, and the rejected `last_touch` and `hmm_dead`).
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
        "physical_dead": {              # listed but NOT used by a rule (see below)
            "still_distance_m": 0.5, "slow_speed_mps": 1.0, "slow_window_s": 5.0, "slow_top_k": 5,
            "min_tracked_players": 5, "agree_with": ["slow"], "min_duration_s": 5.0, "merge_gap_s": 2.0,
        },
        "touches": {                    # feeds snap_to_touch; not a selection rule
            "window_frames": 3, "min_delta_mps": 3.0, "peak_s": 0.4,
            "max_player_dist_m": 2.0, "team_margin_m": 0.5,
        },
        "last_touch": {                 # listed but NOT used by a rule (rejected)
            "window_frames": 3, "min_delta_mps": 3.0, "peak_s": 0.4,
            "max_player_dist_m": 2.0, "team_margin_m": 0.5, "hold_s": 4.0,
        },
        "ball_lost": {                  # D where the ball is untracked for a stoppage-sized stretch
            "min_s": 5.0, "max_s": 20.0, "min_tracked_players": 5, "max_median_speed_mps": 2.0,
        },
    },
    "selection": {
        "rules": [                      # first match wins, top to bottom
            {"use": "dead_ball"},
            {"use": "ball_lost"},
            # {"use": "physical_dead"}, # optional, right after dead_ball / ball_lost: no gain on the public slice
            {"use": "clear_geometry"},
            # {"use": "last_touch"},    # rejected: between clear_geometry and geometry it did not help
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
            {"type": "snap_to_touch", "max_shift_s": 3.0},   # needs `touches`; keep it LAST
        ],
    },
}
```

This is the configuration shipped in `main.py` (`HYBRID_PARAMS`, passed to the pipeline as `params:hybrid`; the result is written to `output/hybrid_results.csv`). **Every listed component is computed, whether or not a rule uses it** (and its input columns are required). `physical_dead` and `last_touch` are listed in `main.py` but have no active rule, so they cannot change a label; they only cost run time. To switch `physical_dead` on, uncomment its rule; to remove either, delete its entry under `components`. `touches` must stay listed because `snap_to_touch` reads it.

The set-piece cutoffs were measured from tracking around tagged set pieces; the other values are judgement values. The `ball_lost` bounds (5-20 s, ceiling 2.0 m/s) were fixed from label-free whole-match statistics (section 8). The keys of `set_piece_cutoffs_s` must equal the values of the `sp_kind` column from the Smart Tag features, or that set piece never triggers dead ball (check with `df["sp_kind"].dropna().unique()`).

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
| `ball_lost` | `min_s`, `max_s` | an untracked-ball run is D when its length is between these (s) | wider band, more D |
| | `max_median_speed_mps` | and the median player speed inside it is at most this (optional; 2.0) | more D, less precise |
| | `min_tracked_players` | the median speed needs this many tracked players | stricter |

`ball_lost` works anywhere in the match, not only in set-piece windows. A run that touches the start or end of a gap-free sequence is ignored (its length is unknown). Why the 5-20 s band: on this match the ball is untracked in short occlusions (< 2 s), stoppage-sized runs (5-20 s) and tracking outages (>= 20 s, up to several minutes), and no run is 2-5 s long. Without the speed ceiling D precision on the public slice fell to 0.60; with 2.0 it is 0.70. The runs are kept in `result.ctx.signals["ball_lost_runs"]`.

### Optional components (off unless listed)

| Component | Parameter | Meaning |
|---|---|---|
| `physical_dead` | `still_distance_m`, `slow_speed_mps`, `min_tracked_players` | same signals as `dead_ball` |
| | `slow_window_s` | window (+- s) over which the speed is smoothed (default 1.0); a longer window asks for SUSTAINED slowness |
| | `slow_top_k` | speed = mean of the k FASTEST tracked players instead of the median (default off). In live play a few players always run; in a stoppage even the fastest walk. The median fires on 7-9% of live frames at 1.0 m/s, the top-5 on 0.7% |
| | `agree_with` | list of `still`, `slow`, `team_still`; with `agree_mode` `any` / `all` they say how the signals combine |
| | `require` | signals that must ALL hold on top of `agree_with` (e.g. `require ["still"]`, `agree_with ["slow","team_still"]`) |
| | `min_duration_s`, `merge_gap_s` | clean-up of the dead stretches; `team_speed_columns` + `team_still_mps` only if `team_still` is used |
| | | dead ball ANYWHERE in the match, not only in set-piece windows. Put `{"use": "physical_dead"}` right after `dead_ball` in the rules. Unknown parameter names raise a `ValueError` (a misspelt `slow_window_s` used to be ignored silently). Choose the setting with `slow_window_check.py` / `hmm_dead_check.py` (accept only if it fires on <= 1% of clearly-live frames). The accepted setting is the one in the example above (top-5, 5 s window, 1.0 m/s: 0.67% of live frames, +154 s of D); it is neutral on the public slice. `hmm_dead` is the same with the HMM's D as an extra condition; on this match the HMM's D fires on 94% of live frames, so do not use it |
| `touches` | `window_frames` | frames before / after t used for the ball velocity (3) |
| | `min_delta_mps` | smallest sudden velocity change that counts as a touch (3.0) |
| | `peak_s` | the change must be the maximum within +- this many seconds (0.4) |
| | `max_player_dist_m` | a player of either team must be this close to the ball (2.0) |
| | `team_margin_m` | one team's nearest player must be this much closer than the other's, else the touch has no team (0.5) |
| | | H / A on the frames where a player plays the ball, X elsewhere. Not a selection rule: it feeds `snap_to_touch`. Check with `touch_check.py` |

### Smoothing steps

| Step | Parameter | Meaning |
|---|---|---|
| `majority_vote` | `half_window` | every H/A frame takes the majority H/A label within +-`half_window` frames; X frames abstain; ties keep the frame's label |
| `merge_short` | `min_possession_s` | while a possession is shorter than this, the shortest is merged into a neighbour (its only neighbour at a block edge, otherwise the longer one; a tie goes to the earlier) |
| `snap_to_touch` | `max_shift_s` | moves each H<->A change to the first touch of the team that gains the ball, within +-`max_shift_s`; no touch, no move. Needs the `touches` component. Put it LAST. |

`smoothing` also accepts `dead_lead_s` / `dead_trail_s` (default 0): grow each dead stretch earlier / later. On the public slice a lead of 1 s cost 8 points of D precision and a lead of 2 s cost 15, so it is off.

Steps run in the order listed, one block at a time (see `dead_handling` below). Set `half_window` or `min_possession_s` to 0 to switch a step off.

## 2. Inputs

One row per frame, any order. Rows are sorted internally and the original order is restored at the end. Only the columns of the components you list are required.

| Column | Needed by | Meaning |
|---|---|---|
| `period_id`, `video_frame` | always | integer frame number, unique within a period. Gaps are allowed. |
| `ball_x`, `ball_y` | geometry, clear_geometry, dead_ball | ball position, normalised 0-1 over the pitch. NaN = not tracked. |
| `min_team_a_dist`, `min_team_b_dist` | geometry, clear_geometry | distance (m) from each team's nearest player to the ball |
| `{n}_s` | dead_ball, ball_lost, physical_dead | player speed (m/s), one column per slot. Every column named digits + `_s` is used. |
| `ball_x`, `ball_y`, `min_team_*_dist` | touches | ball position and nearest-player distances (touch detection) |
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

| `ball_lost` | `D` for ball-untracked runs of `min_s`..`max_s` seconds whose median player speed is at most `max_median_speed_mps`; `X` elsewhere |
| `touches` | H / A at the frame where a player of that team plays the ball (sudden ball-velocity change, a player within `max_player_dist_m`), `X` elsewhere |

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
- **`ball_lost` and tracking outages:** a ball that is untracked for more than `max_s` (a tracking outage) or less than `min_s` (an occlusion) is never D, and neither is a run cut by a sequence edge. D recall is therefore limited to stoppages of 5-20 s with a tracked-player speed at most 2.0 m/s.
- **`snap_to_touch` needs `touches`:** without it the step fails with a `KeyError` on `touch_team`. It only moves a boundary when a touch of the gaining team lies within `max_shift_s`.
- **Where `dead_ball` ranks matters:** it is a normal rule. Put it first (as above) and dead frames are always `D`; put it lower and an earlier rule can take the frame.

## 7. Tests

`tests/test_hybrid_pipeline.py` imports `stages.hybrid_possession`. It checks the configurable code against the previous single-function version on random data: 48 parameter combinations (shuffled rows, a frame gap, missing values, ball outside the pitch, `split`), so the default config reproduces the old labels exactly. It also covers the config switches (remove, reorder, conditions, smoothing steps, registering a new component and smoother), the input errors, and the later additions: `last_touch`, `ball_lost` (stoppage-sized runs, sequence edges, speed ceiling, invalid parameters), `slow_window_s`, `slow_top_k` and the rejection of unknown `physical_dead` parameters.

```bash
python -m pytest tests/test_hybrid_pipeline.py     # or: python tests/test_hybrid_pipeline.py -> ALL TESTS PASSED
```

## 8. Choices made and options rejected

Measured on the public slice (frames 11185-18684: a pipeline check, not tuned on) and on label-free whole-match statistics. A rule is accepted only if it fires on at most 1.0% of clearly-live frames (ball inside, one team's nearest player within 1.5 m, the other's beyond 3.0 m, for at least 0.5 s); among the accepted ones the one adding most dead seconds is preferred. The slice has 17 possessions and 4 dead stretches, so one spell is worth about 0.03 of spellF1 and small differences are noise.

| Option | Result | Decision |
|---|---|---|
| `touches` + `snap_to_touch` (3.0 s) | best single addition | kept |
| `ball_lost` 5-20 s, ceiling 2.0 m/s | recall_D 0.157 -> 0.404, precision_D 0.699, macro_f1 0.711; headline 0.606 vs 0.625 (snap only), one spell | kept |
| `ball_lost` without a ceiling | precision_D 0.60 | rejected (ceiling added) |
| `physical_dead` top-5, 5 s window, 1.0 m/s | live 0.67%, +154 s; scoreboard unchanged | optional |
| `physical_dead` on the median speed | live play is slow too (median speed p10 1.02 m/s); no acceptable cell | rejected |
| slower team's top-3/top-5 speed | AUC 0.85 vs 0.84; +70 s more at the same false-positive rate; team asymmetry itself has AUC 0.54 | not implemented |
| team centroid speed | fires on 16% of live frames at 0.5 m/s | rejected |
| `hmm_dead` / HMM's D | the HMM says D on 94% of live frames | rejected |
| `dead_lead_s` 1 s / 2 s | D precision -8 / -15 points | off |
| `last_touch` | possessions +2, worse latency and home share; with the others worse spell metrics | rejected, off |

Not covered: a 20 s stoppage (public frames 15889-16399) with the ball tracked, no set-piece tag and median player speed about 1.0 m/s; and the tail of a goal-kick stoppage (no cutoff for `goal_kick`). No rule was tuned to them.

## last_touch (optional component, rejected, off)

The ball in flight (nobody within `clear_distance_m`) is otherwise decided by `tags`, i.e. by the attacking phase.
`last_touch` gives it to the team that played the ball last, until the next touch of anybody, for at most `hold_s`
seconds, and never while the ball is outside the pitch or untracked. Parameters: those of `touches` plus `hold_s`. It is not part of the final config (section 8).

    "last_touch": {"window_frames": 3, "min_delta_mps": 3.0, "peak_s": 0.4, "max_player_dist_m": 2.0,
                   "team_margin_m": 0.5, "hold_s": 4.0}
    rules: dead_ball, physical_dead, clear_geometry, last_touch, tags, geometry, hmm
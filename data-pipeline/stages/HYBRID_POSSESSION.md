# Hybrid possession: reference

`hybrid_possession.py` turns four sources of evidence into one label per frame:

- the **HMM** label (`hmm_possession`),
- **geometry** (which team has the player nearest to the ball),
- **Smart Tags** (who is attacking or defending),
- **dead-ball evidence** from tracking (ball out of play, ball still, players slow, set-piece windows).

Output: a copy of the input with one extra column, `hybrid_possession`, holding `H`, `A`, `D` or `X`.

| Label | Meaning |
|---|---|
| `H` | home team (team A) has the ball |
| `A` | away team (team B) has the ball |
| `D` | dead ball: play is stopped |
| `X` | unknown: no source could decide |

The scorer treats the labels differently: `X` is carried forward from the previous label for the spell metric and counts as a miss for macro F1; `D` ends a possession.

## 1. Inputs

One row per frame, in any order. Rows are sorted internally and the original order is restored at the end.

| Column | Meaning | Comes from |
|---|---|---|
| `period_id`, `video_frame` | integer frame number, unique within a period. Gaps are allowed. | tracking |
| `ball_x`, `ball_y` | ball position, normalised 0-1 over the pitch. NaN = not tracked. | tracking |
| `min_team_a_dist`, `min_team_b_dist` | distance (m) from each team's nearest player to the ball | `extract_nearest_player_ball_distance` |
| `{n}_s` | player speed (m/s), one column per slot. Every column named digits + `_s` is used. | tracking |
| `sp_kind`, `sp_since` | set piece in progress (e.g. `corner kick`) and seconds since it started | Smart Tag features |
| `home_attacking`, `away_attacking`, `home_defending`, `away_defending` | 0/1 phase flags | Smart Tag phase flags |
| `hmm_possession` | the HMM label per frame (H / A / D / X) | `hmm_possession.py` |

The function raises a `ValueError` for missing columns, missing or non-integer frame numbers, duplicate `(period_id, video_frame)` pairs, negative distances or speeds, tag flags other than 0/1, HMM labels other than H/A/D/X, and invalid parameters.

## 2. Parameters

| Parameter | Type | Meaning | Effect of raising it |
|---|---|---|---|
| `fps` | int > 0 | frames per second | none (unit conversion) |
| `smoothing_half_window` | int >= 0 | frames each side for the majority votes (50 = 2 s). 0 turns smoothing off. | smoother labels, slower reaction to real changes |
| `geometry_first` | bool | False: tags beat geometry unless a player is "clear". True: geometry beats tags whenever it has distances. | (switch) |
| `clear_distance_m` | float > 0 | a player this close (m) to the ball makes geometry decisive | geometry decides more often, tags less |
| `still_distance_m` | float > 0 | ball moved less than this (m) within +-1 s means "still" | more "still" frames, more dead ball |
| `slow_speed_mps` | float > 0 | median player speed below this within +-1 s means "players slow" | more "slow" frames, more dead ball |
| `dead_min_duration_s` | float >= 0 | dead-ball stretches shorter than this are dropped | fewer, longer dead-ball stretches |
| `dead_merge_gap_s` | float >= 0 | gaps this short between dead-ball stretches are closed | dead-ball stretches join up |
| `min_possession_s` | float >= 0 | possessions shorter than this are merged into a neighbour | fewer, longer possessions |
| `min_tracked_players` | int | the per-frame median speed needs at least this many tracked players | stricter, more NaN medians |
| `pitch_m` | [L, W] | pitch length and width in metres | none (unit conversion) |
| `set_piece_cutoffs_s` | {kind: s} | dead ball is only allowed in the first `s` seconds of that kind of set piece | longer allowed window |
| `split_at_dead` | bool, optional (default False) | smooth each block between dead-ball stretches on its own | (switch, see section 5) |

Example values from an earlier run of this pipeline. The cutoffs were measured from tracking around tagged set pieces; the rest are judgement values:

```yaml
fps: 25
smoothing_half_window: 50
geometry_first: false
clear_distance_m: 3.0
still_distance_m: 0.5
slow_speed_mps: 0.7
dead_min_duration_s: 2.0
dead_merge_gap_s: 2.0
min_possession_s: 3.0
set_piece_cutoffs_s: {corner kick: 5.6, free kick: 5.2, throw-in: 5.8, kick off (after goal): 4.0}
```

## 3. The steps

`compute_steps(df, params)` runs the steps in order and keeps every intermediate result.

**Step 1 - `check_input`.** Verifies the required columns and the frame numbers. Finds the speed columns.

**Step 2 - `read_settings`.** Validates the parameters and returns them as a `Settings` object.

**Step 3 - `prepare_frames`.** Sorts by period and frame. Splits the rows into *sequences*: runs of consecutive frames inside one period. No vote, window or smoothing in later steps crosses a gap or a period change. Converts the numeric columns and checks them.

**Step 4 - `ball_visibility`.** `seen` means a ball position exists. `inside` means it also lies within the 0-1 pitch box.

**Step 5 - `dead_ball`.** Builds the dead-ball evidence.

| Signal | Definition |
|---|---|
| `outside` | the ball is seen but outside the pitch box for at least 1 s |
| `still` | at least 80% of the +-1 s window is tracked and the ball moved less than `still_distance_m` in both x and y (in metres) |
| `slow` | the rolling median over +-1 s of the per-frame median player speed is below `slow_speed_mps` (the per-frame median needs `min_tracked_players` speeds) |
| `event_window` | `sp_kind` is a key of `set_piece_cutoffs_s` and `0 <= sp_since <= cutoff` |

```
dead_raw = outside  OR  ( event_window AND ( still OR slow OR ball not seen ) )
```

Then a clean-up per sequence: gaps of at most `dead_merge_gap_s` between dead-ball stretches are closed, and stretches shorter than `dead_min_duration_s` are dropped. The result is `dead`.

**Step 6 - `geometry`.** On frames where the ball is inside the pitch and both distances exist, the team with the nearer player has the ball (ties go to H). This label is smoothed with a majority vote over +-`smoothing_half_window` frames; unknown frames abstain. A frame is **clear** when the nearest player is within `clear_distance_m`.

**Step 7 - `tags`.** Home has the ball when home is attacking or away is defending. Away has it when away is attacking or home is defending. `tag_known` is true only when exactly one of the two holds; if both or neither hold, the tags do not decide.

**Step 8 - `merge_sources`.** One label per frame. Each line overwrites the previous one, so the last matching rule wins:

| Order | Rule | Source name |
|---|---|---|
| 1 | the HMM label where it is H or A | `hmm` |
| 2 | the smoothed geometry label where there are distances | `geometry` |
| 3 | the tag label where `tag_known` and not clear (or, with `geometry_first`, where there are no distances) | `tags` |
| 4 | the geometry label where the frame is clear | `clear_geometry` |
| 5 | frames with `dead` are set aside (as `X`, or `D` with `split_at_dead`) | (dead) |

Frames that no rule labelled stay `X` with source `none`.

**Step 9 - `smooth_and_finish`.** Smooths the H/A labels again and merges short possessions:

1. Inside each block (the labels are cut at every `D` frame), every H/A frame takes the majority H/A label within +-`smoothing_half_window` known frames.
2. A possession shorter than `min_possession_s` is merged into its longer neighbour, repeating until none is left. The shortest one is merged first.
3. `D` is written on the `dead` frames and the original row order is restored.

## 4. Looking at what happened to a frame

```python
steps = compute_steps(df, params)
steps.table().head(300)     # one row per frame, sorted by period and frame
steps.source_share()        # share of frames decided by each source
steps.dead.sum()            # number of dead-ball frames
```

`steps.table()` columns: `1_hmm` (HMM label), `4_ball` (inside / outside / missing), `5_dead`, `6_geometry`, `6_clear`, `7_tags`, `8_merged`, `8_source` (which rule decided), `9_final`.

A typical question and where to look:

| Question | Where |
|---|---|
| Why is this frame H when the nearest player is away? | `8_source`: if it is `hmm` or `tags`, geometry was overruled. Check `6_clear`. |
| Why is there no D around this set piece? | `5_dead`, and the signals `steps.still`, `steps.slow`, `steps.event_window`. |
| Why did a short possession disappear? | compare `8_merged` with `9_final`; step 9 merged it. |
| Many X frames? | `8_source == "none"`: no HMM label, no distances, no tags. |

## 5. Behaviours worth knowing

- **HMM D and X are ignored.** Only the HMM's H and A labels are used. D comes from the dead-ball evidence only.
- **Possessions can merge across a stoppage (default).** Dead-ball frames are set to `X` before the final smoothing, but the smoothing only cuts its blocks at `D`. So by default it never sees a stoppage, and a short possession after a stoppage can be merged into the one before it. With `split_at_dead = True`, dead-ball frames are `D` during smoothing, each side of a stoppage is smoothed on its own, and nothing is merged across it. The switch affects only frames next to dead-ball stretches. In one constructed example with an 8 s minimum, a 6 s away possession after a stoppage was absorbed by default and kept with the switch on; with a 3 s minimum the two modes agreed.
- **Ties go to H** in the nearest-player comparison (`a <= b`).
- **Unknown frames abstain** in every vote.
- **Sequences:** gaps in `video_frame` and period changes split the data, and nothing is smoothed across them.

## 6. Tests

`test_hybrid_equivalence.py` checks the restructured code against the original single function. It runs 32 parameter combinations on random data (shuffled rows, a frame gap, missing values, ball outside the pitch), compares 14 kinds of invalid input for identical error messages, and exercises `steps.table()`, `source_share()` and `split_at_dead`.

Run it from the folder with `hybrid_possession.py`, `orig_hybrid.py` (a copy of the previous version) and a `stages/hmm_possession.py` that provides `_sequences`:

```bash
python test_hybrid_equivalence.py     # ends with: ALL TESTS PASSED
```

# HMM and Hybrid Possession Pipeline

This document explains the concept, rationale, and execution logic of
[`hmm_pipeline.py`](./hmm_pipeline.py). The pipeline fits a Hidden Markov Model
(HMM) to tracking and event features, then combines its possession estimates
with explicit geometry, smart-tag, and dead-ball rules. The HMM provides
temporally consistent estimates; the hybrid rules use interpretable evidence
to override or supplement those estimates.

## What the pipeline runs

`build_hmm_pipeline()` defines four Kedro nodes, in order:

1. **Correct cached density (optional).** `prepare_cached_density` copies the
   input table and, when `correct_cached_density` is true, masks cached player
   density values for a team on frames where that team's nearest-player
   distance is missing. This avoids treating an unobserved team as having zero
   nearby players. The configured default is false, preserving the existing
   cache.
2. **Fit and apply the HMM.** `add_hmm_possession` adds `hmm_possession`
   labels.
3. **Apply the hybrid rules.** `add_hybrid_possession` adds
   `hybrid_possession`, using geometry, smart tags, the HMM, and independent
   dead-ball evidence.
4. **Export evaluation columns.** `export_hybrid_evaluation` writes
   `video_frame` as `frame` and `hybrid_possession` as `label` to
   `output/hybrid_results.csv`.

The HMM pipeline uses the existing `output/without_hmm.csv` feature cache; it
does not run feature extraction itself. Run it from the repository root with:

```bash
python data-pipeline/main.py --pipeline hmm
```

The output contains the cached input frames, not necessarily every frame in the
video, so it is not by itself a complete submission. The pipeline also writes
HMM state, feature-importance, and convergence diagnostics under
`data-pipeline/output/`.

## HMM: three latent possession states

### Why use an HMM?

Possession is not directly observed in the tracking table. A nearest-player
measurement, for example, can change from frame to frame even while the same
team remains in possession. An HMM represents possession as a hidden state
(`H`, `A`, or `D`) and observed tracking/event measurements as emissions from
that state. Its transition probabilities favor persistence, so the sequence
context can stabilize ambiguous individual frames.

The HMM is unsupervised: it is fitted on the input frames and is not trained
against ground-truth possession labels. Its initial state anchors are
heuristics, not labeled examples. Consequently, its state labels are estimates,
not verified calls.

### Training and decoding logic

1. **Choose and validate features.** The default feature list is `FEATURES` in
   [`stages/hmm_possession.py`](./stages/hmm_possession.py). It includes
   per-team nearest-ball distances, possession advantage, player-density
   counts, ball movement, tag and set-piece fields, phase flags, and team
   positioning/shape features. Some candidate velocity and movement features
   are deliberately commented out and are not HMM inputs. Both teams'
   nearest-ball distances are required because they define the team anchors.
2. **Handle missing observations.** By default, training uses each frame with
   at least one observed selected feature and fills missing values with the
   training mean after transformation and standardization. At decode time,
   missing feature dimensions are marginalized from the Gaussian emission
   likelihood. If every selected feature is missing, the output abstains with
   `X`; a missing ball alone does not imply a stoppage.
3. **Transform and standardize.** Each observed value is transformed with
   signed `log1p`, which compresses large magnitudes while retaining the sign,
   then standardized using training-frame means and standard deviations.
4. **Initialize weak anchors.** Define distance advantage as
   `min_team_b_dist - min_team_a_dist`: positive values mean Team A
   (Barcelona/home) is closer, while negative values mean Team B (Atletic
   Club/away) is closer. Frames in the top 30% of this advantage outside
   set-piece windows seed the H anchor; the bottom 30% seed A. Frames with a
   set-piece kind and usable training features seed D. Optional settings can
   further restrict D anchors to small ball displacement and/or an early
   `sp_since` window. The defaults do not apply those extra filters. Each
   anchor must meet the configured minimum sample count or fitting fails.
5. **Fit a diagonal-Gaussian HMM.** A three-state model is initialized from
   the anchor medians. Its default self-transition probability is `0.997`,
   expressing a preference for a state to persist; EM may update the
   transition matrix and emission parameters. The configured iteration limit,
   tolerance, and covariance floor are in `HMM_PARAMS` in
   [`main.py`](./main.py). Training runs break at missing-feature frames and
   at frame gaps or period boundaries.
6. **Map learned states to H/A/D.** EM can reorder or move the states. By
   default, the implementation finds the state-to-anchor assignment with the
   lowest total squared distance between learned emission means and the
   initial anchor means. This mapping can still be wrong if the learned
   clusters do not correspond to the weak anchors.
7. **Decode and optionally smooth.** Viterbi decoding uses both emissions and
   transition history, and restarts at skipped frame IDs and period boundaries.
   Frames with some observations receive the decoded H/A/D label; fully
   unobserved frames receive X. A centered majority-vote smoothing window is
   applied by default (50 frames on either side in the configured parameters).
   It does not cross periods, frame gaps, or X runs, and ties keep the original
   label. Because the window is centered, it uses future frames and is not
   real-time.

The HMM's D state is an estimated latent state. The hybrid stage does not use
that D value as proof of a dead ball; it detects D separately from its
configured dead-ball evidence.

## Hybrid rule-based possession

The hybrid combines several evidence sources with explicit precedence. Its
configured values are in `HYBRID_PARAMS` in
[`main.py`](./main.py). The current configuration prefers geometry when
available (`geometry_first: True`), uses a 25 fps frame rate, and sets
`smoothing_half_window` to 50.

### 1. Derive and smooth geometry possession

Geometry is available only when the ball is within normalized pitch bounds
(both coordinates between 0 and 1) and both teams' nearest-ball distances are
observed. The closer team supplies the label: Team A is H (ties go to H), and
Team B is A. Known geometry labels are smoothed with a centered majority vote
within continuous frame sequences; unknown frames abstain from the vote.

### 2. Derive smart-tag possession

Tags provide a team-specific signal:

- Home possession evidence is `home_attacking` **or** `away_defending`.
- Away possession evidence is `away_attacking` **or** `home_defending`.
- A tag decides only when exactly one side has evidence. If both sides or
  neither side have evidence, tags abstain.

### 3. Combine HMM, geometry, and tags

The starting label is the HMM's H or A label; HMM D and X are not used as
ownership labels. Available geometry then replaces that fallback. Tag
precedence depends on `geometry_first`:

- With the current `geometry_first: True` setting, an exclusive tag is used
  when geometry is unavailable. When both distances and in-bounds ball
  coordinates are available, geometry remains authoritative even if both
  teams are farther away than `clear_distance_m`.
- If `geometry_first` is false, an exclusive tag replaces the current label
  unless the nearest team is within `clear_distance_m`. Clear geometry is
  applied last and takes precedence over tags.

If none of these sources provides H or A, the label is X. The configured
`clear_distance_m` is 3 m, but it does not gate geometry under the current
geometry-first setting.

### 4. Detect dead-ball evidence independently

The hybrid uses two dead-ball signals:

- **Ball out of bounds:** the ball must be observed outside the normalized
  pitch for at least one second (`fps` consecutive frames).
- **Set-piece evidence:** the frame must be inside an early configured window
  for its `sp_kind`, and at least one of these conditions must hold:
  the ball is still, tracked players are slow, or the ball is not observed.

For stillness, a centered two-second window (at the configured frame rate) is
converted to pitch metres. At least 80% of the window's ball positions must be
observed, and the x and y ranges must each be below `still_distance_m`.
Slowness uses the centered rolling median of per-frame median player speeds;
each frame needs at least `min_tracked_players` observed speeds. The early
set-piece windows are configured per event kind in `set_piece_cutoffs_s`.
Missing ball tracking on its own, outside such a set-piece window, is not
treated as a stoppage.

Dead-ball runs are processed within continuous frame sequences: gaps no longer
than `dead_merge_gap_s` are closed, then runs shorter than
`dead_min_duration_s` are discarded. The configured values are two seconds
for each. Remaining dead frames are temporarily excluded from ownership
smoothing and restored as D afterward.

### 5. Smooth and merge possession spells

Within each non-dead segment, the hybrid applies a centered majority vote to
the known H/A labels. X frames abstain rather than splitting the vote. It then
merges H/A runs shorter than `min_possession_s` (configured as three seconds)
into an adjacent label, preferring the longer neighboring run when both sides
are available. It does not change D or X labels or cross dead-ball, period, or
missing-frame boundaries. Dead labels are restored as D after this smoothing.

## Interpretation and limitations

Both stages are heuristic. Set-piece windows can include live play, distance
to the ball is not proof of control, and tracking gaps can make geometry
unavailable. The centered smoothing windows use future observations, so hybrid
predictions are also not real-time. HMM and hybrid results should be checked
against independent matches or held-out clips; the diagnostics and the small
public ground-truth window are not independent validation.

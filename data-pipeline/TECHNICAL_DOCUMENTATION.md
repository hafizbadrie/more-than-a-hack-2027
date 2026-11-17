# Technical Documentation – Ball Possession Detection

## 1. Approach

### Objective and design

The solution produces a possession label for each frame by combining spatial evidence, ball movement, team behaviour and tactical context. It uses a two-stage pipeline: feature engineering followed by possession inference.

A nearest-player-only approach was not sufficient because proximity does not prove control, especially during loose balls, passes and challenges. An HMM-only approach was also insufficient: temporal consistency does not guarantee that its states correspond to actual possession, and an unsupervised model can be difficult to interpret. We therefore combined an unsupervised HMM with explicit geometric, tactical and dead-ball rules.

### Stage 1 – Feature engineering

The first stage turns tracking and Smart Tagging information into frame-level signals.

- **Ball-focused features:** nearest player-to-ball distances, possession-distance advantage, player density around the ball, ball speed, acceleration, direction changes, nearest-player/ball relative velocity, and ball trajectory relative to the attacking goals.
- **Team-focused features:** team centroids and dispersion, centroid velocity, average radial movement, width and depth, movement coherence, opponent proximity/closing speed, and team position relative to the ball.
- **Smart Tagging features:** attacking and defending phase indicators, attacking transitions, active-tag counts, players-in-the-box information and set-piece context.

These features provide different views of the same situation: individual proximity, ball behaviour, tactical phase and collective team movement. The processed feature table is the hand-off between feature engineering and possession inference.

### Stage 2 – Possession inference

The second stage has two complementary parts:

1. **Unsupervised temporal model:** A three-state, diagonal-Gaussian Hidden Markov Model (HMM) estimates home possession (`H`), away possession (`A`) and a latent dead-ball state (`D`) from selected features. Weak spatial and set-piece anchors initialise the states; they are not ground-truth possession labels.
2. **Hybrid decision layer:** A configurable rule system combines HMM predictions with geometric possession, Smart Tagging and three dead-ball detectors (set-piece/outside-pitch, team-centroid motion and ball-lost). It selects one label per frame using an explicit priority order, applies temporal smoothing, and finally aligns possession changes with qualifying touches where available.

The HMM captures relationships across multiple signals and neighbouring frames. The hybrid layer makes the final decisions interpretable and allows strong, explicit evidence—such as a detected stoppage or a player very close to the ball—to take precedence over a statistical prediction.

The result is one of four labels: `H` (home), `A` (away), `D` (dead ball) or `X` (unknown/abstain).

## 2. Assets used and how

The final pipeline directly consumes two prepared data sources:

| Asset             | Use in the implementation                                                                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tactical tracking | Primary source for ball and player positions, frame timing and player movement. Used to derive spatial, ball-motion and team-shape features.                                    |
| Smart Tagging     | Joined to tracking frames to provide attacking/defending phase flags, attacking transitions and set-piece context. Used as model features and as one hybrid decision component. |

The executable pipeline does not consume a homography file or video stream: it does not perform a camera-to-pitch transformation or video-analysis step. Possession inference operates on the prepared tracking coordinates.

Smart Tagging intervals are contextual signals, not ground-truth possession labels. A set-piece tag identifies an event window; it does not, by itself, establish that play is stopped. No private ground-truth labels are used to fit the HMM or generate predictions.

The pipeline derives all features and hybrid inputs from these organiser-provided sources.

## 3. Possession rules

### 3.1 HMM

The HMM uses selected ball, player, team-shape and Smart Tagging features. Signed values are transformed with signed `log1p` and standardised. Three weak initial anchors are used:

- `H`: frames in the upper 30% of home-versus-away nearest-player distance advantage, outside set-piece windows.
- `A`: frames in the lower 30%, outside set-piece windows.
- `D`: frames inside a set-piece window with usable model observations.

The model then estimates its parameters using Expectation-Maximisation and decodes the sequence using Viterbi. Learned states are mapped back to `H`, `A` and `D` using their distance to the initial anchor means. Missing feature dimensions are marginalised during decoding. A frame with no observed HMM feature is labelled `X`.

The HMM's `D` and `X` predictions are not used by the hybrid selector; only its `H` and `A` predictions are eligible as fallback decisions. Dead-ball labels come from the dedicated detectors described below.

### 3.2 Geometric possession

For each frame, the nearest tracked player from each team is compared using the minimum team-to-ball distances. The closer team receives the geometric label; ties go to home (`H`). Geometry only has an opinion when the ball is inside the pitch and both team distances are available.

`Clear geometry` uses the same nearest-team comparison but only decides when the nearest player is within 3 metres of the ball. General geometry can still decide when both teams are farther away. This distinction makes close proximity a stronger signal without leaving all other valid frames undecided.

### 3.3 Smart Tagging

The tag component uses attacking and defending phase flags:

- Home attacking **or** away defending → `H`.
- Away attacking **or** home defending → `A`.
- Both conditions or neither condition → `X` (abstain).

Attacking transitions and other tag-derived features are also available to the HMM. Tags are not treated as proof of possession because a tactical phase can span multiple possession events or transitions.

### 3.4 Loose balls, challenges and restarts

There is no dedicated player-control, duel or challenge classifier. A loose ball or contested challenge is therefore not assigned a special state. The system relies on the combination of proximity, ball movement, player density, relative velocity, team behaviour, tactical context and temporal modelling. If all applicable components abstain, the hybrid output remains `X`.

Dead-ball detection is separate from the HMM and has three independent components. The current configuration is in `HYBRID_PARAMS` in [`main.py`](main.py).

**Set-piece detector (`dead_ball`).** It labels a frame `D` when either:

- the ball is observed outside the pitch for at least one second; or
- a recognised set-piece event is within its configured time window and at least one supporting condition holds:
  - the ball is sufficiently still;
  - the rolling median player speed is below the configured threshold; or
  - the ball is not tracked.

The current configuration uses 0.5 m for ball stillness, 0.7 m/s for median player speed, at least five tracked players for the speed signal, and a minimum dead-ball duration of 2 seconds. Dead-ball intervals separated by gaps of up to 2 seconds are merged. Set-piece windows are configured by restart type: corner 5.6 s, free kick 5.2 s, throw-in 5.8 s and kick-off after goal 4.0 s. These are heuristics; a restart tag alone does not establish that play is dead.

**Team-centroid detector (`centroid_dead`).** This detector can identify an untagged stoppage even when the ball is tracked and moving. At each frame, it takes the greater of the two team centroid speeds if both speeds are available and each team has at least five tracked players. Within each consecutive-frame sequence and period, it calculates a centred median over ±5 seconds (251 frames at 25 fps), requiring at least 126 valid observations. A median strictly below 0.6 m/s is a candidate dead frame. Gaps of up to 2 seconds between candidates are filled, then runs shorter than 5 seconds are discarded. The detector outputs `D` for the remaining frames and `X` elsewhere.

The tracking requirement applies to observations used in the rolling median, not necessarily every output frame; gap filling can also cover frames without sufficient tracking. The centred window uses future frames and is not suitable for real-time use. It can miss stoppage boundaries or mark slow live play as dead. The threshold was explored against the public ground-truth window and requires independent validation.

**Ball-lost detector (`ball_lost`).** Most stoppages last 5-20 s, longer than the set-piece windows above, and the ball is often not tracked during them. On this match the ball is untracked for 43% of the frames, in three kinds of runs: short occlusions (under 2 s; none last 2-5 s), stoppage-sized runs (5-20 s; 93 runs, about 13% of the graded frames) and tracking outages (20 s up to several minutes; 74% of all untracked frames). A run of untracked ball is labelled `D` when:

- its length is between 5 s and 20 s;
- it does not touch the start or end of a gap-free stretch of frames (its true length would be unknown);
- the median player speed inside the run is at most 2.0 m/s (players walk when play is stopped; without this ceiling live play with a lost ball was labelled dead too often).

The bounds come from these whole-match, label-free statistics (the run-length distribution and the median player speed inside the 5-20 s runs: median 1.24 m/s, 90th percentile 2.52 m/s). This detector does not label outages longer than 20 s as `D`; another detector may still do so.

**Not used.** A sustained-low-speed signal for dead ball anywhere in the match (`physical_dead`) is implemented and listed in the configuration but has no active rule: with the median player speed no threshold stays below 1% false positives on clearly live frames (live play is also slow), and with the mean of the five fastest players (below 1.0 m/s for 5 s) it fires on 0.67% of such frames, adds 154 s of dead ball and leaves the scores on the public slice unchanged. The HMM's `D` state is also not selected (it fires on 94% of clearly live frames), and lengthening dead stretches backwards in time was not used. Earlier tests of team-motion signals did not use the current `centroid_dead` rule.

### 3.5 Hybrid selection and smoothing

The hybrid layer uses ordered, first-match-wins selection:

1. Dead-ball detection (set pieces)
2. Team-centroid dead-ball detection
3. Ball-lost detection
4. Clear geometry
5. General geometry
6. Smart Tagging
7. HMM

This is a priority system, not a weighted vote or confidence average. A component wins as soon as it provides a non-`X` opinion. Thus `centroid_dead` can assign `D` before geometry, tags or the HMM assign possession. If no component decides, the frame remains `X`.

After selection, the pipeline applies three smoothing steps:

1. A centred majority vote over a ±50-frame window. `X` abstains; ties retain the original frame label.
2. Merging of possession spells shorter than 3 seconds into a neighbouring spell.
3. **Snap to touch.** The nearest-player labels change team when the new team's player gets nearer to the ball than the old team's, which is usually before that player plays it. A touch is a sudden change of the ball velocity (at least 3.0 m/s within ±3 frames, and the maximum within ±0.4 s) with a player of either team within 2.0 m; it belongs to a team when that team's nearest player is at least 0.5 m closer than the other's. Each change of possession is moved to the first touch of the team that gains the ball within ±3 s; if there is no such touch it stays where it is. A change never moves past its neighbours.

The current `dead_handling="gap"` setting temporarily treats dead-ball frames as unknown during smoothing and restores `D` afterwards. This can allow a short possession immediately after a stoppage to merge with the possession before it. Processing windows and smoothing do not cross period boundaries or gaps in frame numbers.

### 3.6 Options evaluated and not used

Earlier options were assessed using a label-independent rule: the signal could fire on at most 1.0% of clearly live frames (ball inside the pitch, one team's nearest player within 1.5 m and the other's beyond 3.0 m, for at least 0.5 s), with preference for more dead-ball time among accepted settings. The current `centroid_dead` threshold was instead explored using the public ground-truth slice (frames 11185-18684), so it has not been independently validated. The slice contains only 17 possessions and 4 dead-ball stretches; one spell can materially change the headline score.

| Option | Outcome |
|---|---|
| Snap to touch | kept; the best single addition |
| Ball-lost detector (5-20 s, ceiling 2.0 m/s) | kept; dead-ball recall on the slice 0.16 -> 0.40, precision 0.70 |
| Team-centroid detector (`centroid_dead`) | enabled; public-slice dead-ball recall 0.404 -> 0.779, precision 0.699 -> 0.759 |
| Last-touch holding of the ball in flight (`last_touch`) | rejected; it changed the possession count and worsened latency and home share |
| Sustained slow-speed dead ball anywhere (`physical_dead`) | not used; neutral on the slice |
| HMM `D` state and earlier start of dead stretches | rejected, see 3.4 and below |

Disclosure: the speed ceiling of the ball-lost detector was added after a first check on the public slice showed that without it dead-ball precision was 0.60. The ceiling value itself (2.0 m/s) lies near the upper end of the speed range of the 5-20 s runs measured on the whole match. Starting dead stretches 1 s or 2 s earlier cost 8 and 15 points of dead-ball precision on the slice and was left off.

In the public-slice comparison, enabling `centroid_dead` increased macro F1 from 0.711 to 0.815 and mean spell F1 from 0.533 to 0.558. It added 108 false `D` labels on live frames and approximately 25,397 new `D` labels outside the graded public window, where their accuracy is unknown. These results are exploratory, not an independent estimate of generalisation.

## 4. Limitations and Future Work

### Data completeness

The organiser-provided tactical tracking data contains a row for every submission frame, so the final pipeline produces one possession label for each of the 199,319 frames. The limitation is incomplete tracking observations within some of those frames: the ball or one or more players may be untracked.

The current implementation does not interpolate or otherwise reconstruct missing ball and player observations before feature extraction. It uses the available signals, which can leave features unavailable or cause the hybrid decision layer to abstain when the evidence is insufficient.

An upstream tracking-reconstruction step could estimate missing ball and player positions before feature engineering. The reconstructed observations would then feed the existing feature and possession stages, potentially increasing usable evidence and improving possession accuracy while retaining complete submission-frame coverage.

### Dead-ball coverage

Dead-ball coverage remains imperfect. For the untagged, ball-tracked stoppage at frames 15889-16399, the centroid detector marks 390 of 511 ground-truth dead frames; it starts late and extends into live play. Goal kicks have no configured set-piece window, although other detectors can identify parts of a stoppage. The centroid rule may also flag slow live play, and its performance outside the public slice is unknown. The ball-lost detector alone does not mark outages longer than 20 s as dead.

### Future improvements

Further development would focus on:

- **Reconstructing tracking observations:** Evaluate interpolation or model-based reconstruction of missing ball and player positions upstream of feature engineering, then validate that reconstructed inputs improve possession decisions without creating false certainty.
- **Improving possession detection:** Introduce more explicit ball-control and player-challenge detection to distinguish controlled possession from loose balls and contested situations.
- **Refining decision rules:** Further evaluate geometric thresholds, dead-ball detection and component priorities against independently reviewed match situations; add a forced dead-ball rule for the period start before kick-off.
- **Broader validation:** Test the approach on additional matches to understand how well the methodology generalises to different teams and playing styles.

These improvements would help make the pipeline more robust, increase prediction coverage and improve the reliability of frame-level possession classification.

## 5. Reproduction

### 5.1 Environment setup

Run commands from the repository root. Install the pipeline dependencies:

```bash
pip install -r data-pipeline/requirements.txt
```

The current requirements file specifies minimum package versions rather than a fully pinned environment. For strict reproducibility, package versions should be locked before the final release.

### 5.2 Input files

The pipeline uses the tactical assets provided by the challenge organiser. Unpack the data bundle
under `data/`; no pre-prepared CSV input files are required:

```text
data/
└── tactical/
    ├── BAR-ATH_tactical_atd.xml
    ├── BAR-ATH_tactical_atd.txt
    └── BAR-ATH_tactical_smart_tagging.xml
```

The `data_preparation` stage converts these source files into CSVs used internally by the
subsequent pipeline stages.

### 5.3 Execute the pipeline

Run all three stages sequentially:

```bash
python data-pipeline/main.py
```

Run each stage independently, in order:

```bash
# Stage 1: convert the organiser-provided source files to CSV
python data-pipeline/main.py --pipeline data_preparation

# Stage 2: clean tracking and generate features
python data-pipeline/main.py --pipeline features

# Stage 3: generate the submission-format hybrid results using the existing feature cache
python data-pipeline/main.py  --pipeline hybrid
```

The `hmm` stage reads the existing processed feature file. Run `features` first if that file needs
to be created or refreshed; run `data_preparation` first if its input CSVs need to be created or
refreshed from the organiser-provided source files.

Running `python data-pipeline/main.py  --pipeline hybrid` generates
`data-pipeline/output/hybrid_results.csv` with the submission-ready `frame,label` format and all
199,319 frames in order.

### 5.4 Outputs

The pipeline writes outputs under `data-pipeline/output/`:

| File                         | Contents                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `tactical_processed.csv`     | Tracking rows enriched with engineered features                               |
| `hybrid_results.csv`         | Submission-format frame-level hybrid possession labels (`frame,label`)         |
| `hmm_feature_importance.csv` | HMM feature-separation diagnostics                                            |
| `hmm_convergence.csv`        | HMM convergence history                                                       |
| `hmm_states.csv`             | Learned-state mapping and transition diagnostics                              |

### 5.5 Copy and validate the submission file

Copy the generated file to the evaluation results folder:

```bash
cp data-pipeline/output/hybrid_results.csv evaluation/results/merah_putih.csv
```

Validate the copied file before submission:

```bash
python evaluation/validate_submission.py evaluation/results/merah_putih.csv
```

The validator checks the `frame,label` header, all 199,319 frames in order, allowed labels and
absence of blank cells. It checks the file format, not whether the predicted labels are accurate.

### 5.6 Tests

The centroid detector and its default configuration are covered by `tests/test_hybrid_possession.py` and `tests/test_pipeline.py`.

```bash
PYTHONPATH=data-pipeline/tests python -m unittest \
  test_hybrid_possession.CentroidDeadTests \
  test_pipeline.PipelineTests.test_centroid_dead_is_enabled_by_default
```

---

- **Project:** More Than A Hack 2027
- **Primary pipeline:** `data-pipeline/main.py`
- **Hybrid implementation:** `data-pipeline/stages/hybrid_possession.py`
- **HMM implementation:** `data-pipeline/stages/hmm_possession.py`

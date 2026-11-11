# Merah Putih: technical document (DRAFT)

More than a Hack 2027, BAR-ATH possession challenge. Home = Barcelona (H), away = Athletic (A).

## 1. Summary

A fully rule-based pipeline labels each of the 199,319 frames as H, A, D (dead ball) or X (unknown). It combines (a) tracking geometry (who is nearest to the ball, and clearly so), (b) Smart Tagging phases (attacking / defending tags per team) and (c) a tracking-only dead-ball detector. Two smoothing passes remove flicker. There is no training; every threshold is either a label-free rule or a stated judgement (section 4).

## 2. Assets

| Asset | Used |
|---|---|
| ATD tracking (positions, ball, speeds) | yes |
| Smart Tagging events | yes |
| Video / homography | no |
| Public ground truth (frames 11185-18684) | **pipeline check only**, never for tuning |
| Training data | none |

Documented rule applied: frames 110808-113741 are all D. Frames outside the graded ranges are not scored.

## 3. Method

1. **Geometry.** Per frame, the nearest tracked player to the ball decides the team if that player is within the clear distance (3.0 m) of the ball and clearly nearer than the closest opponent; a looser "far" fallback is used otherwise. The result is smoothed with a 50-frame window.
2. **Tags.** Per-frame flags home/away attacking and defending are derived from the Smart Tagging table (the Defending tags are redundant with the opponent's attacking tags: 0 frames differ, verified). A frame with exactly one team attacking gets that team; conflicts or none give no tag label. Tags and the pseudo-labels from geometry agree about 83% of the time, so tags are used to fill where geometry has no label.
3. **Dead ball (D).** Four tracking-only signals: ball missing, ball outside the pitch (run of at least 1 s), ball still (range under 0.5 m over +-1 s), and players slow (median tracked-player speed under 0.7 m/s over +-1 s). D is only allowed in windows around tagged set pieces (corner, free kick, throw-in, kick-off after a goal) and only before the estimated restart: cutoffs after the set-piece start of 5.6 s (corner), 5.2 s (free kick), 5.8 s (throw-in), 4.0 s (kick-off). Goal kicks have no estimate and use no cutoff.
4. **Unknowns.** Frames where no source gives a label are X (not carried forward by us). The scorer carries X forward for spellF1 and counts it as a miss for macroF1.
5. **Final smoothing.** A second pass (post-window 50 frames, minimum run 3 s) merges short possessions: 926 raw possessions (mean 5.9 s) become 295 (18.5 s).

## 4. Parameter provenance

| Parameter | Value | Origin |
|---|---|---|
| geometry window | 50 frames | judgement (starter docs suggest about 2 s) |
| clear distance | 3.0 m | judgement |
| post window / min run | 50 frames / 3 s | judgement |
| dead min / merge | 2 s / 2 s | judgement |
| players-slow threshold | 0.7 m/s | label-free rule: largest threshold with at most 5% false positives on frames pseudo-labelled live |
| set-piece cutoffs | see above | measured on tracking around 58 tagged set-piece windows (median restart signal time); small samples (free kick n=4, so the 25th percentile was used) |
| pseudo-label radius 1.5 m / gap 3 m / 0.5 s | | diagnostics only, not in the final labels |

The set-piece measurements include the public minutes' windows because only tracking and tags are used there, no labels.

## 5. Honest disclosure on the public slice

The public ground truth was used only to check that the pipeline runs and to compare variants after they were designed; it was not used for fitting any number. We did look at the public score tables while choosing between variants (for example, whether to add the second smoothing), so the ranking among our variants is not fully independent of that slice. Seventeen possessions make a tiny sample.

## 6. Results (public slice, pipeline check only)

| Variant | spellF1 (bootstrap 95% CI) | macroF1 |
|---|---|---|
| tag only | 0.095 | n/a |
| combined, no smoothing | 0.17 | n/a |
| + smoothing | 0.326 | n/a |
| **final (combined + dead ball + cutoffs)** | **0.386** [0.207, 0.552] | 0.618 [0.463, 0.701] |
| baseline, smoothing 125 | 0.424 [0.267, 0.594] | |

Final per class F1: H 0.828, A 0.756, D 0.269 (precision 0.96, recall 0.157). The confidence intervals overlap with the baselines: the variants are statistically indistinguishable on this slice.

Full-match output: graded frames H 41.1%, A 26.2%, D 21.0%, X 11.7%. Label sources: geometry 37.8%, geometry-far 6.5%, tags 35.7%, none 20.0%.

## 7. Limitations

- D recall is low (0.157 on the public slice) although precision is high: the dead-ball detector is conservative.
- Predicted spells are too long (24.2 s vs 14.2 s in the public GT; 12 predicted vs 17 true).
- 11.7% of graded frames are X.
- Cutoffs rest on small samples (58 windows, free kick n=4); goal kicks have no estimate.
- Tags and geometry disagree about 17% of the time; the tag rule for ownership is simple.
- `prepare_data.py` mirrors the data-prep notebook but could not be executed in our development sandbox (no data or kloppy there); we compare its output with the notebook's CSVs.

## 8. Reproduction

See `README.md`. `bash run_all.sh` regenerates `Merah Putih.csv` from the raw files with the parameters above; no randomness is involved.

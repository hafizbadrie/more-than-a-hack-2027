# Defining "controlled possession" events — discovery report (BAR-ATH, tactical ATD)

Method: join the tactical ATD (`BAR-ATH_tactical_atd.txt/.xml`) with the 5-minute public ground truth
(GT, video frames 11185–18684, 7,500 frames, 21 segments / 17 possessions), and read what the GT
calls "owning" and "dead" in terms of tracking signals. Slots 1–24 = Barcelona (`H`), 25–46 = Athletic (`A`);
`video_frame = frame_id − 1`; positions converted to metres (105.3 × 68).

Code: [tracking.py](tracking.py) (loader), [feat.py](feat.py) (per-frame features),
`explore1..6.py` (exploration), [events.py](events.py) (event extraction + samples → [samples/](samples/)).

## 0. Data findings you must know first

- **`atd.txt` is truncated.** It is 88,080,384 bytes (exactly 84 MiB) and ends mid-line at frame 88,863 of 199,319 —
  the whole second half and the last ~55% of the first are missing. The public-GT window (frames 11–18k) is intact,
  so this analysis is unaffected, but anything built on the file will have no tracking for frames ≥ 88,862 → re-download it.
  The loader keeps complete lines only.
- The ball's 3rd column is NaN inside the GT window (not usable); player speeds are m/s (median 1.4, p99 6.6).
- Ball visible on 99.9% (H) / 85% (A) / 66% (D) of GT frames. Ball gaps are **not** a dead-ball signal by themselves
  (two live gaps of 5.3 s and 9.3 s exist, both labelled `A`).

## 1. What the GT says (the facts that shape the definitions)

1. Possessions are **short and frequent**: 12 live→live hand-overs in 5 min; durations 0.76 s – 42 s (median ≈ 5.5 s).
   Two are < 1.5 s (0.76 s, 1.32 s). So "controlled" is **not** "keeps the ball for a few seconds".
2. A state machine "flip to the team whose nearest player is within R m of a slow ball after N consecutive frames"
   (grid over R∈{1.5,2,3,4}, V∈{3,6,10,∞} m/s, N∈{1,5,12,25,50} frames, margin∈{0,1,2} m; `explore6.py`)
   gives live-frame accuracy **~0.87–0.88 at N = 1** and *drops* (0.856–0.867) for longer N. Requiring a dwell
   time does not help; the GT flips at the first credible touch by the opposing team.
3. Raw proximity is a weak signal: at 12 GT hand-overs the new owner's nearest player was **within ±1 s** for only
   0–50 frames (3 of 12 had 0), and in 5 of 12 hand-overs the *old* owner was still nearer at the flip.
4. Opposing player within 2 m and nearer than the owner (35 episodes) → GT flips within 2 s only 27–60 % of the time,
   and the rate does **not** rise with episode duration (`samples/opposing_proximity_vs_flip.csv`). Those that don't
   flip are exactly the "touch without control / deflection" cases of CHALLENGE.md.
5. Dead ball (4 segments, 5.5–20.4 s): team-mean player speed falls to 1.1–1.5 m/s vs 2.2–2.6 live
   (fraction of players > 2.5 m/s: 0.01–0.08 vs 0.17–0.21), and the ball is either lost (2/4), out of the pitch
   (1/4), or tracked but wandering/slow (1/4, frames 15889–16399, ball still in-pitch). Ball-lost alone is ambiguous.

## 2. Proposed event definitions

Each event is detected from tracking only; thresholds are starting points from the GT window (n is tiny — validate
on the full match once more ground truth/ATD is available).

| # | Event | Detection rule (proposed) | Effect on label |
|---|---|---|---|
| E1 | **Restart / kick-off of a possession** (D→H/A) | Ball reappears/starts moving after a stoppage with team-mean player speed rising back above ~2 m/s; owner = team whose nearest player is closest to the ball (4/4 in GT, d≈2–4 m, other team 5–16 m) | `D`→owner |
| E2 | **Controlled reception / gain of control** | Opposing team's nearest player ≤ ~2–3 m, nearer than the owner by ≥ ~2 m, ball speed ≤ ~6 m/s — evaluate on the first frame it holds (N=1 beats dwell) | Flip owner |
| E3 | **Touch without control** (NOT a change) | Opposing player ≤ 2 m but ball speed stays high (> ~8–10 m/s, direction mostly preserved) or the contact is a few frames at passing speed — e.g. frames 17905, 18009 (ball 11–23 m/s) | Keep owner |
| E4 | **Loose ball** | Neither team within R, or margin < threshold | Keep last owner (CHALLENGE rule) |
| E5 | **Live ball gap** | Ball NaN ≥ 2 s but players still moving fast and ball last seen in pitch → hold last owner, flip only if the same team's nearest player is clearly the first to re-engage | Keep owner (or `X`) |
| E6 | **Dead-ball onset** (live→D) | Ball leaves the pitch (x<0, x>105.3, y<0, y>68) or is lost near a line, followed by a sustained drop in team-mean player speed (< ~1.6 m/s over 2 s) | `D` |
| E7 | **Dead-ball stoppage while ball is tracked** | Ball slow/wandering with player speeds low and no team engagement (frames 15889–16399) | `D` |

Caveats from the GT: dead-ball onset in 3/4 cases is the ball at 11–23 m/s heading to a line (last in-pitch position
31.6/61.5, 19.7/55.2, −0.8/24.4); the one lost-ball case has no last position at all.

## 3. Samples of records

Frame-level windows (−25 … +50 frames around each event; columns `frame, lab (GT), bx, by (m), bspeed (m/s),
dH, dA (nearest H/A player distance, m), near, n_players`) are in [samples/](samples/): `<event>_records.csv`
(sampled events) and `<event>_summary.csv` (all events found in the window).

### E1 Restart (all 4 in the window; `restart_summary.csv`)

| frame | dead_s | new owner | ball x,y | d_owner | d_other | ball speed |
|--:|--:|:-:|--:|--:|--:|--:|
| 12989 | 5.5 | H | 24.8, 54.8 | 4.1 | 8.5 | 9.6 |
| 16400 | 20.4 | A | 48.6, 65.2 | 2.3 | 4.8 | 5.3 |
| 17838 | 18.3 | A | 0.9, 62.7 | 2.1 | 6.3 | 5.4 |
| 18389 | 14.5 | H | 4.5, 41.4 | 3.6 | 16.1 | 4.8 |

Excerpt (frame 16400, A restart): the label flips as the ball speeds from 2.5 → 11 m/s with A's player 2.1–2.3 m away:

| rel | frame | lab | bx | by | bspeed | dH | dA |
|--:|--:|:-:|--:|--:|--:|--:|--:|
| −10 | 16390 | D | 48.3 | 66.2 | 2.5 | 5.9 | 2.1 |
| 0 | 16400 | A | 48.6 | 65.2 | 5.3 | 4.8 | 2.3 |
| 10 | 16410 | A | 51.5 | 63.7 | 10.6 | 6.2 | 2.7 |

### E2 Possession change (12 in the window; `possession_change_summary.csv`)

| frame | change | prev. dur (s) | d_new | d_old | ball speed | new nearest frames in ±1 s |
|--:|:-:|--:|--:|--:|--:|--:|
| 11351 | A→H | 6.6 | 3.4 | 2.2 | 4.7 | 0 |
| 11553 | H→A | 8.1 | 2.5 | 4.8 | 7.2 | 16 |
| 11624 | A→H | 2.8 | 1.1 | 4.0 | 0.8 | 50 |
| 12665 | H→A | 41.6 | 7.0 | 11.4 | 20.5 | 19 |
| 13704 | H→A | 28.6 | 1.8 | 3.6 | 11.1 | 35 |
| 13785 | A→H | 3.2 | 7.9 | 7.5 | 1.0 | 0 |
| 14846 | H→A | 42.4 | 1.1 | 3.8 | 8.5 | 13 |
| 15346 | A→H | 20.0 | 2.1 | 1.3 | 3.7 | 14 |
| 15365 | H→A | 0.8 | 2.9 | 1.5 | 8.5 | 18 |
| 15766 | A→H | 16.0 | 3.3 | 2.5 | 1.4 | 0 |
| 15799 | H→A | 1.3 | 1.2 | 1.7 | 6.3 | 38 |
| 17293 | A→H | 35.7 | 3.4 | 5.1 | 6.9 | 24 |

Clean example (11624, ball stationary 0.4–1 m/s, H player 0.4–1.1 m away, A 4–5 m): GT flips at the frame the H player
becomes nearest and stays so. Messy examples (11351, 13785, 15766): GT flips with the *old* team nearest — a
receiver/dead-spot effect; position-only rules lag or lead here.

### E3 Touches that did not change possession (6 sampled, `non_change_touch_summary.csv`)

| frame | owner | opposing contact (s) | min dist (m) | mean ball speed (m/s) |
|--:|:-:|--:|--:|--:|
| 11405 | H | 3.04 | 0.68 | 3.6 |
| 11708 | H | 0.72 | 0.41 | 3.3 |
| 12432 | H | 1.04 | 0.93 | 4.8 |
| 15365 | A | 0.08 | 1.52 | 8.8 |
| 17905 | A | 0.16 | 1.54 | 11.2 |
| 18009 | A | 0.16 | 0.52 | 23.0 |

Note the first three are slow, long, close contacts that still did *not* flip — proximity + slow ball is not sufficient;
the owning team's own nearby players (not measured here) or ball direction change are the next features to test.

### E5 Live ball gaps (2; `live_ball_gap_summary.csv`)
Frames 16646 (5.3 s, A nearest before and after → hold A) and 17047 (9.3 s, A before, H after, GT stays A through
the gap and its flip to H only comes at 17293 → don't flip on first re-appearance of the ball).

### E6 Dead-ball onset (4; `dead_onset_summary.csv`)

| frame | prev owner | dead (s) | ball pre-speed | last ball x,y | in pitch before | team speed in dead (m/s) |
|--:|:-:|--:|--:|--:|:-:|--:|
| 12851 | A | 5.5 | — (lost 0.4 s before) | — | n/a | 1.54 |
| 15889 | A | 20.4 | 11.1 | 31.6, 61.5 | yes | 1.12 |
| 17380 | H | 18.3 | 12.3 | 19.7, 55.2 | yes | 1.17 |
| 18027 | A | 14.5 | 23.1 | −0.8, 24.4 | no (x < 0) | 2.10 |

## 4. Implications for the design

- Don't require a dwell time (>2–3 s or "2+ passes") to flip — the GT flips at first credible control; use
  *evidence of control* (close, nearer by a margin, ball slows or changes direction) instead of *duration*.
- Make the D decision separately from ownership, using player-speed collapse + ball out/lost; ~15–20 s stoppages
  dominate the D frames, so mislabelling a short ball gap as D (or vice-versa) is costly.
- Use `X` for the unresolved gaps (E5) when the same team is not clearly re-engaging; `spellF1` fills `X` with the last label.
- Limits: 12 hand-overs / 4 restarts / 4 stoppages is too few to fit thresholds; use these as hypotheses and
  tune on the full match with a self-consistency check (e.g. restart owner vs Smart Tagging `Team`).

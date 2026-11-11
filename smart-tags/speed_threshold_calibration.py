"""Choose the 'players are standing' speed threshold WITHOUT labels.

    python speed_threshold_calibration.py BAR-ATH_tactical_ds.csv events.csv [--cap 5]

The signal is: median speed of the tracked players (smoothed over 2 s) below a threshold.  For each threshold the table shows
  live_%      how often the signal fires on pseudo-live frames (ball seen, one team clearly on it).  This is the
              false-positive proxy: standing players are rare while a team is clearly on the ball.
  start_%     how often it fires in the first --start-s seconds of the corner / free-kick / throw-in / after-goal
              kick-off windows (where the dead-ball tables showed the stoppage), per type and overall
  graded_%    how often it fires on all graded frames
CHOICE RULE (fixed before looking at the numbers): the LARGEST threshold with live_% <= --cap (default 5).
No ground truth is read.  Needs the same helper scripts as dead_ball_diagnostic.py.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from tracking_io import read_tracking

from dead_ball_diagnostic import FPS, N_SLOTS, _ordered, player_speed_median, set_piece_windows
from pseudo_label_agreement import _graded, pseudo_labels
from tag_only_submission import N_FRAMES

GRID = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1)
KINDS = ("corner kick", "free kick", "throw-in", "kick off (after goal)")
SHORT = {"corner kick": "corner", "free kick": "free_kick", "throw-in": "throw_in", "kick off (after goal)": "kickoff_after_goal"}


def calibrate(med, live, windows, grid=GRID, start_s=5.0, cap=5.0):
    """med: median player speed per video frame; live: bool mask of pseudo-live frames. -> (table, chosen threshold or None)"""
    g = _graded()
    start = {k: np.zeros(N_FRAMES, bool) for k in KINDS}
    for kind, a, b in windows:
        if kind in start:
            start[kind][max(a, 0):min(a + int(round(start_s * FPS)), b, N_FRAMES)] = True
    rows = []
    finite = np.isfinite(med)
    for th in grid:
        slow = finite & (med < th)
        row = {"threshold_m_s": th, "live_%": round(100 * slow[live & g].mean(), 1)}
        allstart = np.zeros(N_FRAMES, bool)
        for k in KINDS:
            m = start[k] & g
            allstart |= m
            row["start_%s_%%" % SHORT[k]] = round(100 * slow[m].mean(), 1) if m.any() else np.nan
        row["start_all_%"] = round(100 * slow[allstart & g].mean(), 1)
        row["graded_%"] = round(100 * slow[g].mean(), 1)
        rows.append(row)
    tab = pd.DataFrame(rows)
    ok = tab[tab["live_%"] <= cap]
    return tab, (float(ok["threshold_m_s"].max()) if len(ok) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracking"); ap.add_argument("events")
    ap.add_argument("--cap", type=float, default=5.0); ap.add_argument("--start-s", type=float, default=5.0)
    a = ap.parse_args()
    header = pd.read_csv(a.tracking, nrows=0).columns
    need = ["frame_id", "ball_x", "ball_y"] + ["%d_%s" % (n, c) for n in range(1, N_SLOTS + 1)
                                               for c in ("ball_dist_m", "team", "x_m", "y_m")]
    absent = [c for c in need if c not in header and c != "frame_id"]       # frame_id / video_frame: see tracking_io
    if absent:
        sys.exit("tracking csv lacks columns: %s" % absent[:6])
    table = read_tracking(a.tracking, need)
    pseudo, _ = pseudo_labels(table)
    med = player_speed_median(_ordered(table))
    tab, chosen = calibrate(med, pseudo != "", set_piece_windows(pd.read_csv(a.events, dtype=str, keep_default_na=False)),
                            start_s=a.start_s, cap=a.cap)
    pd.set_option("display.width", 220)
    print("PLAYERS-STANDING SIGNAL BY SPEED THRESHOLD (graded frames; %% of frames where the signal fires)\n")
    print(tab.to_string(index=False))
    live_speed = med[(pseudo != "") & _graded() & np.isfinite(med)]
    if live_speed.size:
        print("\nlive-play percentiles of the median player speed (m/s): " +
              ", ".join("p%d=%.2f" % (q, v) for q, v in zip((5, 10, 25, 50), np.percentile(live_speed, (5, 10, 25, 50)))))
    else:
        print("\nno pseudo-live frames found, so there is nothing to calibrate against")
    print("\nchoice rule: largest threshold with live_%% <= %.1f  ->  %s" %
          (a.cap, ("%.1f m/s  (pass --slow-ms %.1f to combined_submission.py)" % (chosen, chosen)) if chosen else "none in the grid"))


if __name__ == "__main__":
    main()

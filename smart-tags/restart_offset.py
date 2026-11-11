"""Estimate, per set-piece type, how many seconds into a template window the restart happens.

    python restart_offset.py BAR-ATH_tactical_ds.csv events.csv

Label-free: tracking signals + Smart Tagging windows only; no ground truth of any kind is read.  The public
minutes (frames 11185-18684) are part of the match data and are INCLUDED; pass --exclude-public to drop the windows
that overlap them.

Two estimators, both "when does the stoppage end":
  standing_end   end of the first run (>= --min-run-s) of 'players barely moving' that starts within the first
                 --max-start-s of the window  (set pieces where players wait: corners, free kicks, kick-offs)
  outside_end    end of the first run of 'ball outside the pitch' that starts within the first --max-start-s
                 (throw-ins: the ball re-enters when the throw is taken)
Per type it prints how many windows gave an estimate and the median / quartiles of the offset in seconds.
Use the median (or a more cautious lower quartile) as the dead-ball cutoff for that type.
Needs the same helper scripts as dead_ball_diagnostic.py.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from tracking_io import read_tracking

from dead_ball_diagnostic import FPS, N_SLOTS, set_piece_windows, signals
from pseudo_label_agreement import _graded
from tag_only_submission import N_FRAMES

PUBLIC = (11185, 18684)                       # inclusive; the minutes published with the repository


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def first_run_end(mask, a, b, max_start_frames, min_len):
    """end (exclusive) of the first True run of length >= min_len that starts in [a, a + max_start_frames), else None."""
    starts, ends = _runs(mask[a:b])
    for s, e in zip(starts, ends):
        if s < max_start_frames and e - s >= min_len:
            return a + e
    return None


def estimate(sig, windows, max_start_s=10.0, min_run_s=2.0, exclude_public=False):
    g = _graded()
    rows = []
    for kind, a, b in windows:
        a, b = max(a, 0), min(b, N_FRAMES)
        if b <= a or not g[a:b].all():
            continue                                         # only windows fully inside graded play
        if exclude_public and a <= PUBLIC[1] and b > PUBLIC[0]:
            continue                                         # optional: drop windows that overlap the public minutes
        ms, ml = int(round(max_start_s * FPS)), int(round(min_run_s * FPS))
        se = first_run_end(sig["players_slow"].to_numpy(), a, b, ms, ml)
        oe = first_run_end(sig["ball_outside"].to_numpy(), a, b, int(round(15 * FPS)), int(round(0.5 * FPS)))
        rows.append({"type": kind, "length_s": (b - a) / FPS,
                     "standing_end_s": (se - a) / FPS if se is not None else np.nan,
                     "outside_end_s": (oe - a) / FPS if oe is not None else np.nan})
    return pd.DataFrame(rows)


def summarise(est):
    out = []
    for kind, d in est.groupby("type"):
        for col in ("standing_end_s", "outside_end_s"):
            v = d[col].dropna()
            out.append({"type": kind, "estimator": col.replace("_s", ""), "windows": len(d), "with_estimate": len(v),
                        "median_s": round(v.median(), 1) if len(v) else np.nan,
                        "q25_s": round(v.quantile(.25), 1) if len(v) else np.nan,
                        "q75_s": round(v.quantile(.75), 1) if len(v) else np.nan,
                        "min_s": round(v.min(), 1) if len(v) else np.nan, "max_s": round(v.max(), 1) if len(v) else np.nan})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracking"); ap.add_argument("events")
    ap.add_argument("--max-start-s", type=float, default=10.0); ap.add_argument("--min-run-s", type=float, default=2.0)
    ap.add_argument("--exclude-public", action="store_true", help="drop windows overlapping frames 11185-18684")
    a = ap.parse_args()
    header = pd.read_csv(a.tracking, nrows=0).columns
    need = ["frame_id", "ball_x", "ball_y"] + ["%d_%s" % (n, c) for n in range(1, N_SLOTS + 1) for c in ("x_m", "y_m")]
    absent = [c for c in need if c not in header and c != "frame_id"]       # frame_id / video_frame: see tracking_io
    if absent:
        sys.exit("tracking csv lacks columns: %s" % absent[:6])
    sig = signals(read_tracking(a.tracking, need))
    est = estimate(sig, set_piece_windows(pd.read_csv(a.events, dtype=str, keep_default_na=False)), a.max_start_s, a.min_run_s, a.exclude_public)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print("windows used (fully inside graded play%s): %d\n" % (", public minutes excluded" if a.exclude_public else "", len(est)))
    print(summarise(est).to_string(index=False))
    est.to_csv("restart_offsets.csv", index=False)
    print("\nper-window estimates written to restart_offsets.csv")


if __name__ == "__main__":
    main()

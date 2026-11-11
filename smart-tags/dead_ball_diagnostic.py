"""Which tracking signals separate dead ball from live play?  A diagnostic, not a rule yet.

    python dead_ball_diagnostic.py BAR-ATH_tactical_ds.csv events.csv

No ground truth is used.  "Live" = the pseudo-labelled H/A frames (ball seen, one team clearly on it).
"Dead-ish" = the Smart Tagging set-piece windows (35 s templates for corner / free kick / goal kick / throw-in,
5 s for kick-offs), which contain the stoppage but also play around it.  A good dead-ball signal is RARE in
live play and COMMON inside set-piece windows, and the second table shows WHERE in each window it happens.

Signals (all from tracking, thresholds are arguments):
  ball_missing    the ball is not tracked
  ball_outside    the ball is seen outside the pitch for a run of >= --out-s seconds
  ball_still      the ball is seen on >= 80% of the frames within +/- --still-s seconds and moved < --still-m metres
  players_slow    the median speed of the tracked players over +/- 1 s is below --slow-ms metres per second
  any_signal      ball_outside or ball_still or players_slow
Needs smart_tags.py, smart_tags_throw_ins.py, tag_only_submission.py and pseudo_label_agreement.py alongside.
Only graded frames are used.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from tracking_io import read_tracking

from pseudo_label_agreement import GRADED, N_SLOTS, _graded, _runs_at_least, pseudo_labels
from smart_tags import _code_key, _text
from smart_tags_throw_ins import ensure_frames
from tag_only_submission import N_FRAMES

FPS = 25
PITCH = (105.3, 68.0)
SIGNALS = ["ball_missing", "ball_outside", "ball_still", "players_slow", "any_signal"]


def _ordered(table, frame_offset=-1):
    """the table sorted by VIDEO frame; asserts that every frame 0..N_FRAMES-1 appears exactly once."""
    vf = table["frame_id"].to_numpy(np.int64) + frame_offset
    ok = (vf >= 0) & (vf < N_FRAMES)
    assert ok.all(), "frame_id outside 0..%d" % (N_FRAMES - 1)
    order = np.argsort(vf)
    t = table.iloc[order].reset_index(drop=True)
    assert (np.diff(vf[order]) == 1).all() and len(t) == N_FRAMES, "tracking rows must cover every video frame once"
    return t


def player_speed_median(t, step=5, smooth_s=2.0):
    """median speed (m/s) of the tracked players per frame, over +/- smooth_s/2 s.  t: table ordered by video frame."""
    px = t[["%d_x_m" % n for n in range(1, N_SLOTS + 1)]].to_numpy(float)
    py = t[["%d_y_m" % n for n in range(1, N_SLOTS + 1)]].to_numpy(float)
    v = np.full(px.shape, np.nan)
    v[step:] = np.hypot(px[step:] - px[:-step], py[step:] - py[:-step]) / (step / FPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # frames with no tracked player at all
        med = np.nanmedian(v, axis=1)
    return pd.Series(med).rolling(int(round(smooth_s * FPS)) + 1, center=True, min_periods=FPS).median().to_numpy()


def signals(table, out_s=1.0, still_s=1.0, still_m=0.5, slow_ms=0.7, frame_offset=-1):
    """-> DataFrame indexed by VIDEO frame (0..N_FRAMES-1) with boolean columns SIGNALS."""
    t = _ordered(table, frame_offset)
    bx, by = t["ball_x"].to_numpy(float), t["ball_y"].to_numpy(float)
    seen = np.isfinite(bx) & np.isfinite(by)
    inside = seen & (bx >= 0) & (bx <= 1) & (by >= 0) & (by <= 1)
    outside = _runs_at_least(seen & ~inside, max(int(round(out_s * FPS)), 1))

    w = int(round(still_s * FPS))
    xs, ys = pd.Series(np.where(seen, bx * PITCH[0], np.nan)), pd.Series(np.where(seen, by * PITCH[1], np.nan))
    k = 2 * w + 1
    cnt = pd.Series(seen.astype(float)).rolling(k, center=True, min_periods=1).sum().to_numpy()
    rng_x = (xs.rolling(k, center=True, min_periods=1).max() - xs.rolling(k, center=True, min_periods=1).min()).to_numpy()
    rng_y = (ys.rolling(k, center=True, min_periods=1).max() - ys.rolling(k, center=True, min_periods=1).min()).to_numpy()
    still = (cnt >= 0.8 * k) & (rng_x < still_m) & (rng_y < still_m)

    med = player_speed_median(t)
    slow = np.isfinite(med) & (med < slow_ms)

    df = pd.DataFrame({"ball_missing": ~seen, "ball_outside": outside, "ball_still": still, "players_slow": slow})
    df["any_signal"] = df["ball_outside"] | df["ball_still"] | df["players_slow"]
    return df


def set_piece_windows(events):
    """list of (kind, start, end) for the set-piece tags."""
    f = ensure_frames(events)
    out = []
    for code, typ, a, b in zip(f["code"], f["Type"] if "Type" in f.columns else [""] * len(f), f["start_frame"], f["end_frame"]):
        if _code_key(_text(code)) == "set_pieces":
            out.append((_text(typ).lower() or "(no type)", int(float(a)), int(float(b))))
    return out


def report(sig, pseudo, windows, bin_s=5):
    g = _graded()
    live = (pseudo != "") & g
    in_sp = np.zeros(N_FRAMES, bool)
    for _, a, b in windows:
        in_sp[max(a, 0):min(b, N_FRAMES)] = True
    in_sp &= g
    rows = []
    for name, m in (("pseudo-live (H/A)", live), ("inside set-piece windows", in_sp), ("all graded frames", g)):
        rows.append({"frames": name, "n": int(m.sum()), **{s: round(100 * sig.loc[m, s].mean(), 1) for s in SIGNALS}})
    tableA = pd.DataFrame(rows)
    rows = []
    for kind in sorted({k for k, _, _ in windows}):
        ws = [(a, b) for k, a, b in windows if k == kind]
        span = max(b - a for a, b in ws)
        for lo in range(0, span, bin_s * FPS):
            idx = np.concatenate([np.arange(a + lo, min(a + lo + bin_s * FPS, b)) for a, b in ws])
            idx = idx[(idx >= 0) & (idx < N_FRAMES)]
            idx = idx[g[idx]]
            if len(idx) == 0:
                continue
            rows.append({"set piece": kind, "seconds into window": "%d-%d" % (lo // FPS, lo // FPS + bin_s), "n_frames": len(idx),
                         **{s: round(100 * sig[s].to_numpy()[idx].mean(), 1) for s in SIGNALS}})
    return tableA, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracking"); ap.add_argument("events")
    ap.add_argument("--out-s", type=float, default=1.0); ap.add_argument("--still-s", type=float, default=1.0)
    ap.add_argument("--still-m", type=float, default=0.5); ap.add_argument("--slow-ms", type=float, default=0.7)
    a = ap.parse_args()
    header = pd.read_csv(a.tracking, nrows=0).columns
    need = ["frame_id", "ball_x", "ball_y"] + ["%d_%s" % (n, s) for n in range(1, N_SLOTS + 1)
                                               for s in ("ball_dist_m", "team", "x_m", "y_m")]
    absent = [c for c in need if c not in header and c != "frame_id"]       # frame_id / video_frame: see tracking_io
    if absent:
        sys.exit("tracking csv lacks columns: %s" % absent[:6])
    table = read_tracking(a.tracking, need)
    pseudo, _ = pseudo_labels(table)
    sig = signals(table, a.out_s, a.still_s, a.still_m, a.slow_ms)
    events = pd.read_csv(a.events, dtype=str, keep_default_na=False)
    tA, tB = report(sig, pseudo, set_piece_windows(events))
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("\n%% OF FRAMES WITH EACH SIGNAL  (a dead-ball signal should be low in live play, high in set-piece windows)\n" + tA.to_string(index=False))
    print("\n%% OF FRAMES WITH EACH SIGNAL, BY SET PIECE AND SECONDS INTO ITS WINDOW\n" + tB.to_string(index=False))


if __name__ == "__main__":
    main()

"""Step 2: geometry where the ball is seen, Smart Tags where it is not, X where neither knows.

    python combined_submission.py BAR-ATH_tactical_ds.csv events.csv -o combined.csv
    python ../evaluation/validate_submission.py combined.csv          # from the starter folder

Per video frame (home = Barcelona = H, away = Athletic = A):
  1. GEOMETRY.  On frames where the ball is visible and inside the pitch, the team of the nearest tracked
     player has the ball.  That raw label is smoothed with a centred majority vote over +/- --window frames
     (default 50 = 2 s, the starter baseline's best full-match setting), counting only frames where the ball is seen.
  2. COMBINE.  Ball seen and the nearest player within --clear-m (default 3 m): the smoothed geometry label.
     Ball seen but nobody close: the tag label if it says H or A, otherwise the smoothed geometry label.
     Ball not seen: the tag label if it says H or A, otherwise X.
  3. D only where documented: frames 110808-113741.  (Dead ball elsewhere is NOT detected yet.)
Tags = the tag-only rule with the Defending rows on (Team on Defending tags = the defending team, as the
pseudo-label check showed).  X is written as X; the scorer carries the previous label forward for spellF1.
No ground truth, no hand labels, no training.

Reads only frame_id, ball_x, ball_y, {n}_ball_dist_m and {n}_team of the big tracking CSV.
Needs smart_tags.py, smart_tags_throw_ins.py and tag_only_submission.py in the same folder.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from tracking_io import read_tracking

from smart_tags import _make_signer
from dead_ball_diagnostic import set_piece_windows, signals
from tag_only_submission import DEAD_BLOCK, N_FRAMES, summarise, tag_only_labels, write_submission

N_SLOTS = 46


def geometry(table, match_info=None, frame_offset=-1):
    """-> (side, nearest_m, seen), each of length N_FRAMES indexed by video frame.
    side is +1 home / -1 away / 0 unknown; nearest_m is the distance of the nearest tracked player to the ball."""
    sign_of = _make_signer({}, match_info or {"home_team": "Barcelona", "away_team": "Athletic"})
    slots = range(1, N_SLOTS + 1)
    dist = table[["%d_ball_dist_m" % n for n in slots]].to_numpy(float)
    teams = np.array([sign_of(str(table["%d_team" % n].mode().iloc[0])) for n in slots])
    if not ((teams > 0).any() and (teams < 0).any()):
        raise ValueError("could not find both teams in the {n}_team columns: %s" % sorted(set(teams)))
    dist = np.where(np.isnan(dist), np.inf, dist)
    dist[:, teams == 0] = np.inf                                     # unknown-team slots never decide
    nearest = dist.min(axis=1)
    who = teams[dist.argmin(axis=1)]
    bx, by = table["ball_x"].to_numpy(float), table["ball_y"].to_numpy(float)
    seen = np.isfinite(bx) & np.isfinite(by) & (bx >= 0) & (bx <= 1) & (by >= 0) & (by <= 1) & np.isfinite(nearest)
    side = np.where(seen, who, 0)
    vf = table["frame_id"].to_numpy(np.int64) + frame_offset
    ok = (vf >= 0) & (vf < N_FRAMES)
    out_side, out_near, out_seen = np.zeros(N_FRAMES, np.int8), np.full(N_FRAMES, np.inf), np.zeros(N_FRAMES, bool)
    out_side[vf[ok]], out_near[vf[ok]], out_seen[vf[ok]] = side[ok], nearest[ok], seen[ok]
    return out_side, out_near, out_seen


def smooth(side, window):
    """centred majority vote of the home / away labels within +/- window frames; ties keep the frame's own label."""
    if window <= 0:
        return side.copy()
    h = np.concatenate([[0], np.cumsum(side > 0)])
    a = np.concatenate([[0], np.cumsum(side < 0)])
    idx = np.arange(len(side))
    lo, hi = np.clip(idx - window, 0, len(side)), np.clip(idx + window + 1, 0, len(side))
    nh, na = h[hi] - h[lo], a[hi] - a[lo]
    out = np.where(nh > na, 1, np.where(na > nh, -1, side))
    return np.where(side != 0, out, 0).astype(np.int8)               # only defined where the ball is seen


def combine(side, nearest, seen, tag_label, window=50, clear_m=3.0):
    sm = smooth(side, window)
    geo = np.where(sm > 0, "H", np.where(sm < 0, "A", "X"))
    tag_ok = (tag_label == "H") | (tag_label == "A")
    clear = seen & (nearest <= clear_m)
    label = np.full(N_FRAMES, "X", dtype="<U1")
    label[~seen & tag_ok] = tag_label[~seen & tag_ok]                           # ball missing: tags
    label[seen & ~clear] = np.where(tag_ok, tag_label, geo)[seen & ~clear]       # ball seen, nobody close: tags, else geometry
    label[clear] = geo[clear]                                                    # a player is clearly on the ball: geometry
    label[DEAD_BLOCK[0]:DEAD_BLOCK[1] + 1] = "D"
    source = np.full(N_FRAMES, "none", dtype="<U8")
    source[~seen & tag_ok] = "tags"; source[seen & ~clear & tag_ok] = "tags"
    source[seen & ~clear & ~tag_ok] = "geo_far"; source[clear] = "geo"
    return label, source


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def parse_cutoffs(text):
    """'corner kick=8,free kick=12' -> {'corner kick': 8.0, 'free kick': 12.0}"""
    out = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        key, _, val = part.rpartition("=")
        if not key:
            raise ValueError("bad --sp-cutoff entry %r, expected 'type=seconds'" % part)
        out[key.strip().lower()] = float(val)
    return out


def dead_mask(sig, windows=(), use=("players_slow", "ball_outside", "ball_still"), min_s=2.0, merge_gap_s=2.0,
              sp_start_s=0.0, sp_cutoffs=None, sp_mode="gated", fps=25):
    """Frames judged dead ball from tracking signals (see dead_ball_diagnostic.py).

    1. OR of the chosen signals;  2. optionally, the first sp_start_s seconds of a corner / free-kick / after-goal
    kick-off window count as dead while the ball is untracked;  3. gaps shorter than merge_gap_s between dead
    stretches are closed;  4. dead stretches shorter than min_s are dropped.
    """
    m = np.zeros(N_FRAMES, bool)
    for name in use:
        m |= sig[name].to_numpy()
    if sp_start_s > 0:
        miss = sig["ball_missing"].to_numpy()
        for kind, a, b in windows:
            if kind in ("corner kick", "free kick", "kick off (after goal)"):
                lo, hi = max(a, 0), min(a + int(round(sp_start_s * fps)), b, N_FRAMES)
                m[lo:hi] |= miss[lo:hi]
    if sp_cutoffs:                                         # per-type: dead from the window start until the cutoff
        any_sig = m | sig["ball_missing"].to_numpy()
        for kind, a, b in windows:
            if kind in sp_cutoffs:
                lo, hi = max(a, 0), min(a + int(round(sp_cutoffs[kind] * fps)), b, N_FRAMES)
                if sp_mode == "always":
                    m[lo:hi] = True
                else:                                        # gated: only where a dead-ball signal (or a hidden ball) agrees
                    m[lo:hi] |= any_sig[lo:hi]
    gap = int(round(merge_gap_s * fps))
    s0, e0 = _runs(~m)                                     # close short gaps that lie BETWEEN dead stretches
    for a, b in zip(s0, e0):
        if 0 < a and b < N_FRAMES and b - a <= gap:
            m[a:b] = True
    keep = np.zeros(N_FRAMES, bool)
    for a, b in zip(*_runs(m)):
        if b - a >= int(round(min_s * fps)):
            keep[a:b] = True
    return keep


def smooth_labels(label, window=50, min_run_s=3.0, fps=25):
    """Second smoothing stage on the FINAL labels, so source switches stop creating short spells.

    Works block by block between D stretches.  Within a block:
      1. every H/A frame takes the majority H/A label within +/- window frames (X and D do not vote; ties keep the label)
      2. a possession as the scorer sees it (X carried forward) shorter than min_run_s is merged into its longer neighbour
    X frames stay X and D frames stay D.
    """
    out = label.copy()
    n = len(out)
    is_d = out == "D"
    starts = np.flatnonzero(np.diff(np.concatenate([[1], is_d.astype(np.int8)])) == -1)
    ends = np.flatnonzero(np.diff(np.concatenate([is_d.astype(np.int8), [1]])) == 1) + 1
    for lo, hi in zip(starts, ends):
        seg = out[lo:hi]
        idx = np.flatnonzero((seg == "H") | (seg == "A"))
        if len(idx) == 0:
            continue
        val = np.where(seg[idx] == "H", 1, -1)
        if window > 0:                                    # 1. majority vote among labelled frames within +/- window
            cs = np.concatenate([[0], np.cumsum(val)])
            left = np.searchsorted(idx, idx - window, side="left")
            right = np.searchsorted(idx, idx + window, side="right")
            tot = cs[right] - cs[left]
            val = np.where(tot > 0, 1, np.where(tot < 0, -1, val))
        # 2. merge short runs (a run lasts until the next differing label)
        while True:
            change = np.flatnonzero(np.diff(val) != 0) + 1
            starts_i = np.concatenate([[0], change])
            if len(starts_i) == 1:
                break
            pos = np.concatenate([idx[starts_i], [hi - lo]])
            dur = np.diff(pos) / fps
            k = int(np.argmin(dur))
            if dur[k] >= min_run_s:
                break
            lo_i, hi_i = starts_i[k], (starts_i[k + 1] if k + 1 < len(starts_i) else len(val))
            if k == 0:
                nb = val[hi_i]
            elif k == len(starts_i) - 1:
                nb = val[lo_i - 1]
            else:
                nb = val[lo_i - 1] if dur[k - 1] >= dur[k + 1] else val[hi_i]
            val[lo_i:hi_i] = nb
        seg[idx] = np.where(val > 0, "H", "A")
        out[lo:hi] = seg
    return out


def spell_stats(label, fps=25):
    """number and mean length (s) of possessions as the scorer sees them: X carried forward inside a block, D breaks."""
    nsp, tot = 0, 0
    is_d = label == "D"
    for lo, hi in zip(np.flatnonzero(np.diff(np.concatenate([[1], is_d.astype(np.int8)])) == -1),
                      np.flatnonzero(np.diff(np.concatenate([is_d.astype(np.int8), [1]])) == 1) + 1):
        seg = label[lo:hi]
        idx = np.flatnonzero((seg == "H") | (seg == "A"))
        if len(idx):
            nsp += int((np.diff(seg[idx] == "H") != 0).sum()) + 1
            tot += hi - lo - idx[0]
    return nsp, (tot / fps / max(nsp, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracking"); ap.add_argument("events"); ap.add_argument("-o", "--out", default="combined.csv")
    ap.add_argument("--window", type=int, default=50); ap.add_argument("--clear-m", type=float, default=3.0)
    ap.add_argument("--no-defending", action="store_true")
    ap.add_argument("--no-dead-ball", action="store_true", help="only the documented D block, no tracking-based D")
    ap.add_argument("--dead-min-s", type=float, default=2.0); ap.add_argument("--dead-merge-s", type=float, default=2.0)
    ap.add_argument("--sp-start-s", type=float, default=0.0,
                    help="also call D the first N s of corner / free kick / after-goal kick-off windows while the ball is untracked")
    ap.add_argument("--slow-ms", type=float, default=0.7,
                    help="'players standing' threshold: median tracked-player speed below this (m/s); see speed_threshold_calibration.py")
    ap.add_argument("--sp-cutoff", default="",
                    help="per-type dead-ball cutoffs in seconds from the window start, e.g. 'corner kick=8,free kick=12'")
    ap.add_argument("--sp-mode", choices=["gated", "always"], default="gated",
                    help="gated: dead only where a signal agrees; always: the whole stretch up to the cutoff")
    ap.add_argument("--post-window", type=int, default=50, help="majority vote on the final labels, +/- frames (0 = off)")
    ap.add_argument("--min-run-s", type=float, default=3.0, help="merge possessions shorter than this (0 = off)")
    a = ap.parse_args()
    header = pd.read_csv(a.tracking, nrows=0).columns
    need = ["frame_id", "ball_x", "ball_y"] + ["%d_%s" % (n, c) for n in range(1, N_SLOTS + 1)
                                               for c in ("ball_dist_m", "team", "x_m", "y_m")]
    absent = [c for c in need if c not in header and c != "frame_id"]       # frame_id / video_frame: see tracking_io
    if absent:
        sys.exit("tracking csv lacks columns: %s" % absent[:6])
    table = read_tracking(a.tracking, need)
    side, nearest, seen = geometry(table)
    tag_label = tag_only_labels(pd.read_csv(a.events, dtype=str, keep_default_na=False), use_defending=not a.no_defending)
    label, source = combine(side, nearest, seen, tag_label, a.window, a.clear_m)
    events = pd.read_csv(a.events, dtype=str, keep_default_na=False)
    if not a.no_dead_ball:
        dead = dead_mask(signals(table, slow_ms=a.slow_ms), set_piece_windows(events), min_s=a.dead_min_s, merge_gap_s=a.dead_merge_s,
                         sp_start_s=a.sp_start_s, sp_cutoffs=parse_cutoffs(a.sp_cutoff), sp_mode=a.sp_mode)
        label[dead] = "D"
    n0, m0 = spell_stats(label)
    if a.post_window > 0 or a.min_run_s > 0:
        label = smooth_labels(label, a.post_window, a.min_run_s)
    n1, m1 = spell_stats(label)
    print("possessions before / after the second smoothing: %d (mean %.1f s) -> %d (mean %.1f s)" % (n0, m0, n1, m1))
    write_submission(label, a.out)
    summarise(label)
    g = np.zeros(N_FRAMES, bool); g[11185:90652] = True; g[110808:190198] = True
    print("graded frames by source: " + "  ".join("%s %.1f%%" % (k, 100 * np.mean(source[g] == k)) for k in ("geo", "geo_far", "tags", "none")))
    print("wrote", a.out, "- now run: python ../evaluation/validate_submission.py", a.out)


if __name__ == "__main__":
    main()

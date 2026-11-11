"""How well does the tag rule agree with an algorithmic possession signal from the tracking?

    python pseudo_label_agreement.py full_tracking.csv events.csv [--radius 1.5 --gap 3.0 --min-s 0.5]

PSEUDO-LABEL (tracking only, no ground truth, no hand labels): a frame is H when the ball is visible and
inside the pitch, a home player is within --radius metres of it, every away player is further than --gap,
and that has held for at least --min-s seconds; A is the mirror.  Everything else gets no pseudo-label.
This is a noisy yardstick, not a label: it only exists where the ball is seen and one team is clearly on it.

Reads only frame_id, ball_x, ball_y and the {n}_ball_dist_m / {n}_team columns of the big CSV.
The slot -> team mapping comes from the {n}_team columns themselves.
Needs smart_tags.py, smart_tags_throw_ins.py and tag_only_submission.py in the same folder.
Only graded frames (11185-90651, 110808-190197) are used.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from tracking_io import read_tracking

from smart_tags import _make_signer, build_phase_flags
from smart_tags_throw_ins import ensure_frames
from tag_only_submission import N_FRAMES, tag_only_labels

GRADED = [(11185, 90651), (110808, 190197)]
N_SLOTS = 46


def _runs_at_least(mask, n):
    """keep only the True runs of `mask` that are at least n frames long."""
    if n <= 1:
        return mask
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    out = np.zeros_like(mask)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= n:
            out[s:e] = True
    return out


def pseudo_labels(table, radius=1.5, gap=3.0, min_s=0.5, fps=25, match_info=None, frame_offset=-1):
    """-> (array of 'H'/'A'/'' per VIDEO frame (length N_FRAMES), info dict)"""
    sign_of = _make_signer({}, match_info or {"home_team": "Barcelona", "away_team": "Athletic"})
    slots = list(range(1, N_SLOTS + 1))
    dist = table[["%d_ball_dist_m" % n for n in slots]].to_numpy(float)
    teams = []
    for n in slots:
        vals = table["%d_team" % n].astype(str)
        mode = vals.mode().iloc[0]
        if vals.nunique() > 1:
            print("note: slot %d changes team name during the match (using %r)" % (n, mode), file=sys.stderr)
        teams.append(sign_of(mode))
    teams = np.array(teams)
    if (teams == 0).any():
        print("note: %d slot(s) with an unrecognised team name: %s" %
              ((teams == 0).sum(), sorted({str(table["%d_team" % n].iloc[0]) for n in slots if teams[n - 1] == 0})), file=sys.stderr)
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, teams > 0].min(axis=1) if (teams > 0).any() else np.full(len(dist), np.inf)
    d_away = dist[:, teams < 0].min(axis=1) if (teams < 0).any() else np.full(len(dist), np.inf)
    bx, by = table["ball_x"].to_numpy(float), table["ball_y"].to_numpy(float)
    seen = np.isfinite(bx) & np.isfinite(by) & (bx >= 0) & (bx <= 1) & (by >= 0) & (by <= 1)
    h = seen & (d_home <= radius) & (d_away >= gap)
    a = seen & (d_away <= radius) & (d_home >= gap)
    k = max(int(round(min_s * fps)), 1)
    h, a = _runs_at_least(h, k), _runs_at_least(a, k)
    lab = np.where(h, "H", np.where(a, "A", ""))
    out = np.full(N_FRAMES, "", dtype="<U1")
    vf = table["frame_id"].to_numpy(np.int64) + frame_offset
    keep = (vf >= 0) & (vf < N_FRAMES)
    out[vf[keep]] = lab[keep]
    info = {"home_slots": int((teams > 0).sum()), "away_slots": int((teams < 0).sum()),
            "median_nearest_player_m_when_seen": float(np.median(np.minimum(d_home, d_away)[seen & np.isfinite(np.minimum(d_home, d_away))]))
            if (seen & np.isfinite(np.minimum(d_home, d_away))).any() else float("nan")}
    return out, info


def _graded():
    g = np.zeros(N_FRAMES, bool)
    for lo, hi in GRADED:
        g[lo:hi + 1] = True
    return g


def report(pseudo, events, match_info=None):
    mi = match_info or {"home_team": "Barcelona", "away_team": "Athletic"}
    tags = ensure_frames(events)
    frame = pd.DataFrame({"frame": np.arange(N_FRAMES)})
    fl = build_phase_flags(frame, tags, {}, mi)
    tr = build_phase_flags(frame, tags, {"attacking_codes": {"attacking_transition"}, "defending_codes": set()}, mi)
    g = _graded()
    has = (pseudo != "") & g
    ph, pa = (pseudo == "H"), (pseudo == "A")
    res = {}
    res["coverage"] = pd.DataFrame({"frames": [int(g.sum())], "pseudo_H": [int((ph & g).sum())], "pseudo_A": [int((pa & g).sum())],
                                    "pseudo_%_of_graded": [round(100 * has.sum() / g.sum(), 1)],
                                    "pseudo_home_share_%": [round(100 * (ph & g).sum() / max(has.sum(), 1), 1)]})
    hA, aA = fl["home_attacking"].to_numpy().astype(bool), fl["away_attacking"].to_numpy().astype(bool)
    hD, aD = fl["home_defending"].to_numpy().astype(bool), fl["away_defending"].to_numpy().astype(bool)

    def table(mask_map):
        rows = []
        for name, m in mask_map.items():
            m = m & has
            n_h, n_a = int((m & ph).sum()), int((m & pa).sum())
            rows.append({"tag state": name, "frames_with_pseudo": n_h + n_a, "pseudo_H": n_h, "pseudo_A": n_a,
                         "%_pseudo_H": round(100 * n_h / max(n_h + n_a, 1), 1)})
        return pd.DataFrame(rows)

    res["attacking_only"] = table({
        "home attacking only": hA & ~aA, "away attacking only": aA & ~hA, "both attacking": hA & aA,
        "no attacking tag": ~hA & ~aA})
    combos = {}
    for a1 in (0, 1):
        for d1 in (0, 1):
            for a2 in (0, 1):
                for d2 in (0, 1):
                    combos["homeAtt=%d homeDef=%d awayAtt=%d awayDef=%d" % (a1, d1, a2, d2)] = (
                        (hA == bool(a1)) & (hD == bool(d1)) & (aA == bool(a2)) & (aD == bool(d2)))
    t = table(combos)
    res["all_flag_combinations"] = t[t["frames_with_pseudo"] >= 100].reset_index(drop=True)
    htr, atr = tr["home_attacking"].to_numpy().astype(bool), tr["away_attacking"].to_numpy().astype(bool)
    res["transition_split"] = table({
        "home attacking only, transition": hA & ~aA & htr, "home attacking only, NOT transition": hA & ~aA & ~htr,
        "away attacking only, transition": aA & ~hA & atr, "away attacking only, NOT transition": aA & ~hA & ~atr})
    lab = tag_only_labels(events, use_defending=False, match_info=mi)
    rows = []
    for t_lab in ("H", "A", "X"):
        m = (lab == t_lab) & has
        rows.append({"tag_only says": t_lab, "pseudo_H": int((m & ph).sum()), "pseudo_A": int((m & pa).sum())})
    c = pd.DataFrame(rows)
    spoken = c[c["tag_only says"] != "X"]
    res["tag_only_vs_pseudo"] = c
    res["tag_only_agreement_%_when_it_speaks"] = round(100 * (c.loc[c["tag_only says"] == "H", "pseudo_H"].sum() +
                                                              c.loc[c["tag_only says"] == "A", "pseudo_A"].sum()) /
                                                       max(spoken[["pseudo_H", "pseudo_A"]].sum().sum(), 1), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracking"); ap.add_argument("events")
    ap.add_argument("--radius", type=float, default=1.5); ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--min-s", type=float, default=0.5)
    a = ap.parse_args()
    header = pd.read_csv(a.tracking, nrows=0).columns
    need = ["frame_id", "ball_x", "ball_y"] + ["%d_ball_dist_m" % n for n in range(1, N_SLOTS + 1)] + \
           ["%d_team" % n for n in range(1, N_SLOTS + 1)]
    absent = [c for c in need if c not in header and c != "frame_id"]       # frame_id / video_frame: see tracking_io
    if absent:
        sys.exit("tracking csv lacks columns: %s" % absent[:6])
    table = read_tracking(a.tracking, need)
    pseudo, info = pseudo_labels(table, a.radius, a.gap, a.min_s)
    print("slots: home %d, away %d; median nearest-player distance when the ball is seen: %.2f m (sanity: should be a few metres)" %
          (info["home_slots"], info["away_slots"], info["median_nearest_player_m_when_seen"]))
    res = report(pseudo, pd.read_csv(a.events, dtype=str, keep_default_na=False))
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    for k, v in res.items():
        print("\n%s\n%s" % (k.upper().replace("_", " "), v.to_string(index=False) if hasattr(v, "to_string") else v))


if __name__ == "__main__":
    main()

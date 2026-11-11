"""Assign a team to the Throw-in tags from the "ALL OFFENSIVE ACTIONS" timeline row.

    python smart_tags_throw_ins.py events.csv                 # report + write events_throw_ins.csv
    python smart_tags_throw_ins.py events.csv --first-part 0.5 --min-margin-s 1

Kedro node (before build_tag_features / build_phase_flags):
    node(assign_throw_ins, ["raw_smart_tag", "params:features", "match_info"], "smart_tag_filled")

RULE (offensive actions = union per team of its attacking-phase tags, same codes as build_phase_flags)
  1. MOST TIME.  Over the throw-in window [start, start + first_part * (end - start)) (default: the whole
     tag window) count the seconds each team is in an offensive action.  The team with more seconds owns
     the throw-in if its lead is larger than min_margin_s (default 0 = any lead).
  2. LAST TEAM (off by default; fallback=True / --fallback).  If that is not decisive (tie, or neither team attacks in the window) the team whose
     offensive action was last active, looking back from the end of the window, owns it.  Both teams
     active on the very last frame of the window -> still undecided.  max_lookback_s (default None =
     unlimited) limits how far back to look; throw_in_last_gap_s reports how far back it actually was,
     so you can judge how trustworthy a fallback assignment is.
  3. Otherwise unresolved (the default outcome when step 1 is not decisive).
Nothing else is used: no ball, no defending rows, no video, no ground truth.  The ordering of the 35 s
window relative to the actual throw-in is not known, which is why the whole window is the default.
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

from smart_tags import PHASE_ATTACKING, SET_PIECES, _code_key, _make_signer, _text

log = logging.getLogger(__name__)
FPS = 25


def _seconds(col):
    """numbers are seconds; strings like 0:12:34.5 are parsed as h:mm:ss."""
    col = col.astype(str).str.strip()
    num = pd.to_numeric(col, errors="coerce")
    return num if num.notna().all() else pd.to_timedelta(col).dt.total_seconds()


def ensure_frames(smart_tag, fps=25):
    """Return the table with start_frame / end_frame; if absent, frame = timestamp (seconds) x fps."""
    f = smart_tag.copy()
    f.columns = [str(c).strip() for c in f.columns]
    if "start_frame" in f.columns and "end_frame" in f.columns:
        return f
    if "timestamp" not in f.columns:
        raise ValueError("smart tag table has neither start_frame/end_frame nor timestamp; it has %s" % list(f.columns))
    f["start_frame"] = (_seconds(f["timestamp"]) * fps).round()
    if "end_timestamp" in f.columns and f["end_timestamp"].astype(str).str.strip().ne("").any():
        f["end_frame"] = (_seconds(f["end_timestamp"]) * fps).round()
    else:
        f["end_frame"] = f["start_frame"] + pd.to_numeric(f["duration_frames"])
    ko = f[f["Type"].astype(str).str.contains("Kick Off .Start", case=False, regex=True)] if "Type" in f.columns else f.iloc[:0]
    hit = [bool(((ko["start_frame"] <= k) & (ko["end_frame"] > k)).any()) for k in (11185, 113742)]
    if all(hit):
        log.info("frames derived as timestamp x %d; both documented kick-offs fall inside a 'Kick Off (Start)' window", fps)
    else:
        log.warning("frames derived as timestamp x %d but the documented kick-off frames 11185 / 113742 are NOT inside "
                    "a 'Kick Off (Start)' window (%s): the time base may be wrong", fps, hit)
    return f


OUT_COLUMNS = ["Team_filled", "team_fill_source", "throw_in_att_s_home", "throw_in_att_s_away",
               "throw_in_rule", "throw_in_last_gap_s"]


def _prep(smart_tag, params, match_info):
    f = ensure_frames(smart_tag)
    missing = [c for c in ("code", "start_frame", "end_frame", "Team", "Type") if c not in f.columns]
    if missing:
        raise ValueError("smart tag table is missing column(s) %s; it has %s" % (missing, list(f.columns)))
    a = pd.to_numeric(f["start_frame"], errors="coerce")
    b = pd.to_numeric(f["end_frame"], errors="coerce")
    ok = (a.notna() & b.notna()).to_numpy()
    a, b = a.fillna(-1).astype(np.int64).to_numpy(), b.fillna(-1).astype(np.int64).to_numpy()
    sign_of = _make_signer(params, match_info)
    signs = np.array([sign_of(_text(t)) for t in f["Team"]])
    n = int(b[ok].max()) + 1
    codes = {_code_key(c) for c in params.get("attacking_codes", PHASE_ATTACKING)}
    diff = {1: np.zeros(n + 1, np.int32), -1: np.zeros(n + 1, np.int32)}
    for code, s, e, sg, good in zip(f["code"], a, b, signs, ok):
        if good and sg != 0 and _code_key(_text(code)) in codes:
            s, e = max(s, 0), min(e, n)
            if e > s:
                diff[sg][s] += 1
                diff[sg][e] -= 1
    on = {sg: np.cumsum(d[:n]) > 0 for sg, d in diff.items()}                       # offensive action active?
    pref = {sg: np.concatenate([[0], np.cumsum(v, dtype=np.int64)]) for sg, v in on.items()}
    idx = np.arange(n)
    last_on = {sg: np.maximum.accumulate(np.where(v, idx, -1)) for sg, v in on.items()}   # last active frame <= t
    is_throw = np.array([_code_key(_text(c)) == SET_PIECES and "throw" in _text(t).lower()
                         for c, t in zip(f["code"], f["Type"])]) & ok
    names = {}
    for sg, fallback in ((1, match_info.get("home_team", "Barcelona")), (-1, match_info.get("away_team", "Athletic"))):
        seen = [_text(t) for t, s in zip(f["Team"], signs) if s == sg and _text(t)]
        names[sg] = pd.Series(seen).mode().iloc[0] if seen else fallback
    return f, a, b, signs, is_throw, pref, last_on, names, n


def assign_throw_ins(smart_tag: pd.DataFrame, params: dict = None, match_info: dict = None) -> pd.DataFrame:
    """Copy of the tag table plus OUT_COLUMNS (see module docstring).

    params: first_part=1.0, min_margin_s=0.0, fallback=False, max_lookback_s=None, replace_team=False,
    attacking_codes, team_aliases.
    throw_in_rule: most_time_home / most_time_away / last_team_home / last_team_away / undecided / given.
    """
    params, match_info = params or {}, match_info or {}
    first_part = params.get("first_part", 1.0)
    margin_min = params.get("min_margin_s", 0.0)
    use_fallback, max_back = params.get("fallback", False), params.get("max_lookback_s", None)
    f, a, b, signs, is_throw, pref, last_on, names, n = _prep(smart_tag, params, match_info)
    filled = [_text(t) if sg != 0 else "" for t, sg in zip(f["Team"], signs)]
    source = ["given" if s != 0 else "" for s in signs]
    sec_h, sec_a, rule, gap = [np.nan] * len(f), [np.nan] * len(f), [""] * len(f), [np.nan] * len(f)
    for i in np.flatnonzero(is_throw):
        lo = max(a[i], 0)
        hi = min(a[i] + max(int(round((b[i] - a[i]) * first_part)), 1), n)
        sh = (pref[1][hi] - pref[1][lo]) / FPS if hi > lo else 0.0
        sa = (pref[-1][hi] - pref[-1][lo]) / FPS if hi > lo else 0.0
        sec_h[i], sec_a[i] = round(sh, 2), round(sa, 2)
        if signs[i] != 0:
            rule[i] = "given"
            continue
        winner, how = 0, "undecided"
        if abs(sh - sa) > margin_min and sh != sa:
            winner, how = (1 if sh > sa else -1), "most_time"
        elif use_fallback and hi > lo:
            end = min(hi, n) - 1
            lh, la = int(last_on[1][end]), int(last_on[-1][end])
            if max_back is not None:
                lh = lh if end - lh <= max_back * FPS else -1
                la = la if end - la <= max_back * FPS else -1
            if lh != la and max(lh, la) >= 0:
                winner, how = (1 if lh > la else -1), "last_team"
                gap[i] = round((end - max(lh, la)) / FPS, 2)
        if winner:
            filled[i], source[i] = names[winner], "throw_in_rule"
            rule[i] = "%s_%s" % (how, "home" if winner > 0 else "away")
        else:
            source[i], rule[i] = "unresolved", "undecided"
    out = f.copy()
    out["Team_filled"], out["team_fill_source"] = filled, source
    out["throw_in_att_s_home"], out["throw_in_att_s_away"] = sec_h, sec_a
    out["throw_in_rule"], out["throw_in_last_gap_s"] = rule, gap
    if params.get("replace_team", False):
        out["Team"] = [t if s == "throw_in_rule" else o for t, s, o in zip(filled, source, f["Team"])]
    r = pd.Series([rule[i] for i in np.flatnonzero(is_throw & (signs == 0))]).value_counts().to_dict()
    log.info("throw_ins: %d without a team: %s", int((is_throw & (signs == 0)).sum()), r)
    return out


def sweep(smart_tag, params=None, match_info=None, first_parts=(0.5, 1.0)):
    """Counts of the outcomes for the team-less throw-ins, for the whole window / its first half, with / without the fallback."""
    rows = []
    for fp in first_parts:
        for fb in (False, True):
            o = assign_throw_ins(smart_tag, {**(params or {}), "first_part": fp, "fallback": fb}, match_info)
            t = o[(o["throw_in_rule"] != "") & (o["throw_in_rule"] != "given")]
            vc = t["throw_in_rule"].str.replace("_home", "").str.replace("_away", "").value_counts()
            rows.append({"first_part": fp, "fallback": fb, "throw_ins": len(t), "most_time": int(vc.get("most_time", 0)),
                         "last_team": int(vc.get("last_team", 0)), "undecided": int(vc.get("undecided", 0))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events"); ap.add_argument("-o", "--out")
    ap.add_argument("--first-part", type=float, default=1.0); ap.add_argument("--min-margin-s", type=float, default=0.0)
    ap.add_argument("--fallback", action="store_true", help="step 2: use the last attacking team when step 1 is not decisive"); ap.add_argument("--max-lookback-s", type=float, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tag = pd.read_csv(a.events, dtype=str, keep_default_na=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("SWEEP (team-less throw-ins only)\n", sweep(tag).to_string(index=False))
    out = assign_throw_ins(tag, {"first_part": a.first_part, "min_margin_s": a.min_margin_s,
                                 "fallback": a.fallback, "max_lookback_s": a.max_lookback_s})
    t = out[~out["throw_in_rule"].isin(["", "given"])]
    print("\nSETTING first_part=%s min_margin_s=%s fallback=%s max_lookback_s=%s\n" %
          (a.first_part, a.min_margin_s, a.fallback, a.max_lookback_s), t["throw_in_rule"].value_counts().to_string())
    fb = t[t["throw_in_rule"].str.startswith("last_team")]
    if len(fb):
        print("\nhow far back the fallback had to look (s):\n", fb["throw_in_last_gap_s"].describe().round(1).to_string())
    path = a.out or a.events.rsplit(".", 1)[0] + "_throw_ins.csv"
    out.to_csv(path, index=False); print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())

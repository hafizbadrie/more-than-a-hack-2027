"""Step 1: a possession file from the Smart Tagging alone.  No tracking, no ball, no training.

    python tag_only_submission.py events.csv -o tag_only.csv
    python evaluation/validate_submission.py tag_only.csv          # from the challenge repo

Rule, per video frame (home = Barcelona = H, away = Athletic = A, as in DATA.md):
  - frames 110808-113741            -> D   (documented: all dead ball between the half-time break and the 2nd-half kick-off)
  - the home team is "in possession" when it is attacking (the ALL OFFENSIVE ACTIONS union) and, with
    --use-defending, also when the away team is defending; the away team likewise
  - only home in possession -> H, only away -> A
  - both teams, or neither  -> X   (abstain; spellF1 carries the previous label forward, macroF1 counts a miss)
Nothing else is labelled D: the 35 s set-piece windows are templates, not measured stoppages.

Needs smart_tags.py and smart_tags_throw_ins.py (for the timestamp x 25 fallback) in the same folder.
Output: header `frame,label`, 199,319 rows, frames 0..199318, as SUBMISSION.md requires.
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

from smart_tags import build_phase_flags
from smart_tags_throw_ins import ensure_frames

N_FRAMES = 199_319
DEAD_BLOCK = (110_808, 113_741)                 # inclusive, DATA.md / EVALUATION.md


def tag_only_labels(events: pd.DataFrame, use_defending=False, defending_means="defender",
                    match_info=None) -> np.ndarray:
    tags = ensure_frames(events)
    frame = pd.DataFrame({"frame": np.arange(N_FRAMES)})          # video frames, no offset
    flags = build_phase_flags(frame, tags, {"defending_team_means": defending_means},
                              match_info or {"home_team": "Barcelona", "away_team": "Athletic"})
    home = flags["home_attacking"].to_numpy().astype(bool)
    away = flags["away_attacking"].to_numpy().astype(bool)
    if use_defending:
        home = home | flags["away_defending"].to_numpy().astype(bool)
        away = away | flags["home_defending"].to_numpy().astype(bool)
    label = np.full(N_FRAMES, "X", dtype="<U1")
    label[home & ~away] = "H"
    label[away & ~home] = "A"
    label[DEAD_BLOCK[0]:DEAD_BLOCK[1] + 1] = "D"
    return label


def write_submission(label: np.ndarray, path: str):
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": label}).to_csv(path, index=False)


def summarise(label: np.ndarray):
    graded = np.zeros(N_FRAMES, bool)
    graded[11185:90652] = True
    graded[110808:190198] = True
    g = label[graded]
    print("graded frames: %d   " % graded.sum() + "   ".join("%s %.1f%%" % (k, 100 * np.mean(g == k)) for k in "HADX"))
    for half, (lo, hi) in (("1st", (11185, 90651)), ("2nd", (113742, 190197))):
        seg = label[lo:hi + 1]
        live = seg[(seg == "H") | (seg == "A")]
        runs = int((np.diff((seg == "H").astype(int)) == 1).sum() + (np.diff((seg == "A").astype(int)) == 1).sum())
        print("%s half: %d H/A runs, mean run length %.1f s" % (half, runs, (len(live) / max(runs, 1)) / 25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events"); ap.add_argument("-o", "--out", default="tag_only.csv")
    ap.add_argument("--use-defending", action="store_true")
    ap.add_argument("--defending-means", choices=["defender", "possessor"], default="defender")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    label = tag_only_labels(pd.read_csv(a.events, dtype=str, keep_default_na=False),
                            a.use_defending, a.defending_means)
    write_submission(label, a.out)
    summarise(label)
    print("wrote", a.out, "- now run: python evaluation/validate_submission.py", a.out)


if __name__ == "__main__":
    sys.exit(main())

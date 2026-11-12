"""Hybrid possession: HMM + geometry + Smart Tags + dead-ball evidence -> one H / A / D / X label per frame.

WHAT IT PRODUCES
----------------
`add_hybrid_possession(df, params)` returns a copy of `df` with one extra column, `hybrid_possession`:

    H  home team (team A) has the ball          A  away team (team B) has the ball
    D  dead ball (play stopped)                 X  unknown / no source could decide

The scorer carries X forward for the spell metric and counts it as a miss for macro F1; D breaks a possession.

HOW TO READ THIS FILE
---------------------
The whole method is the function `compute_steps` at the bottom: it calls the numbered steps in order, and each step is a
short function with its own docstring.  `add_hybrid_possession` (the Kedro node) just runs `compute_steps` and attaches
the final label, exactly as the previous single-function version did.

    Step 1  check_input        the columns exist, no missing / duplicate frames
    Step 2  read_settings      validate the parameters
    Step 3  prepare_frames     sort by period + frame, split into gap-free sequences, check the numeric columns
    Step 4  ball_visibility    is the ball tracked, and is it inside the pitch?
    Step 5  dead_ball          dead-ball evidence: ball out, ball still, players slow, inside a set-piece window
    Step 6  geometry           nearest team to the ball, smoothed; "clear" = a player is right on the ball
    Step 7  tags               Smart Tags: who is attacking / defending
    Step 8  merge_sources      priority: HMM < geometry < tags < clear geometry; dead-ball frames set aside
    Step 9  smooth_and_finish  smooth H/A, merge short possessions, write D, restore the original row order

To SEE every step in a notebook:
    steps = compute_steps(df, params)
    steps.table().head(300)            # one column per step, sorted by period/frame
    steps.source_share()               # share of frames decided by each source

INPUT COLUMNS (all required; one row per frame, any row order)
--------------------------------------------------------------
    period_id, video_frame        integer frame number, unique per period; gaps are allowed (they split sequences)
    ball_x, ball_y                ball position, normalised 0-1 over the pitch (NaN = not tracked)
    min_team_a_dist, min_team_b_dist   distance (m) of each team's nearest player to the ball (NaN = unknown)
    {n}_s  (one per player slot)  player speed (m/s); every column named digits + "_s" is used
    sp_kind, sp_since             Smart Tag set piece in progress (e.g. "corner kick") and seconds since it started
    home_attacking, away_attacking, home_defending, away_defending    0/1 Smart Tag phase flags
    hmm_possession                the HMM label per frame (H / A / D / X) from hmm_possession.py

PARAMETERS (params dict; checked in `read_settings`)
----------------------------------------------------
    fps                    int > 0      frames per second (25)
    smoothing_half_window  int >= 0     +- frames for the majority votes (50 = 2 s); 0 switches smoothing off
    geometry_first         bool         False: tags beat geometry unless a player is "clear"; True: geometry beats tags
                                        whenever it has distances (default False)
    clear_distance_m       float > 0    a player this close (m) to the ball makes geometry decisive
    still_distance_m       float > 0    ball moved less than this (m) within +-1 s  -> "still"
    slow_speed_mps         float > 0    median player speed below this within +-1 s -> "players slow"
    dead_min_duration_s    float >= 0   dead-ball stretches shorter than this are dropped
    dead_merge_gap_s       float >= 0   gaps this short between dead-ball stretches are closed
    min_possession_s       float >= 0   possessions shorter than this are merged into their neighbour
    min_tracked_players    int          the per-frame median speed needs at least this many tracked players
    pitch_m                [L, W]       pitch length and width in metres (turns the 0-1 ball position into metres)
    set_piece_cutoffs_s    {kind: s}    dead ball is only allowed during the first s seconds of that set-piece kind
    split_at_dead          bool         optional, default False.  True: smooth each block between dead-ball stretches on its
                                        own, so a possession is never merged across a stoppage (see Step 8/9 notes)

    Example values used in an earlier run of this pipeline (set-piece cutoffs were measured from tracking, the rest are
    judgement values): clear_distance_m 3.0, still_distance_m 0.5, slow_speed_mps 0.7, dead_min_duration_s 2,
    dead_merge_gap_s 2, min_possession_s 3, smoothing_half_window 50, set_piece_cutoffs_s
    {"corner kick": 5.6, "free kick": 5.2, "throw-in": 5.8, "kick off (after goal)": 4.0}.

THINGS TO KNOW
--------------
* The HMM contributes only its H and A labels.  Its D and X labels are ignored: D comes from the dead-ball evidence only.
* Ties in the nearest-player distance go to H (`a <= b`).
* "Sequence" = consecutive frames inside one period.  No vote, window or smoothing ever crosses a gap or a period change.
* Unknown (X) frames abstain in every vote; they neither support H nor A.
* Dead-ball frames are set to X before the final smoothing (the default).  Because the smoothing only splits its blocks at
  D frames, it then merges possessions ACROSS a stoppage.  `split_at_dead=True` changes that; it only affects frames next to
  dead-ball stretches.
* The result equals the previous single-function version frame by frame (test_hybrid_equivalence.py), except for the
  optional `split_at_dead` switch.
"""
import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stages.hmm_possession import _sequences

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "period_id", "video_frame", "min_team_a_dist", "min_team_b_dist",
    "ball_x", "ball_y", "sp_kind", "sp_since", "home_attacking",
    "away_attacking", "home_defending", "away_defending", "hmm_possession",
]
TAG_COLUMNS = ("home_attacking", "away_attacking", "home_defending", "away_defending")


# ======================================================================================================================
# Small building blocks (unchanged from the original code)
# ======================================================================================================================
def _runs_at_least(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Keep only the runs of True that are at least `minimum` frames long."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = np.zeros(len(mask), dtype=bool)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= minimum:
            result[start:end] = True
    return result


def _smooth_known_labels(
    labels: np.ndarray, sequences: list[np.ndarray], window: int
) -> np.ndarray:
    """Smooth known H/A labels, letting unknown frames abstain rather than split votes."""
    result = labels.copy()
    if window == 0:
        return result
    for sequence in sequences:
        original = labels[sequence]
        indices = np.arange(len(sequence))
        lo = np.maximum(0, indices - window)
        hi = np.minimum(len(sequence), indices + window + 1)
        home = np.r_[0, np.cumsum(original == "H")]
        away = np.r_[0, np.cumsum(original == "A")]
        home_votes = home[hi] - home[lo]
        away_votes = away[hi] - away[lo]
        known = original != "X"
        result[sequence[known & (home_votes > away_votes)]] = "H"
        result[sequence[known & (away_votes > home_votes)]] = "A"
    return result


def _filter_dead_runs(
    dead: np.ndarray,
    sequences: list[np.ndarray],
    minimum: int,
    merge_gap: int,
) -> np.ndarray:
    """Close short internal gaps and discard short dead-ball runs per sequence."""
    result = np.zeros(len(dead), dtype=bool)
    for sequence in sequences:
        local = dead[sequence].copy()
        edges = np.diff(np.r_[False, local, False].astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for i in range(len(starts) - 1):
            if starts[i + 1] - ends[i] <= merge_gap:
                local[ends[i]:starts[i + 1]] = True
        edges = np.diff(np.r_[False, local, False].astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends):
            if end - start >= minimum:
                result[sequence[start:end]] = True
    return result


def _smooth_possession_labels(
    labels: np.ndarray,
    sequences: list[np.ndarray],
    window: int,
    minimum_run: float,
) -> np.ndarray:
    """Smooth H/A votes and merge short possessions without changing D or X frames.

    The labels are cut into blocks at every "D" frame, and each block is smoothed on its own (so a possession can not be
    merged across a dead-ball stretch).  Inside a block: (1) every H/A frame takes the majority H/A label within
    +/- window frames, (2) a possession shorter than `minimum_run` frames is merged into its longer neighbour.
    """
    result = labels.copy()
    for sequence in sequences:
        start = 0
        for end in range(len(sequence) + 1):
            if end < len(sequence) and labels[sequence[end]] != "D":
                continue
            segment = sequence[start:end]
            local = labels[segment]
            positions = np.flatnonzero(np.isin(local, ["H", "A"]))
            if len(positions):
                values = local[positions].copy()
                if window:
                    left = np.searchsorted(positions, positions - window, side="left")
                    right = np.searchsorted(positions, positions + window, side="right")
                    home = np.r_[0, np.cumsum(values == "H")]
                    away = np.r_[0, np.cumsum(values == "A")]
                    home_votes = home[right] - home[left]
                    away_votes = away[right] - away[left]
                    values[home_votes > away_votes] = "H"
                    values[away_votes > home_votes] = "A"
                if minimum_run:
                    while True:
                        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
                        if len(starts) == 1:
                            break
                        ends = np.r_[starts[1:], len(values)]
                        run_ends = np.r_[positions[starts[1:]], len(local)]
                        durations = run_ends - positions[starts]
                        shortest = int(np.argmin(durations))
                        if durations[shortest] >= minimum_run:
                            break
                        if shortest == 0:
                            neighbor = values[ends[shortest]]
                        elif shortest == len(starts) - 1:
                            neighbor = values[starts[shortest] - 1]
                        else:
                            left_duration = durations[shortest - 1]
                            right_duration = durations[shortest + 1]
                            neighbor = (
                                values[starts[shortest] - 1]
                                if left_duration >= right_duration
                                else values[ends[shortest]]
                            )
                        values[starts[shortest]:ends[shortest]] = neighbor
                result[segment[positions]] = values
            start = end + 1
    return result


# ======================================================================================================================
# Data containers: what each step hands to the next one
# ======================================================================================================================
@dataclass(frozen=True)
class Settings:
    """Step 2 output: the validated parameters."""
    fps: int                    # frames per second
    window: int                # half window (frames) for the majority votes
    geometry_first: bool       # True: geometry beats tags whenever the ball is inside the pitch (not only when "clear")
    clear_distance_m: float    # a player this close to the ball "clearly" decides who has it
    still_distance_m: float    # ball moved less than this over +-1 s  -> "still"
    slow_speed_mps: float      # median player speed below this over +-1 s -> "players slow"
    dead_min_duration_s: float # dead-ball stretches shorter than this are dropped
    dead_merge_gap_s: float    # gaps shorter than this between dead-ball stretches are closed
    min_possession_s: float    # possessions shorter than this are merged into their neighbour
    min_tracked_players: int   # the median player speed needs at least this many speeds
    pitch_m: tuple             # (length, width) in metres, to turn the 0-1 ball position into metres
    set_piece_cutoffs_s: dict  # {set-piece kind: seconds after its start in which dead ball is allowed}
    split_at_dead: bool        # NEW, default False: smooth each block between dead-ball stretches on its own


@dataclass
class Frames:
    """Step 3 output: the rows sorted by (period, frame), plus everything derived directly from them."""
    ordered: pd.DataFrame      # the input, sorted
    order: np.ndarray          # positions of the sorted rows in the input (to restore the order at the end)
    sequences: list            # gap-free runs of consecutive frames inside one period (positions in `ordered`)
    numeric: pd.DataFrame      # the numeric columns the method uses, as numbers
    speed_columns: list        # the {n}_s player speed columns
    hmm: np.ndarray            # the HMM label per frame (H/A/D/X)


@dataclass
class HybridSteps:
    """Every intermediate result, all aligned with `frames.ordered` (sorted by period and frame)."""
    settings: Settings
    frames: Frames
    seen: np.ndarray           # Step 4: ball position known
    inside: np.ndarray         # Step 4: ... and inside the pitch
    still: np.ndarray          # Step 5: ball hardly moved within +-1 s
    slow: np.ndarray           # Step 5: median player speed low within +-1 s
    outside: np.ndarray        # Step 5: ball outside the pitch for at least 1 s
    event_window: np.ndarray   # Step 5: inside the first N seconds of a set piece (N from the cutoffs)
    dead_raw: np.ndarray       # Step 5: evidence before the min-duration / merge-gap clean-up
    dead: np.ndarray           # Step 5: dead-ball frames after the clean-up
    geometry_labels: np.ndarray  # Step 6: H/A/X from the nearest team, smoothed
    has_distances: np.ndarray  # Step 6: ball inside the pitch and both team distances known
    clear: np.ndarray          # Step 6: ... and a player within clear_distance_m
    tags: np.ndarray           # Step 7: "H" if the tags say home has the ball, else "A" (only meaningful where tag_known)
    tag_known: np.ndarray      # Step 7: the tags name exactly one team
    merged: np.ndarray         # Step 8: the label after merging the sources, dead-ball frames already set
    source: np.ndarray         # Step 8: which source decided each frame
    smoothed: np.ndarray       # Step 9: after smoothing, before D is written
    final: np.ndarray          # Step 9: the final label (sorted order)
    restored: np.ndarray       # Step 9: the final label in the ORIGINAL row order

    def table(self) -> pd.DataFrame:
        """One row per frame (sorted), one column per step - the easiest way to follow what happened to a frame."""
        f = self.frames.ordered
        return pd.DataFrame({
            "period_id": f["period_id"].to_numpy(),
            "video_frame": f["video_frame"].to_numpy(),
            "1_hmm": self.frames.hmm,
            "4_ball": np.where(self.inside, "inside", np.where(self.seen, "outside", "missing")),
            "5_dead": self.dead,
            "6_geometry": self.geometry_labels,
            "6_clear": self.clear,
            "7_tags": np.where(self.tag_known, self.tags, "X"),
            "8_merged": self.merged,
            "8_source": self.source,
            "9_final": self.final,
        })

    def source_share(self) -> pd.Series:
        """Share of all frames decided by each source (dead-ball frames are listed as 'dead')."""
        src = np.where(self.dead, "dead", self.source)
        return pd.Series(src).value_counts(normalize=True).round(4)


# ======================================================================================================================
# Step 1-3: input checks and preparation
# ======================================================================================================================
def check_input(df: pd.DataFrame):
    """Step 1 - the required columns exist and every (period_id, video_frame) pair is present once.

    -> (names of the {n}_s speed columns, video_frame as float array)
    """
    speed_columns = [name for name in df if re.fullmatch(r"\d+_s", name)]
    missing = [name for name in REQUIRED_COLUMNS if name not in df]
    if missing or not speed_columns:
        raise ValueError(f"Hybrid input is missing required columns: {missing}; "
                         f"player speed columns found: {len(speed_columns)}")
    if df.empty or df[["period_id", "video_frame"]].isna().any().any():
        raise ValueError("Hybrid requires frames with non-missing period_id and video_frame")
    frames = pd.to_numeric(df["video_frame"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(frames).all() or (frames != np.floor(frames)).any():
        raise ValueError("Hybrid requires finite integer video_frame values")
    if df.assign(video_frame=frames).duplicated(["period_id", "video_frame"]).any():
        raise ValueError("Hybrid requires unique period_id/video_frame pairs")
    return speed_columns, frames


def read_settings(params: dict, n_speed_columns: int) -> Settings:
    """Step 2 - validate the parameters (same rules and messages as before)."""
    fps = params["fps"]
    window = params["smoothing_half_window"]
    geometry_first = params.get("geometry_first", False)
    split_at_dead = params.get("split_at_dead", False)
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("Hybrid fps must be a positive integer")
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("Hybrid smoothing_half_window must be a non-negative integer")
    if not isinstance(geometry_first, bool):
        raise ValueError("Hybrid geometry_first must be a boolean")
    if not isinstance(split_at_dead, bool):
        raise ValueError("Hybrid split_at_dead must be a boolean")
    for key in ("clear_distance_m", "still_distance_m", "slow_speed_mps"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Hybrid {key} must be positive and finite")
    for key in ("dead_min_duration_s", "dead_merge_gap_s", "min_possession_s"):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"Hybrid {key} must be non-negative and finite")
    minimum_players = params["min_tracked_players"]
    if (isinstance(minimum_players, bool) or not isinstance(minimum_players, int)
            or not 1 <= minimum_players <= n_speed_columns):
        raise ValueError("Hybrid min_tracked_players must be between 1 and the number of speed columns")
    pitch = params["pitch_m"]
    if len(pitch) != 2 or not all(np.isfinite(v) and v > 0 for v in pitch):
        raise ValueError("Hybrid pitch_m must contain two positive finite dimensions")
    cutoffs = params["set_piece_cutoffs_s"]
    if not isinstance(cutoffs, dict) or any(
        not isinstance(kind, str) or not np.isfinite(value) or value < 0
        for kind, value in cutoffs.items()
    ):
        raise ValueError("Hybrid set_piece_cutoffs_s must map names to non-negative finite seconds")
    return Settings(
        fps=fps, window=window, geometry_first=geometry_first,
        clear_distance_m=params["clear_distance_m"], still_distance_m=params["still_distance_m"],
        slow_speed_mps=params["slow_speed_mps"], dead_min_duration_s=params["dead_min_duration_s"],
        dead_merge_gap_s=params["dead_merge_gap_s"], min_possession_s=params["min_possession_s"],
        min_tracked_players=minimum_players, pitch_m=tuple(pitch), set_piece_cutoffs_s=cutoffs,
        split_at_dead=split_at_dead,
    )


def prepare_frames(df: pd.DataFrame, speed_columns: list, video_frame: np.ndarray) -> Frames:
    """Step 3 - sort by (period, frame), find the gap-free sequences, convert and check the numeric columns."""
    order = np.lexsort((video_frame, df["period_id"].to_numpy()))
    ordered = df.iloc[order].copy()
    ordered["video_frame"] = video_frame[order]
    sequences = _sequences(ordered)       # runs of consecutive frames inside one period; smoothing never crosses a gap
    numeric_columns = [
        "min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y", "sp_since",
        *TAG_COLUMNS, *speed_columns,
    ]
    numeric = ordered[numeric_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Hybrid input numeric values must be finite when present")
    if (numeric[["min_team_a_dist", "min_team_b_dist", *speed_columns]] < 0).any().any():
        raise ValueError("Hybrid distances and player speeds must be non-negative")
    for column in TAG_COLUMNS:
        if numeric[column].isna().any() or not numeric[column].isin([0, 1]).all():
            raise ValueError(f"Hybrid {column} must contain only 0 or 1")
    hmm = ordered["hmm_possession"].to_numpy()
    if not np.isin(hmm, ["H", "A", "D", "X"]).all():
        raise ValueError("Hybrid hmm_possession must contain only H, A, D or X")
    return Frames(ordered, order, sequences, numeric, speed_columns, hmm)


# ======================================================================================================================
# Step 4-5: ball visibility and dead-ball evidence
# ======================================================================================================================
def ball_visibility(f: Frames):
    """Step 4 - seen: a ball position exists.  inside: ... and it lies within the 0-1 pitch box.

    -> (seen, inside, ball_x, ball_y)   (ball_x / ball_y are the 0-1 normalised coordinates)
    """
    ball_x = f.numeric["ball_x"].to_numpy(dtype=float)
    ball_y = f.numeric["ball_y"].to_numpy(dtype=float)
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)
    inside = seen & (ball_x >= 0) & (ball_x <= 1) & (ball_y >= 0) & (ball_y <= 1)
    return seen, inside, ball_x, ball_y


def dead_ball(f: Frames, s: Settings, seen, inside, ball_x, ball_y):
    """Step 5 - dead-ball evidence.  A frame is dead when

        the ball has been OUTSIDE the pitch for >= 1 s                       (outside)
        OR  it is in the first N seconds of a set piece (N per kind) AND at least one of
              - the ball hardly moved within +-1 s                           (still)
              - the median tracked-player speed is low within +-1 s          (slow)
              - the ball is not tracked at all                               (~seen)

    and the result is cleaned up: gaps shorter than dead_merge_gap_s are closed, stretches shorter than
    dead_min_duration_s are dropped.

    -> (still, slow, outside, event_window, dead_raw, dead)
    """
    n = len(f.ordered)
    speeds = f.numeric[f.speed_columns]
    medians = speeds.median(axis=1).where(speeds.notna().sum(axis=1) >= s.min_tracked_players)
    still = np.zeros(n, dtype=bool)
    slow = np.zeros(n, dtype=bool)
    outside = np.zeros(n, dtype=bool)
    size = 2 * s.fps + 1                                      # +-1 second
    for seq in f.sequences:
        x = pd.Series(np.where(seen[seq], ball_x[seq] * s.pitch_m[0], np.nan))
        y = pd.Series(np.where(seen[seq], ball_y[seq] * s.pitch_m[1], np.nan))
        count = pd.Series(seen[seq].astype(int)).rolling(size, center=True, min_periods=1).sum()
        dx = x.rolling(size, center=True, min_periods=1).max() - x.rolling(size, center=True, min_periods=1).min()
        dy = y.rolling(size, center=True, min_periods=1).max() - y.rolling(size, center=True, min_periods=1).min()
        still[seq] = ((count >= 0.8 * size) & (dx < s.still_distance_m)
                      & (dy < s.still_distance_m)).to_numpy()
        speed = medians.iloc[seq].reset_index(drop=True).rolling(
            size, center=True, min_periods=s.fps
        ).median()
        slow[seq] = (speed < s.slow_speed_mps).to_numpy()
        outside[seq] = _runs_at_least(seen[seq] & ~inside[seq], s.fps)

    since = f.numeric["sp_since"].to_numpy(dtype=float)       # seconds since the set piece started
    event_window = np.zeros(n, dtype=bool)
    for kind, cutoff in s.set_piece_cutoffs_s.items():
        matches = f.ordered["sp_kind"].eq(kind).fillna(False).to_numpy(dtype=bool)
        event_window |= matches & np.isfinite(since) & (since >= 0) & (since <= cutoff)
    dead_raw = outside | (event_window & (still | slow | ~seen))
    dead = _filter_dead_runs(
        dead_raw,
        f.sequences,
        int(round(s.dead_min_duration_s * s.fps)),
        int(round(s.dead_merge_gap_s * s.fps)),
    )
    return still, slow, outside, event_window, dead_raw, dead


# ======================================================================================================================
# Step 6-7: geometry and tags
# ======================================================================================================================
def geometry(f: Frames, s: Settings, inside):
    """Step 6 - geometry: the team with the nearest player to the ball (ties go to H), smoothed with a majority vote.

    Only where the ball is inside the pitch and both distances exist; everywhere else X (unknown).
    "clear" marks the frames where the nearest player is within clear_distance_m of the ball.

    -> (geometry_labels, has_distances, clear)
    """
    a = f.numeric["min_team_a_dist"].to_numpy(dtype=float)
    b = f.numeric["min_team_b_dist"].to_numpy(dtype=float)
    has_distances = inside & np.isfinite(a) & np.isfinite(b)
    nearest_team = np.where(a <= b, "H", "A")
    labels = np.full(len(f.ordered), "X", dtype="<U1")
    labels[has_distances] = nearest_team[has_distances]
    labels = _smooth_known_labels(labels, f.sequences, s.window)
    clear = has_distances & (np.minimum(a, b) <= s.clear_distance_m)
    return labels, has_distances, clear


def tags(f: Frames):
    """Step 7 - Smart Tags.  Home has the ball when home is attacking OR away is defending; away the other way round.

    tag_known: exactly one of the two holds (both or neither -> the tags do not decide).
    -> (tags "H"/"A", tag_known)
    """
    n = f.numeric
    home_tag = (n["home_attacking"].to_numpy(dtype=int) == 1) | (n["away_defending"].to_numpy(dtype=int) == 1)
    away_tag = (n["away_attacking"].to_numpy(dtype=int) == 1) | (n["home_defending"].to_numpy(dtype=int) == 1)
    return np.where(home_tag, "H", "A"), home_tag ^ away_tag


# ======================================================================================================================
# Step 8-9: merge the sources, smooth, finish
# ======================================================================================================================
def merge_sources(f: Frames, s: Settings, geometry_labels, has_distances, clear, tag_labels, tag_known, dead):
    """Step 8 - one label per frame.  Later lines overwrite earlier ones, so the LAST matching rule wins:

        1. HMM            where the HMM says H or A                            (source "hmm")
        2. geometry       where the ball is inside the pitch with distances     (source "geometry")
        3. tags           where the tags name one team, and
                            geometry_first=False: the ball is not "clear"
                            geometry_first=True : there are no distances        (source "tags")
        4. clear geometry where a player is right on the ball                   (source "clear_geometry")
        5. dead-ball frames are set aside:  X  (or D when split_at_dead, so smoothing treats them as block borders)

    -> (labels, source)
    """
    n = len(f.ordered)
    hmm_known = np.isin(f.hmm, ["H", "A"])
    labels = np.full(n, "X", dtype="<U1")
    source = np.full(n, "none", dtype="<U14")
    labels[hmm_known] = f.hmm[hmm_known]
    source[hmm_known] = "hmm"
    labels[has_distances] = geometry_labels[has_distances]
    source[has_distances] = "geometry"
    tag_priority = ~has_distances if s.geometry_first else ~clear
    use_tags = tag_known & tag_priority
    labels[use_tags] = tag_labels[use_tags]
    source[use_tags] = "tags"
    labels[clear] = geometry_labels[clear]
    source[clear] = "clear_geometry"
    labels[dead] = "D" if s.split_at_dead else "X"
    return labels, source


def smooth_and_finish(f: Frames, s: Settings, merged, dead):
    """Step 9 - smooth the H/A labels (majority vote, then merge possessions shorter than min_possession_s),
    write D on the dead-ball frames and return to the original row order.

    -> (smoothed, final, restored)
    """
    smoothed = _smooth_possession_labels(merged, f.sequences, s.window, s.min_possession_s * s.fps)
    final = smoothed.copy()
    final[dead] = "D"
    restored = np.empty(len(final), dtype="<U1")
    restored[f.order] = final
    return smoothed, final, restored


# ======================================================================================================================
# The whole method, top to bottom
# ======================================================================================================================
def compute_steps(df: pd.DataFrame, params: dict) -> HybridSteps:
    """Run steps 1-9 and keep every intermediate result (see HybridSteps)."""
    speed_columns, video_frame = check_input(df)                                   # step 1
    s = read_settings(params, len(speed_columns))                                  # step 2
    f = prepare_frames(df, speed_columns, video_frame)                             # step 3
    seen, inside, ball_x, ball_y = ball_visibility(f)                              # step 4
    still, slow, outside, event_window, dead_raw, dead = dead_ball(                # step 5
        f, s, seen, inside, ball_x, ball_y)
    geometry_labels, has_distances, clear = geometry(f, s, inside)                 # step 6
    tag_labels, tag_known = tags(f)                                                # step 7
    merged, source = merge_sources(                                                # step 8
        f, s, geometry_labels, has_distances, clear, tag_labels, tag_known, dead)
    smoothed, final, restored = smooth_and_finish(f, s, merged, dead)              # step 9
    return HybridSteps(
        settings=s, frames=f, seen=seen, inside=inside, still=still, slow=slow, outside=outside,
        event_window=event_window, dead_raw=dead_raw, dead=dead, geometry_labels=geometry_labels,
        has_distances=has_distances, clear=clear, tags=tag_labels, tag_known=tag_known,
        merged=merged, source=source, smoothed=smoothed, final=final, restored=restored,
    )


def add_hybrid_possession(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Kedro node: add the column `hybrid_possession` (H / A / D / X) to the input, keeping its row order and columns."""
    steps = compute_steps(df, params)
    result = df.copy()
    result["hybrid_possession"] = steps.restored
    log.info("Hybrid label counts %s; dead-ball evidence frames %d",
             pd.Series(steps.restored).value_counts().to_dict(), int(steps.dead.sum()))
    return result
"""Hybrid possession, configurable: components -> selection rules -> smoothing steps -> one H / A / D / X label per frame.

The method has three parts, each driven by the config, so you can add, remove or re-configure approaches without
touching the code of the others:

    1. COMPONENTS   one feature (one label per frame) per approach:  hmm, geometry, clear_geometry, tags, dead_ball,
                    physical_dead, centroid_dead, ball_lost, touches, last_touch.  A component says H, A (or D for the dead-ball ones) where it has an
                    opinion and X where it abstains.  `touches` marks the frames where a player plays the ball; it is
                    not a selection rule but feeds the smoothing step snap_to_touch.
    2. SELECTION    an ordered list of rules: for every frame, the first rule whose component has an opinion (and whose
                    optional conditions hold) decides.  This is "priority" and "if-then-else" in one list.
    3. SMOOTHING    an ordered list of steps applied to the selected H/A labels: majority_vote, merge_short,
                    snap_to_touch (moves each possession change to the new team's first touch), ...

    D (dead ball) is handled in one place:  `smoothing.dead_handling` = "gap" (default; the smoothing does not see the
    stoppage, so a possession can be merged across it) or "split" (each side of a stoppage is smoothed on its own).
    D is written back on the dead frames at the end.

    config = {
      "fps": 25, "pitch_m": [105.0, 68.0],
      "components": {                       # a component is ON when it is listed here; delete it to remove it
        "hmm":            {},
        "geometry":       {"smoothing_half_window": 50},
        "clear_geometry": {"smoothing_half_window": 50, "clear_distance_m": 3.0},
        "tags":           {},
        "dead_ball":      {"still_distance_m": 0.5, "slow_speed_mps": 0.7, "dead_min_duration_s": 2.0,
                           "dead_merge_gap_s": 2.0, "min_tracked_players": 3,
                           "set_piece_cutoffs_s": {"corner kick": 5.6, ...}},
      },
      "selection": {"rules": [                     # first match wins, top to bottom
          {"use": "dead_ball"},
          {"use": "clear_geometry"},
          {"use": "tags"},                         # optional: "when_known": [...], "when_unknown": [...]
          {"use": "geometry"},
          {"use": "hmm"}]},
      "smoothing": {"dead_handling": "gap",
                    "steps": [{"type": "majority_vote", "half_window": 50},
                              {"type": "merge_short", "min_possession_s": 3.0}]},
    }

    DEFAULT_CONFIG (below) reproduces the previous hybrid_possession.py frame by frame (test_hybrid_pipeline.py).

TO ADD A NEW APPROACH
    @component("my_feature", requires=["some_column"])
    def my_feature(ctx, params):
        ...return an array of "H" / "A" / "X" (one per row of ctx.ordered, sorted by period and frame)
  then list it under "components" and put a rule {"use": "my_feature"} where it should rank.
TO ADD A NEW SMOOTHING STEP
    @smoother("my_step")
    def my_step(local_labels, step, ctx): ...return the new labels of one block (array of "H"/"A"/"X")
  (use @smoother("my_step", needs_index=True) and fn(local, step, ctx, index) to read per-frame signals of ctx.signals)

HOW TO LOOK AT THE PIECES
    result = run(df, config)
    result.components_table()    # one column per component (original row order) - the "features"
    result.table()               # per frame: every component, who decided, selected, smoothed, final
    result.source_share()        # share of frames decided by each rule
    features(df, config)         # just the component columns, named hp_<component>, for Kedro
"""
import copy
import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stages.hmm_possession import _sequences

log = logging.getLogger(__name__)

TAG_COLUMNS = ("home_attacking", "away_attacking", "home_defending", "away_defending")
BASE_COLUMNS = ("period_id", "video_frame")

DEFAULT_CONFIG = {
    "fps": 25,
    "pitch_m": [105.0, 68.0],
    "components": {
        "hmm": {},
        "geometry": {"smoothing_half_window": 50},
        "clear_geometry": {"smoothing_half_window": 50, "clear_distance_m": 3.0},
        "tags": {},
        "dead_ball": {
            "still_distance_m": 0.5, "slow_speed_mps": 0.7, "dead_min_duration_s": 2.0, "dead_merge_gap_s": 2.0,
            "min_tracked_players": 3,
            "set_piece_cutoffs_s": {"corner kick": 5.6, "free kick": 5.2, "throw-in": 5.8, "kick off (after goal)": 4.0},
        },
    },
    "selection": {"rules": [
        {"use": "dead_ball"}, {"use": "clear_geometry"}, {"use": "tags"}, {"use": "geometry"}, {"use": "hmm"},
    ]},
    "smoothing": {
        "dead_handling": "gap",
        "steps": [{"type": "majority_vote", "half_window": 50},
                  {"type": "merge_short", "min_possession_s": 3.0}],
    },
}


def deep_update(base: dict, changes: dict) -> dict:
    """A copy of `base` with `changes` merged in (dicts are merged key by key, everything else is replaced)."""
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_from_legacy_params(params: dict) -> dict:
    """Translate the parameter dict of hybrid_possession.py into the new config (same behaviour)."""
    window = params["smoothing_half_window"]
    geometry_first = params.get("geometry_first", False)
    order = ["dead_ball", "clear_geometry"] + (["geometry", "tags"] if geometry_first else ["tags", "geometry"]) + ["hmm"]
    return {
        "fps": params["fps"], "pitch_m": list(params["pitch_m"]),
        "components": {
            "hmm": {},
            "geometry": {"smoothing_half_window": window},
            "clear_geometry": {"smoothing_half_window": window, "clear_distance_m": params["clear_distance_m"]},
            "tags": {},
            "dead_ball": {
                "still_distance_m": params["still_distance_m"], "slow_speed_mps": params["slow_speed_mps"],
                "dead_min_duration_s": params["dead_min_duration_s"], "dead_merge_gap_s": params["dead_merge_gap_s"],
                "min_tracked_players": params["min_tracked_players"],
                "set_piece_cutoffs_s": params["set_piece_cutoffs_s"],
            },
        },
        "selection": {"rules": [{"use": name} for name in order]},
        "smoothing": {
            "dead_handling": "split" if params.get("split_at_dead", False) else "gap",
            "steps": [{"type": "majority_vote", "half_window": window},
                      {"type": "merge_short", "min_possession_s": params["min_possession_s"]}],
        },
    }


# ======================================================================================================================
# Small building blocks (same code as hybrid_possession.py)
# ======================================================================================================================
def _runs_at_least(mask, minimum):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = np.zeros(len(mask), dtype=bool)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= minimum:
            result[start:end] = True
    return result


def _smooth_known_labels(labels, sequences, window):
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
        home_votes, away_votes = home[hi] - home[lo], away[hi] - away[lo]
        known = original != "X"
        result[sequence[known & (home_votes > away_votes)]] = "H"
        result[sequence[known & (away_votes > home_votes)]] = "A"
    return result


def _filter_dead_runs(dead, sequences, minimum, merge_gap):
    result = np.zeros(len(dead), dtype=bool)
    for sequence in sequences:
        local = dead[sequence].copy()
        edges = np.diff(np.r_[False, local, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for i in range(len(starts) - 1):
            if starts[i + 1] - ends[i] <= merge_gap:
                local[ends[i]:starts[i + 1]] = True
        edges = np.diff(np.r_[False, local, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends):
            if end - start >= minimum:
                result[sequence[start:end]] = True
    return result


# ======================================================================================================================
# Context: the sorted frames and everything components may need
# ======================================================================================================================
@dataclass
class Context:
    ordered: pd.DataFrame          # the input sorted by (period_id, video_frame)
    order: np.ndarray              # positions of the sorted rows in the input
    sequences: list                # gap-free runs of consecutive frames inside one period
    numeric: pd.DataFrame          # the numeric columns the enabled components use, as numbers
    speed_columns: list            # the {n}_s columns
    fps: int
    pitch_m: tuple
    signals: dict = field(default_factory=dict)   # named intermediate arrays (e.g. dead-ball evidence), for inspection
    _cache: dict = field(default_factory=dict)

    def cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def ball(self):
        """-> (seen, inside, ball_x, ball_y); x / y are the 0-1 normalised coordinates."""
        def make():
            bx = self.numeric["ball_x"].to_numpy(dtype=float)
            by = self.numeric["ball_y"].to_numpy(dtype=float)
            seen = np.isfinite(bx) & np.isfinite(by)
            inside = seen & (bx >= 0) & (bx <= 1) & (by >= 0) & (by <= 1)
            return seen, inside, bx, by
        return self.cached("ball", make)

    def geometry(self, window):
        """-> (labels, has_distances, nearest_distance): nearest team (ties go to H), smoothed with `window`."""
        def make():
            _, inside, _, _ = self.ball()
            a = self.numeric["min_team_a_dist"].to_numpy(dtype=float)
            b = self.numeric["min_team_b_dist"].to_numpy(dtype=float)
            has = inside & np.isfinite(a) & np.isfinite(b)
            labels = np.full(len(a), "X", dtype="<U1")
            labels[has] = np.where(a <= b, "H", "A")[has]
            return _smooth_known_labels(labels, self.sequences, window), has, np.minimum(a, b)
        return self.cached(("geometry", window), make)


# ======================================================================================================================
# Registries
# ======================================================================================================================
COMPONENTS = {}      # name -> (function, required columns)
SMOOTHERS = {}       # name -> function(local_labels, step, ctx) -> new labels of one block
INDEXED_SMOOTHERS = set()   # smoothers that also receive the block's row positions in ctx.ordered: fn(local, step, ctx, index)


def component(name, requires=()):
    """Register a component.  `requires` lists columns that must exist; "{n}_s" means the player speed columns."""
    def wrap(fn):
        COMPONENTS[name] = (fn, tuple(requires))
        return fn
    return wrap


def smoother(name, needs_index=False):
    """Register a smoothing step.  With needs_index=True the function is called as fn(local, step, ctx, index), where
    `index` holds the positions of the block's frames in ctx.ordered (to look up per-frame signals in ctx.signals)."""
    def wrap(fn):
        SMOOTHERS[name] = fn
        if needs_index:
            INDEXED_SMOOTHERS.add(name)
        return fn
    return wrap


def _positive(params, key, strict=True):
    v = params[key]
    ok = np.isfinite(v) and (v > 0 if strict else v >= 0)
    if not ok:
        raise ValueError("Hybrid %s must be %s and finite" % (key, "positive" if strict else "non-negative"))
    return v


def _window(params, key="smoothing_half_window"):
    w = params[key]
    if not isinstance(w, int) or isinstance(w, bool) or w < 0:
        raise ValueError("Hybrid %s must be a non-negative integer" % key)
    return w


# ---------------------------------------------------------------------------------------------------------------- hmm
@component("hmm", requires=["hmm_possession"])
def hmm_component(ctx, params):
    """The HMM label where it says H or A.  Its D and X are ignored (D comes from dead_ball only)."""
    hmm = ctx.ordered["hmm_possession"].to_numpy()
    if not np.isin(hmm, ["H", "A", "D", "X"]).all():
        raise ValueError("Hybrid hmm_possession must contain only H, A, D or X")
    return np.where(np.isin(hmm, ["H", "A"]), hmm, "X").astype("<U1")


# ---------------------------------------------------------------------------------------------------------- geometry
@component("geometry", requires=["min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y"])
def geometry_component(ctx, params):
    """The team with the nearest player to the ball (ties go to H), majority-smoothed; X without distances."""
    labels, _, _ = ctx.geometry(_window(params))
    return labels


@component("clear_geometry", requires=["min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y"])
def clear_geometry_component(ctx, params):
    """The same geometry label, but only where a player is within clear_distance_m of the ball; X elsewhere."""
    labels, has, nearest = ctx.geometry(_window(params))
    clear = has & (nearest <= _positive(params, "clear_distance_m"))
    ctx.signals["clear"] = clear
    return np.where(clear, labels, "X").astype("<U1")


# ------------------------------------------------------------------------------------------------------------- tags
@component("tags", requires=TAG_COLUMNS)
def tags_component(ctx, params):
    """Home has the ball when home is attacking OR away is defending; away the other way round.  X if both or neither."""
    n = ctx.numeric
    home = (n["home_attacking"].to_numpy(dtype=int) == 1) | (n["away_defending"].to_numpy(dtype=int) == 1)
    away = (n["away_attacking"].to_numpy(dtype=int) == 1) | (n["home_defending"].to_numpy(dtype=int) == 1)
    return np.where(home ^ away, np.where(home, "H", "A"), "X").astype("<U1")


# ---------------------------------------------------------------------------------------------------------- dead_ball
def player_speed(speeds, min_players, top_k=None):
    """One speed per frame from the {n}_s columns (a DataFrame): the median of the tracked players, or with top_k the mean of
    the top_k fastest ones.  NaN where fewer than min_players (and, with top_k, fewer than top_k) players are tracked."""
    tracked = speeds.notna().sum(axis=1)
    if top_k is None:
        return speeds.median(axis=1).where(tracked >= min_players)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("Hybrid slow_top_k must be a positive integer")
    values = np.sort(np.where(speeds.notna().to_numpy(), speeds.to_numpy(dtype=float), -np.inf), axis=1)[:, ::-1][:, :top_k]
    fastest = pd.Series(values.mean(axis=1), index=speeds.index)
    return fastest.where((tracked >= max(min_players, top_k)) & np.isfinite(fastest))


def physical_signals(ctx, still_m, slow_mps, min_players, team_cols=None, team_thr=None, slow_window_s=1.0, slow_top_k=None):
    """The label-free physical signals, shared by the dead-ball components (cached per parameter set).

    still       the ball moved less than still_m (m) within +-1 s (and is tracked for 80% of that window)
    slow        the player speed is below slow_mps within +-slow_window_s; the speed is the MEDIAN over the tracked players,
                or with slow_top_k = k the mean of the k FASTEST tracked players (in live play a few players always run,
                in a stoppage even the fastest walk) (default 1 s; a longer window
                asks for SUSTAINED slowness, so brief bursts of running do not hide a walking stoppage)
    outside     the ball has been outside the pitch for at least 1 s
    team_still  (only if team_cols / team_thr are given) both teams' centroid speeds are below team_thr within +-1 s
    -> (still, slow, outside, team_still)   boolean arrays, one value per row of ctx.ordered
    """
    key = ("physical", still_m, slow_mps, min_players, tuple(team_cols or ()), team_thr, slow_window_s, slow_top_k)
    if key in ctx._cache:
        return ctx._cache[key]
    fps, (length, width) = ctx.fps, ctx.pitch_m
    seen, inside, bx, by = ctx.ball()
    n = len(ctx.ordered)
    speeds = ctx.numeric[ctx.speed_columns]
    medians = player_speed(speeds, min_players, slow_top_k)
    still, slow, outside, team_still = (np.zeros(n, dtype=bool) for _ in range(4))
    if team_thr is not None:
        motion = ctx.numeric[list(team_cols)].max(axis=1, skipna=True).where(ctx.numeric[list(team_cols)].notna().any(axis=1))
    size = 2 * fps + 1                                                       # +-1 second
    slow_size = 2 * int(round(slow_window_s * fps)) + 1                      # +-slow_window_s for the player speeds
    for seq in ctx.sequences:
        x = pd.Series(np.where(seen[seq], bx[seq] * length, np.nan))
        y = pd.Series(np.where(seen[seq], by[seq] * width, np.nan))
        count = pd.Series(seen[seq].astype(int)).rolling(size, center=True, min_periods=1).sum()
        dx = x.rolling(size, center=True, min_periods=1).max() - x.rolling(size, center=True, min_periods=1).min()
        dy = y.rolling(size, center=True, min_periods=1).max() - y.rolling(size, center=True, min_periods=1).min()
        still[seq] = ((count >= 0.8 * size) & (dx < still_m) & (dy < still_m)).to_numpy()
        speed = medians.iloc[seq].reset_index(drop=True).rolling(slow_size, center=True, min_periods=slow_size // 2).median()
        slow[seq] = (speed < slow_mps).to_numpy()
        outside[seq] = _runs_at_least(seen[seq] & ~inside[seq], fps)
        if team_thr is not None:
            smooth = motion.iloc[seq].reset_index(drop=True).rolling(size, center=True, min_periods=fps).median()
            team_still[seq] = (smooth < team_thr).to_numpy()
    ctx._cache[key] = (still, slow, outside, team_still)
    return ctx._cache[key]


@component("dead_ball", requires=["ball_x", "ball_y", "sp_kind", "sp_since", "{n}_s"])
def dead_ball_component(ctx, params):
    """D where play is stopped, X elsewhere.  A frame is dead when

        the ball has been OUTSIDE the pitch for >= 1 s                                   (outside)
        OR  it is in the first N seconds of a set piece (N per kind) AND at least one of
              the ball hardly moved within +-1 s (still) / median player speed is low within +-1 s (slow) / ball not tracked

    then gaps shorter than dead_merge_gap_s are closed and stretches shorter than dead_min_duration_s are dropped.
    The intermediate signals are kept in ctx.signals (still, slow, outside, event_window, dead_raw, team_still).

    OPTIONAL team-level stillness (off unless "team_still_mps" is given): both teams' centroid speeds are low
    (columns "team_speed_columns", e.g. team_centroid_speed_<team> from the team features; the larger of the two,
    smoothed with a +-1 s rolling median, must be below team_still_mps).  A team that is not moving as a unit
    is a dead-ball sign even when single players still walk about, which the median-speed "slow" signal misses.
        team_still_anywhere False (default): it counts like still / slow, i.e. only inside set-piece windows
        team_still_anywhere True : it also marks dead ball outside the windows (still cleaned with dead_min_duration_s)
    """
    still_m = _positive(params, "still_distance_m")
    slow_mps = _positive(params, "slow_speed_mps")
    min_s = _positive(params, "dead_min_duration_s", strict=False)
    gap_s = _positive(params, "dead_merge_gap_s", strict=False)
    min_players = params["min_tracked_players"]
    if (isinstance(min_players, bool) or not isinstance(min_players, int)
            or not 1 <= min_players <= len(ctx.speed_columns)):
        raise ValueError("Hybrid min_tracked_players must be between 1 and the number of speed columns")
    cutoffs = params["set_piece_cutoffs_s"]
    if not isinstance(cutoffs, dict) or any(
            not isinstance(k, str) or not np.isfinite(v) or v < 0 for k, v in cutoffs.items()):
        raise ValueError("Hybrid set_piece_cutoffs_s must map names to non-negative finite seconds")

    team_cols = params.get("team_speed_columns")
    team_thr = params.get("team_still_mps")
    anywhere = params.get("team_still_anywhere", False)
    if not isinstance(anywhere, bool):
        raise ValueError("Hybrid team_still_anywhere must be a boolean")
    if (team_thr is None) != (not team_cols):
        raise ValueError("Hybrid team_still_mps and team_speed_columns must be given together")
    if team_thr is not None:
        _positive(params, "team_still_mps")
    fps = ctx.fps
    seen = ctx.ball()[0]
    n = len(ctx.ordered)
    still, slow, outside, team_still = physical_signals(ctx, still_m, slow_mps, min_players, team_cols, team_thr)

    since = ctx.numeric["sp_since"].to_numpy(dtype=float)
    event_window = np.zeros(n, dtype=bool)
    for kind, cutoff in cutoffs.items():
        matches = ctx.ordered["sp_kind"].eq(kind).fillna(False).to_numpy(dtype=bool)
        event_window |= matches & np.isfinite(since) & (since >= 0) & (since <= cutoff)
    dead_raw = outside | (event_window & (still | slow | team_still | ~seen))
    if anywhere:
        dead_raw = dead_raw | team_still
    dead = _filter_dead_runs(dead_raw, ctx.sequences, int(round(min_s * fps)), int(round(gap_s * fps)))
    ctx.signals.update(still=still, slow=slow, outside=outside, event_window=event_window, dead_raw=dead_raw,
                       team_still=team_still)
    return np.where(dead, "D", "X").astype("<U1")


@component("centroid_dead")
def centroid_dead_component(ctx, params):
    """Detect sustained low movement of both tracked team centroids."""
    speed_cols = params["team_speed_columns"]
    tracked_cols = params["team_tracked_columns"]
    for columns in (speed_cols, tracked_cols):
        if (not isinstance(columns, (list, tuple)) or len(columns) != 2
                or not all(isinstance(col, str) and col for col in columns)
                or len(set(columns)) != 2):
            raise ValueError("Hybrid centroid_dead requires two distinct columns per team")
    threshold = _positive(params, "max_speed_mps")
    window_s = _positive(params, "window_s")
    min_s = _positive(params, "min_duration_s", strict=False)
    gap_s = _positive(params, "merge_gap_s", strict=False)
    min_players = params["min_tracked_players"]
    if isinstance(min_players, bool) or not isinstance(min_players, int) or min_players < 1:
        raise ValueError("Hybrid centroid_dead min_tracked_players must be a positive integer")

    enough_players = ctx.numeric[list(tracked_cols)].ge(min_players).all(axis=1)
    speeds = ctx.numeric[list(speed_cols)].max(axis=1, skipna=False).where(enough_players)
    size = 2 * int(round(window_s * ctx.fps)) + 1
    raw = np.zeros(len(ctx.ordered), dtype=bool)
    for seq in ctx.sequences:
        smooth = speeds.iloc[seq].reset_index(drop=True).rolling(
            size, center=True, min_periods=size // 2 + 1
        ).median()
        raw[seq] = (smooth < threshold).to_numpy()
    dead = _filter_dead_runs(raw, ctx.sequences, int(round(min_s * ctx.fps)), int(round(gap_s * ctx.fps)))
    ctx.signals["centroid_dead_raw"] = raw
    return np.where(dead, "D", "X").astype("<U1")


# ---------------------------------------------------------------------------------------------------------- ball_lost
@component("ball_lost", requires=["ball_x", "ball_y", "{n}_s"])
def ball_lost_component(ctx, params):
    """D where the ball is untracked for a stretch that is about as long as a stoppage; X elsewhere.

    On this match the ball is untracked in runs that are either short occlusions (< 2 s), stoppage-sized (5-20 s) or
    tracking outages of 20 s to several minutes (the length distribution has no runs of 2-5 s).  Only the stoppage-sized
    runs are taken: a run is D when
        min_s <= its length <= max_s,
        it does not touch the start or end of a gap-free sequence (a run cut by a period border or a tracking hole has an
        unknown length), and
        (optional) the median player speed inside it is at most max_median_speed_mps (players walk when play is stopped).
    Check the thresholds with untracked_runs_check.py (label-free, whole match) before using them.

        params: min_s, max_s, min_tracked_players, optional max_median_speed_mps
    The runs are kept in ctx.signals["ball_lost_runs"] as (first row, last row, median speed) for inspection.
    """
    min_s, max_s = _positive(params, "min_s"), _positive(params, "max_s")
    if max_s < min_s:
        raise ValueError("Hybrid ball_lost max_s must be at least min_s")
    ceiling = params.get("max_median_speed_mps")
    if ceiling is not None:
        _positive(params, "max_median_speed_mps")
    min_players = params["min_tracked_players"]
    if (isinstance(min_players, bool) or not isinstance(min_players, int)
            or not 1 <= min_players <= len(ctx.speed_columns)):
        raise ValueError("Hybrid min_tracked_players must be between 1 and the number of speed columns")
    fps = ctx.fps
    seen = ctx.ball()[0]
    speeds = ctx.numeric[ctx.speed_columns]
    medians = speeds.median(axis=1).where(speeds.notna().sum(axis=1) >= min_players).to_numpy()
    out = np.full(len(seen), "X", dtype="<U1")
    kept = []
    for seq in ctx.sequences:
        lost = ~seen[seq]
        edges = np.diff(np.r_[False, lost, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            length = (end - start) / fps
            if start == 0 or end == len(seq) or not min_s <= length <= max_s:
                continue
            rows = seq[start:end]
            speed = float(np.nanmedian(medians[rows])) if np.isfinite(medians[rows]).any() else np.nan
            if ceiling is not None and not speed <= ceiling:
                continue
            out[rows] = "D"
            kept.append((int(rows[0]), int(rows[-1]), speed))
    ctx.signals["ball_lost_runs"] = kept
    return out


# ------------------------------------------------------------------------------------- hmm_dead / physical_dead
def _agreement_dead(ctx, params, require_hmm):
    """Shared by hmm_dead (require_hmm True) and physical_dead (False): D where the chosen physical signals hold
    (and, for hmm_dead, the HMM says D too), cleaned with min_duration_s / merge_gap_s.  Works anywhere in the match."""
    known = {"still_distance_m", "slow_speed_mps", "slow_window_s", "slow_top_k", "min_tracked_players", "require", "agree_with",
             "agree_mode", "min_duration_s", "merge_gap_s", "team_speed_columns", "team_still_mps"}
    if set(params) - known:
        raise ValueError("Hybrid physical_dead / hmm_dead: unknown parameter(s) %s (known: %s)"
                         % (sorted(set(params) - known), sorted(known)))
    still_m = _positive(params, "still_distance_m")
    slow_mps = _positive(params, "slow_speed_mps")
    min_s = _positive(params, "min_duration_s", strict=False)
    gap_s = _positive(params, "merge_gap_s", strict=False)
    min_players = params["min_tracked_players"]
    if (isinstance(min_players, bool) or not isinstance(min_players, int)
            or not 1 <= min_players <= len(ctx.speed_columns)):
        raise ValueError("Hybrid min_tracked_players must be between 1 and the number of speed columns")
    require = list(params.get("require", []))
    agree = list(params.get("agree_with", [] if require else ["still", "slow"]))
    mode = params.get("agree_mode", "any")
    names = ("still", "slow", "team_still")
    if any(a not in names for a in require + agree) or not (require or agree):
        raise ValueError("Hybrid agree_with / require must be lists of still / slow / team_still, not both empty")
    if mode not in ("any", "all"):
        raise ValueError("Hybrid agree_mode must be 'any' or 'all'")
    team_cols, team_thr = params.get("team_speed_columns"), params.get("team_still_mps")
    if (team_thr is None) != (not team_cols):
        raise ValueError("Hybrid team_still_mps and team_speed_columns must be given together")
    if "team_still" in agree + require and team_thr is None:
        raise ValueError("Hybrid agree_with has team_still but team_still_mps / team_speed_columns are missing")
    if team_thr is not None:
        _positive(params, "team_still_mps")
    slow_window_s = _positive(params, "slow_window_s") if "slow_window_s" in params else 1.0
    slow_top_k = params.get("slow_top_k")
    still, slow, _, team_still = physical_signals(ctx, still_m, slow_mps, min_players, team_cols, team_thr, slow_window_s,
                                                  slow_top_k)
    signals = {"still": still, "slow": slow, "team_still": team_still}
    physical = np.ones(len(still), bool)
    if agree:
        stack = np.vstack([signals[a] for a in agree])
        physical = stack.all(axis=0) if mode == "all" else stack.any(axis=0)
    for r in require:                                   # `require`: every listed signal must hold; agree_with: any/all of
        physical = physical & signals[r]
    raw = physical
    if require_hmm:
        hmm = ctx.ordered["hmm_possession"].to_numpy()
        if not np.isin(hmm, ["H", "A", "D", "X"]).all():
            raise ValueError("Hybrid hmm_possession must contain only H, A, D or X")
        raw = (hmm == "D") & physical
    dead = _filter_dead_runs(raw, ctx.sequences, int(round(min_s * ctx.fps)), int(round(gap_s * ctx.fps)))
    return raw, physical, dead


@component("hmm_dead", requires=["hmm_possession", "ball_x", "ball_y", "{n}_s"])
def hmm_dead_component(ctx, params):
    """D where the HMM says D AND the physical evidence agrees; X elsewhere.

    CAUTION: on this match the HMM's D state fires on ~94% of clearly live frames (see hmm_dead_check.py), so it is not a
    dead-ball detector and adds no independent information.  Kept for completeness; prefer physical_dead.

        params: still_distance_m, slow_speed_mps, slow_window_s (default 1.0), slow_top_k (default None = median speed), min_tracked_players, require (all must hold), agree_with (list of still / slow / team_still),
                agree_mode ("any" | "all"), min_duration_s, merge_gap_s, optional team_speed_columns + team_still_mps
    """
    raw, physical, dead = _agreement_dead(ctx, params, require_hmm=True)
    ctx.signals.update(hmm_dead_raw=raw, hmm_dead_physical=physical)
    return np.where(dead, "D", "X").astype("<U1")


@component("physical_dead", requires=["ball_x", "ball_y", "{n}_s"])
def physical_dead_component(ctx, params):
    """D where the chosen physical signals hold, anywhere in the match (not only in set-piece windows); X elsewhere.

    Same parameters as hmm_dead, no HMM.  Example "ball still AND (players slow OR team still)":
    require ["still"], agree_with ["slow", "team_still"], agree_mode "any".  Typical choices: agree_with ["still"] (ball at rest for a while), or
    agree_with ["still", "slow"] with agree_mode "all" (ball AND players at rest).  Pick the signals and the minimum
    duration with hmm_dead_check.py: accept a rule only if it fires on <= 1% of clearly live frames.
    """
    raw, physical, dead = _agreement_dead(ctx, params, require_hmm=False)
    ctx.signals.update(physical_dead_raw=raw)
    return np.where(dead, "D", "X").astype("<U1")


# --------------------------------------------------------------------------------------------------------- touches
@component("touches", requires=["min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y"])
def touches_component(ctx, params):
    """H / A on the frames where a player plays the ball, X everywhere else (not meant as a selection rule).

    A touch is a sudden change of the ball's velocity while a player is at the ball:
        v_in(t)  = ball velocity over the window_frames before t,   v_out(t) = over the window_frames after t
        delta(t) = |v_out - v_in|  (m/s)
    t is a touch when delta is a local maximum within +-peak_s, delta >= min_delta_mps, and the nearest player of one
    team is within max_player_dist_m of the ball (the team whose nearest player is closer by more than team_margin_m
    gets the touch; if neither is clearly closer the touch is X).

        params: window_frames (int >= 1), min_delta_mps, peak_s, max_player_dist_m, team_margin_m (>= 0)
    Stores ctx.signals["touch_team"] (the labels) and ctx.signals["touch_delta_v"] (delta for every frame).
    """
    k = params["window_frames"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("Hybrid window_frames must be a positive integer")
    min_delta = _positive(params, "min_delta_mps")
    peak = max(int(round(_positive(params, "peak_s") * ctx.fps)), 1)
    max_dist = _positive(params, "max_player_dist_m")
    margin = _positive(params, "team_margin_m", strict=False)
    fps, (length, width) = ctx.fps, ctx.pitch_m
    seen, _, bx, by = ctx.ball()
    a = np.where(np.isfinite(ctx.numeric["min_team_a_dist"].to_numpy(dtype=float)),
                 ctx.numeric["min_team_a_dist"].to_numpy(dtype=float), np.inf)
    b = np.where(np.isfinite(ctx.numeric["min_team_b_dist"].to_numpy(dtype=float)),
                 ctx.numeric["min_team_b_dist"].to_numpy(dtype=float), np.inf)
    n = len(ctx.ordered)
    labels = np.full(n, "X", dtype="<U1")
    delta_all = np.full(n, np.nan)
    peaks = np.zeros(n, dtype=bool)                  # every touch, also the contested ones
    for seq in ctx.sequences:
        m = len(seq)
        if m < 2 * k + 1:
            continue
        x = np.where(seen[seq], bx[seq] * length, np.nan)
        y = np.where(seen[seq], by[seq] * width, np.nan)
        step_x, step_y = (x[k:] - x[:-k]) * fps / k, (y[k:] - y[:-k]) * fps / k     # velocity between t and t + k
        delta = np.full(m, np.nan)
        delta[k:m - k] = np.hypot(step_x[k:m - k] - step_x[:m - 2 * k], step_y[k:m - k] - step_y[:m - 2 * k])
        delta_all[seq] = delta
        top = pd.Series(delta).rolling(2 * peak + 1, center=True, min_periods=1).max().to_numpy()
        candidates = np.flatnonzero(np.isfinite(delta) & (delta >= min_delta) & (delta >= top))
        near_a = pd.Series(a[seq]).rolling(2 * k + 1, center=True, min_periods=1).min().to_numpy()
        near_b = pd.Series(b[seq]).rolling(2 * k + 1, center=True, min_periods=1).min().to_numpy()
        last = -10 ** 9
        for t in candidates:
            if t - last <= peak:                      # a plateau of equal maxima counts once
                continue
            last = t
            if min(near_a[t], near_b[t]) > max_dist:
                continue
            peaks[seq[t]] = True
            if near_a[t] + margin < near_b[t]:
                labels[seq[t]] = "H"
            elif near_b[t] + margin < near_a[t]:
                labels[seq[t]] = "A"
    ctx.signals["touch_team"] = labels
    ctx.signals["touch_delta_v"] = delta_all
    ctx.signals["touch_peak"] = peaks
    return labels


@component("last_touch", requires=["min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y"])
def last_touch_component(ctx, params):
    """H / A for the team that played the ball last, while the ball is travelling; X otherwise.

    Fills the frames where nobody is near the ball (a pass or a clearance in flight), which clear_geometry leaves open
    and which `tags` would otherwise decide by the attacking phase, i.e. "the team in the attacking phase keeps the
    ball".  Here the ball belongs to whoever kicked it until somebody else plays it, whatever the phase tag says.
    A touch (same detector and parameters as `touches`) holds its team from the touch to the next touch of anybody,
    for at most hold_s seconds, and never while the ball is outside the pitch or untracked.  A touch with no clear team
    (contested) ends the hold and abstains.

        params: the `touches` parameters (window_frames, min_delta_mps, peak_s, max_player_dist_m, team_margin_m) + hold_s
    Rank it after clear_geometry and before tags:  dead_ball, clear_geometry, last_touch, tags, geometry, hmm.
    """
    hold = int(round(_positive(params, "hold_s") * ctx.fps))
    touch_params = {k: v for k, v in params.items() if k != "hold_s"}
    ball_labels = touches_component(ctx, touch_params)
    _, inside, _, _ = ctx.ball()
    # every touch, also the contested ones (the detector's X touches are not in `labels`): use the delta peaks again
    peaks = ctx.signals["touch_peak"]
    labels = np.full(len(ball_labels), "X", dtype="<U1")
    for seq in ctx.sequences:
        t_all = np.flatnonzero(peaks[seq])
        for i, t in enumerate(t_all):
            team = ball_labels[seq[t]]
            if team == "X":
                continue
            end = t_all[i + 1] if i + 1 < len(t_all) else len(seq)
            end = min(end, t + hold + 1)
            span = np.arange(t, end)
            span = span[inside[seq[span]]]
            labels[seq[span]] = team
            outside = np.flatnonzero(~inside[seq[t:end]])
            if len(outside):                                  # stop at the first frame outside / untracked
                labels[seq[t + outside[0]:end]] = "X"
    ctx.signals["last_touch_team"] = labels
    return labels


# ======================================================================================================================
# Smoothing steps (each works on one block of frames between D frames)
# ======================================================================================================================
@smoother("majority_vote")
def majority_vote(local, step, ctx):
    """Every H/A frame takes the majority H/A label within +-half_window frames (X frames abstain; ties keep the label)."""
    window = _window(step, "half_window")
    out = local.copy()
    positions = np.flatnonzero(np.isin(local, ["H", "A"]))
    if len(positions) and window:
        values = local[positions].copy()
        left = np.searchsorted(positions, positions - window, side="left")
        right = np.searchsorted(positions, positions + window, side="right")
        home = np.r_[0, np.cumsum(values == "H")]
        away = np.r_[0, np.cumsum(values == "A")]
        home_votes, away_votes = home[right] - home[left], away[right] - away[left]
        values[home_votes > away_votes] = "H"
        values[away_votes > home_votes] = "A"
        out[positions] = values
    return out


@smoother("merge_short")
def merge_short(local, step, ctx):
    """While a possession is shorter than min_possession_s, merge the shortest into its longer neighbour."""
    minimum_run = _positive(step, "min_possession_s", strict=False) * ctx.fps
    out = local.copy()
    positions = np.flatnonzero(np.isin(local, ["H", "A"]))
    if len(positions) and minimum_run:
        values = local[positions].copy()
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
                neighbor = (values[starts[shortest] - 1] if durations[shortest - 1] >= durations[shortest + 1]
                            else values[ends[shortest]])
            values[starts[shortest]:ends[shortest]] = neighbor
        out[positions] = values
    return out


@smoother("snap_to_touch", needs_index=True)
def snap_to_touch(local, step, ctx, index):
    """Move every H<->A change to the first touch of the team that gains the ball.

    The geometry labels flip when the new team's player gets nearer to the ball than the old team's, which is usually
    before that player actually plays it.  For each change from team P to team Q at frame b, look at the touches
    (ctx.signals["touch_team"], from the `touches` component) within +-max_shift_s of b:
        * Q's touch that FOLLOWS the last touch of P inside the window is the new change point
          (if P never touches in the window: the Q touch nearest to b);
        * no such touch  ->  the change stays where it is.
    A change never moves past its neighbours (every possession keeps at least one frame) and X frames are not touched.

        step: {"type": "snap_to_touch", "max_shift_s": 3.0}
    """
    reach = int(round(_positive(step, "max_shift_s") * ctx.fps))
    touch = ctx.signals["touch_team"][index]
    out = local.copy()
    positions = np.flatnonzero(np.isin(local, ["H", "A"]))
    if len(positions) < 2 or not reach:
        return out
    values = local[positions]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    if len(starts) < 2:
        return out
    boundary = positions[starts[1:]]                 # first frame of every new possession (block-local index)
    from_team, to_team = values[starts[:-1]], values[starts[1:]]
    floor = positions[0] + 1                         # the previous possession keeps at least one frame
    for i, b in enumerate(boundary):
        ceiling = (boundary[i + 1] if i + 1 < len(boundary) else len(local)) - 1
        lo, hi = max(b - reach, floor), min(b + reach, ceiling)
        new_b = b
        if lo <= hi:
            window = touch[lo:hi + 1]
            gains = lo + np.flatnonzero(window == to_team[i])
            losses = lo + np.flatnonzero(window == from_team[i])
            target = None
            if len(gains):
                if len(losses):
                    later = gains[gains > losses[-1]]
                    target = int(later[0]) if len(later) else None
                else:
                    target = int(gains[np.argmin(np.abs(gains - b))])
            if target is not None:
                new_b = target
                if new_b > b:
                    out[b:new_b] = from_team[i]
                elif new_b < b:
                    out[new_b:b] = to_team[i]
        floor = new_b + 1
    return out


def _per_block(labels, sequences, fn):
    """Apply fn(labels_of_block, positions_of_block) to each block of frames between D frames (blocks never cross a
    sequence boundary)."""
    result = labels.copy()
    for sequence in sequences:
        start = 0
        for end in range(len(sequence) + 1):
            if end < len(sequence) and labels[sequence[end]] != "D":
                continue
            segment = sequence[start:end]
            if len(segment):
                result[segment] = fn(labels[segment], segment)
            start = end + 1
    return result


# ======================================================================================================================
# The three stages
# ======================================================================================================================
def check_config(config: dict):
    fps = config["fps"]
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("Hybrid fps must be a positive integer")
    pitch = config["pitch_m"]
    if len(pitch) != 2 or not all(np.isfinite(v) and v > 0 for v in pitch):
        raise ValueError("Hybrid pitch_m must contain two positive finite dimensions")
    for name in config["components"]:
        if name not in COMPONENTS:
            raise ValueError("Unknown component %r (known: %s)" % (name, sorted(COMPONENTS)))
    for rule in config["selection"]["rules"]:
        names = [rule["use"], *rule.get("when_known", []), *rule.get("when_unknown", [])]
        for name in names:
            if name not in config["components"]:
                raise ValueError("Selection rule uses %r, which is not listed under config['components']" % name)
    for step in config["smoothing"]["steps"]:
        if step["type"] not in SMOOTHERS:
            raise ValueError("Unknown smoothing step %r (known: %s)" % (step["type"], sorted(SMOOTHERS)))
        if step["type"] == "snap_to_touch" and "touches" not in config["components"]:
            raise ValueError("Smoothing step 'snap_to_touch' needs the 'touches' component under config['components']")
    if config["smoothing"].get("dead_handling", "gap") not in ("gap", "split"):
        raise ValueError("smoothing.dead_handling must be 'gap' or 'split'")


def build_context(df: pd.DataFrame, config: dict) -> Context:
    """Check the input, sort it by (period, frame), split it into gap-free sequences and convert the numeric columns."""
    check_config(config)
    speed_columns = [c for c in df if re.fullmatch(r"\d+_s", c)]
    needed = set(BASE_COLUMNS)
    for name in config["components"]:
        needed.update(COMPONENTS[name][1])
    extra = []                                   # columns named in a component's params, e.g. team_speed_columns
    for params in config["components"].values():
        for key, value in params.items():
            if key.endswith("_columns") and isinstance(value, (list, tuple)):
                extra += [c for c in value if c not in extra]
    needed.update(extra)
    wants_speed = "{n}_s" in needed
    needed.discard("{n}_s")
    missing = [c for c in sorted(needed) if c not in df]
    if missing or (wants_speed and not speed_columns):
        raise ValueError("Hybrid input is missing required columns: %s; player speed columns found: %d"
                         % (missing, len(speed_columns)))
    if df.empty or df[list(BASE_COLUMNS)].isna().any().any():
        raise ValueError("Hybrid requires frames with non-missing period_id and video_frame")
    video_frame = pd.to_numeric(df["video_frame"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(video_frame).all() or (video_frame != np.floor(video_frame)).any():
        raise ValueError("Hybrid requires finite integer video_frame values")
    if df.assign(video_frame=video_frame).duplicated(["period_id", "video_frame"]).any():
        raise ValueError("Hybrid requires unique period_id/video_frame pairs")

    order = np.lexsort((video_frame, df["period_id"].to_numpy()))
    ordered = df.iloc[order].copy()
    ordered["video_frame"] = video_frame[order]
    sequences = _sequences(ordered)
    numeric_columns = [c for c in ("min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y", "sp_since", *TAG_COLUMNS)
                       if c in needed] + extra
    if wants_speed:
        numeric_columns += speed_columns
    numeric = ordered[numeric_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Hybrid input numeric values must be finite when present")
    nonneg = [c for c in ("min_team_a_dist", "min_team_b_dist", *speed_columns, *extra) if c in numeric]
    if nonneg and (numeric[nonneg] < 0).any().any():
        raise ValueError("Hybrid distances and player speeds must be non-negative")
    for column in TAG_COLUMNS:
        if column in numeric and (numeric[column].isna().any() or not numeric[column].isin([0, 1]).all()):
            raise ValueError(f"Hybrid {column} must contain only 0 or 1")
    return Context(ordered, order, sequences, numeric, speed_columns if wants_speed else [],
                   config["fps"], tuple(config["pitch_m"]))


def compute_components(ctx: Context, config: dict) -> dict:
    """Stage 1 - run every listed component.  -> {name: array of H/A/X (D for dead_ball), sorted order}"""
    return {name: COMPONENTS[name][0](ctx, params) for name, params in config["components"].items()}


def select_labels(components: dict, selection: dict):
    """Stage 2 - for every frame the FIRST rule whose component has an opinion (and whose conditions hold) decides.

    A rule is {"use": name} plus optional  "when_known": [names that must have an opinion]  and
    "when_unknown": [names that must NOT have an opinion] - a component "has an opinion" when its label is not X.
    -> (labels, source)   source is the rule's component name, or "none"
    """
    n = len(next(iter(components.values())))
    labels = np.full(n, "X", dtype="<U1")
    source = np.full(n, "none", dtype="<U20")
    open_ = np.ones(n, dtype=bool)
    for rule in selection["rules"]:
        use = components[rule["use"]]
        take = open_ & (use != "X")
        for name in rule.get("when_known", []):
            take &= components[name] != "X"
        for name in rule.get("when_unknown", []):
            take &= components[name] == "X"
        labels[take] = use[take]
        source[take] = rule["use"]
        open_ &= ~take
    return labels, source


def _extend_dead(dead, ctx, lead_s, trail_s):
    """Grow every run of D by lead_s seconds before it and trail_s seconds after it (never across a period boundary).
    The dead-ball detectors use centered +-1 s windows, so they switch on ~1 s after play stopped; dead_lead_s ~ 1.0
    compensates that delay."""
    for name, v in (("dead_lead_s", lead_s), ("dead_trail_s", trail_s)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0:
            raise ValueError("smoothing.%s must be a number >= 0" % name)
    lead, trail = int(round(lead_s * ctx.fps)), int(round(trail_s * ctx.fps))
    if not (lead or trail) or not dead.any():
        return dead
    out = dead.copy()
    for sequence in ctx.sequences:
        local = dead[sequence]
        edges = np.diff(np.r_[False, local, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        grown = local.copy()
        for st, en in zip(starts, ends):
            grown[max(st - lead, 0):min(en + trail, len(local))] = True
        out[sequence] = grown
    return out


def smooth_labels(selected: np.ndarray, ctx: Context, smoothing: dict):
    """Stage 3 - apply the smoothing steps in order, then write D back on the dead frames.

    dead_lead_s / dead_trail_s (default 0): seconds by which every D run is grown before / after (see _extend_dead).
    dead_handling "gap": D frames are treated as X while smoothing (a possession can be merged across a stoppage).
    dead_handling "split": D frames cut the labels into blocks, each smoothed on its own.
    -> (smoothed, final)
    """
    dead = _extend_dead(selected == "D", ctx, smoothing.get("dead_lead_s", 0.0), smoothing.get("dead_trail_s", 0.0))
    labels = selected.copy()
    if smoothing.get("dead_handling", "gap") == "gap":
        labels[dead] = "X"
    for step in smoothing["steps"]:
        fn = SMOOTHERS[step["type"]]
        if step["type"] in INDEXED_SMOOTHERS:
            labels = _per_block(labels, ctx.sequences, lambda local, index, fn=fn, step=step: fn(local, step, ctx, index))
        else:
            labels = _per_block(labels, ctx.sequences, lambda local, index, fn=fn, step=step: fn(local, step, ctx))
    final = labels.copy()
    final[dead] = "D"
    return labels, final


# ======================================================================================================================
# The whole method, and the outputs
# ======================================================================================================================
@dataclass
class HybridResult:
    config: dict
    ctx: Context
    components: dict         # name -> label array (sorted order)
    selected: np.ndarray     # after the selection rules (D on dead frames)
    source: np.ndarray       # which rule decided each frame
    smoothed: np.ndarray     # after the smoothing steps, before D is written
    final: np.ndarray        # final label (sorted order)
    restored: np.ndarray     # final label in the ORIGINAL row order

    def components_table(self) -> pd.DataFrame:
        """One column per component, in the ORIGINAL row order of the input (the per-approach features)."""
        out = {}
        for name, labels in self.components.items():
            back = np.empty(len(labels), dtype="<U1")
            back[self.ctx.order] = labels
            out["hp_" + name] = back
        return pd.DataFrame(out)

    def table(self) -> pd.DataFrame:
        """One row per frame (sorted): every component, the deciding rule, and the label after each stage."""
        f = self.ctx.ordered
        cols = {"period_id": f["period_id"].to_numpy(), "video_frame": f["video_frame"].to_numpy()}
        cols.update({name: labels for name, labels in self.components.items()})
        cols.update(source=self.source, selected=self.selected, smoothed=self.smoothed, final=self.final)
        return pd.DataFrame(cols)

    def source_share(self) -> pd.Series:
        """Share of all frames decided by each rule."""
        return pd.Series(self.source).value_counts(normalize=True).round(4)


def _as_config(config: dict) -> dict:
    """Accept either the new nested config or the old flat parameter dict of the previous hybrid_possession.py."""
    return config if "components" in config else config_from_legacy_params(config)


def run(df: pd.DataFrame, config: dict) -> HybridResult:
    config = _as_config(config)
    ctx = build_context(df, config)
    components = compute_components(ctx, config)
    selected, source = select_labels(components, config["selection"])
    smoothed, final = smooth_labels(selected, ctx, config["smoothing"])
    restored = np.empty(len(final), dtype="<U1")
    restored[ctx.order] = final
    return HybridResult(config, ctx, components, selected, source, smoothed, final, restored)


def features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Kedro node: just the component features (hp_<name>, H/A/X, D for dead_ball), same row order as the input."""
    config = _as_config(config)
    ctx = build_context(df, config)
    result = HybridResult(config, ctx, compute_components(ctx, config), None, None, None, None, None)
    out = result.components_table()
    out.index = df.index
    return out


def add_hybrid_possession(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Kedro node: add the column `hybrid_possession` (H / A / D / X), keeping the input's rows and columns."""
    result = run(df, config)
    out = df.copy()
    out["hybrid_possession"] = result.restored
    log.info("Hybrid label counts %s; rules used %s",
             pd.Series(result.restored).value_counts().to_dict(), result.source_share().to_dict())
    return out

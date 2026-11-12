"""Hybrid possession, configurable: components -> selection rules -> smoothing steps -> one H / A / D / X label per frame.

The method has three parts, each driven by the config, so you can add, remove or re-configure approaches without
touching the code of the others:

    1. COMPONENTS   one feature (one label per frame) per approach:  hmm, geometry, clear_geometry, tags, dead_ball.
                    A component says H, A (or D for dead_ball) where it has an opinion and X where it abstains.
    2. SELECTION    an ordered list of rules: for every frame, the first rule whose component has an opinion (and whose
                    optional conditions hold) decides.  This is "priority" and "if-then-else" in one list.
    3. SMOOTHING    an ordered list of steps applied to the selected H/A labels: majority_vote, merge_short, ...

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


def component(name, requires=()):
    """Register a component.  `requires` lists columns that must exist; "{n}_s" means the player speed columns."""
    def wrap(fn):
        COMPONENTS[name] = (fn, tuple(requires))
        return fn
    return wrap


def smoother(name):
    def wrap(fn):
        SMOOTHERS[name] = fn
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
@component("dead_ball", requires=["ball_x", "ball_y", "sp_kind", "sp_since", "{n}_s"])
def dead_ball_component(ctx, params):
    """D where play is stopped, X elsewhere.  A frame is dead when

        the ball has been OUTSIDE the pitch for >= 1 s                                   (outside)
        OR  it is in the first N seconds of a set piece (N per kind) AND at least one of
              the ball hardly moved within +-1 s (still) / median player speed is low within +-1 s (slow) / ball not tracked

    then gaps shorter than dead_merge_gap_s are closed and stretches shorter than dead_min_duration_s are dropped.
    The intermediate signals are kept in ctx.signals (still, slow, outside, event_window, dead_raw).
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

    fps, (length, width) = ctx.fps, ctx.pitch_m
    seen, inside, bx, by = ctx.ball()
    n = len(ctx.ordered)
    speeds = ctx.numeric[ctx.speed_columns]
    medians = speeds.median(axis=1).where(speeds.notna().sum(axis=1) >= min_players)
    still, slow, outside = (np.zeros(n, dtype=bool) for _ in range(3))
    size = 2 * fps + 1                                                       # +-1 second
    for seq in ctx.sequences:
        x = pd.Series(np.where(seen[seq], bx[seq] * length, np.nan))
        y = pd.Series(np.where(seen[seq], by[seq] * width, np.nan))
        count = pd.Series(seen[seq].astype(int)).rolling(size, center=True, min_periods=1).sum()
        dx = x.rolling(size, center=True, min_periods=1).max() - x.rolling(size, center=True, min_periods=1).min()
        dy = y.rolling(size, center=True, min_periods=1).max() - y.rolling(size, center=True, min_periods=1).min()
        still[seq] = ((count >= 0.8 * size) & (dx < still_m) & (dy < still_m)).to_numpy()
        speed = medians.iloc[seq].reset_index(drop=True).rolling(size, center=True, min_periods=fps).median()
        slow[seq] = (speed < slow_mps).to_numpy()
        outside[seq] = _runs_at_least(seen[seq] & ~inside[seq], fps)

    since = ctx.numeric["sp_since"].to_numpy(dtype=float)
    event_window = np.zeros(n, dtype=bool)
    for kind, cutoff in cutoffs.items():
        matches = ctx.ordered["sp_kind"].eq(kind).fillna(False).to_numpy(dtype=bool)
        event_window |= matches & np.isfinite(since) & (since >= 0) & (since <= cutoff)
    dead_raw = outside | (event_window & (still | slow | ~seen))
    dead = _filter_dead_runs(dead_raw, ctx.sequences, int(round(min_s * fps)), int(round(gap_s * fps)))
    ctx.signals.update(still=still, slow=slow, outside=outside, event_window=event_window, dead_raw=dead_raw)
    return np.where(dead, "D", "X").astype("<U1")


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


def _per_block(labels, sequences, fn):
    """Apply fn to each block of frames between D frames (blocks never cross a sequence boundary)."""
    result = labels.copy()
    for sequence in sequences:
        start = 0
        for end in range(len(sequence) + 1):
            if end < len(sequence) and labels[sequence[end]] != "D":
                continue
            segment = sequence[start:end]
            if len(segment):
                result[segment] = fn(labels[segment])
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
    if config["smoothing"].get("dead_handling", "gap") not in ("gap", "split"):
        raise ValueError("smoothing.dead_handling must be 'gap' or 'split'")


def build_context(df: pd.DataFrame, config: dict) -> Context:
    """Check the input, sort it by (period, frame), split it into gap-free sequences and convert the numeric columns."""
    check_config(config)
    speed_columns = [c for c in df if re.fullmatch(r"\d+_s", c)]
    needed = set(BASE_COLUMNS)
    for name in config["components"]:
        needed.update(COMPONENTS[name][1])
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
                       if c in needed]
    if wants_speed:
        numeric_columns += speed_columns
    numeric = ordered[numeric_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Hybrid input numeric values must be finite when present")
    nonneg = [c for c in ("min_team_a_dist", "min_team_b_dist", *speed_columns) if c in numeric]
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


def smooth_labels(selected: np.ndarray, ctx: Context, smoothing: dict):
    """Stage 3 - apply the smoothing steps in order, then write D back on the dead frames.

    dead_handling "gap": D frames are treated as X while smoothing (a possession can be merged across a stoppage).
    dead_handling "split": D frames cut the labels into blocks, each smoothed on its own.
    -> (smoothed, final)
    """
    dead = selected == "D"
    labels = selected.copy()
    if smoothing.get("dead_handling", "gap") == "gap":
        labels[dead] = "X"
    for step in smoothing["steps"]:
        fn = SMOOTHERS[step["type"]]
        labels = _per_block(labels, ctx.sequences, lambda local, fn=fn, step=step: fn(local, step, ctx))
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


def run(df: pd.DataFrame, config: dict) -> HybridResult:
    ctx = build_context(df, config)
    components = compute_components(ctx, config)
    selected, source = select_labels(components, config["selection"])
    smoothed, final = smooth_labels(selected, ctx, config["smoothing"])
    restored = np.empty(len(final), dtype="<U1")
    restored[ctx.order] = final
    return HybridResult(config, ctx, components, selected, source, smoothed, final, restored)


def features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Kedro node: just the component features (hp_<name>, H/A/X, D for dead_ball), same row order as the input."""
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
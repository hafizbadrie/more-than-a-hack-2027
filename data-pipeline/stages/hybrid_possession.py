"""Combine geometry and tag possession rules with HMM fallback and dead-ball evidence."""
import logging
import re

import numpy as np
import pandas as pd

from stages.hmm_possession import _sequences

log = logging.getLogger(__name__)


def _runs_at_least(mask: np.ndarray, minimum: int) -> np.ndarray:
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
    """Smooth H/A votes and merge short possessions without changing D or X frames."""
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


def add_hybrid_possession(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Combine HMM, geometry, tags, and dead-ball evidence in a new output column."""
    required = [
        "period_id", "video_frame", "min_team_a_dist", "min_team_b_dist",
        "ball_x", "ball_y", "sp_kind", "sp_since", "home_attacking",
        "away_attacking", "home_defending", "away_defending", "hmm_possession",
    ]
    speed_columns = [name for name in df if re.fullmatch(r"\d+_s", name)]
    missing = [name for name in required if name not in df]
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

    fps = params["fps"]
    window = params["smoothing_half_window"]
    geometry_first = params.get("geometry_first", False)
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("Hybrid fps must be a positive integer")
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("Hybrid smoothing_half_window must be a non-negative integer")
    if not isinstance(geometry_first, bool):
        raise ValueError("Hybrid geometry_first must be a boolean")
    for key in ("clear_distance_m", "still_distance_m", "slow_speed_mps"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Hybrid {key} must be positive and finite")
    for key in ("dead_min_duration_s", "dead_merge_gap_s", "min_possession_s"):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"Hybrid {key} must be non-negative and finite")
    minimum_players = params["min_tracked_players"]
    if (isinstance(minimum_players, bool) or not isinstance(minimum_players, int)
            or not 1 <= minimum_players <= len(speed_columns)):
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

    order = np.lexsort((frames, df["period_id"].to_numpy()))
    ordered = df.iloc[order].copy()
    ordered["video_frame"] = frames[order]
    sequences = _sequences(ordered)
    numeric_columns = [
        "min_team_a_dist", "min_team_b_dist", "ball_x", "ball_y", "sp_since",
        "home_attacking", "away_attacking", "home_defending", "away_defending",
        *speed_columns,
    ]
    numeric = ordered[numeric_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Hybrid input numeric values must be finite when present")
    if (numeric[["min_team_a_dist", "min_team_b_dist", *speed_columns]] < 0).any().any():
        raise ValueError("Hybrid distances and player speeds must be non-negative")
    for column in (
        "home_attacking", "away_attacking", "home_defending", "away_defending"
    ):
        if numeric[column].isna().any() or not numeric[column].isin([0, 1]).all():
            raise ValueError(f"Hybrid {column} must contain only 0 or 1")
    hmm = ordered["hmm_possession"].to_numpy()
    if not np.isin(hmm, ["H", "A", "D", "X"]).all():
        raise ValueError("Hybrid hmm_possession must contain only H, A, D or X")

    ball_x = numeric["ball_x"].to_numpy(dtype=float)
    ball_y = numeric["ball_y"].to_numpy(dtype=float)
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)
    inside = seen & (ball_x >= 0) & (ball_x <= 1) & (ball_y >= 0) & (ball_y <= 1)
    speeds = numeric[speed_columns]
    medians = speeds.median(axis=1).where(speeds.notna().sum(axis=1) >= minimum_players)
    still = np.zeros(len(ordered), dtype=bool)
    slow = np.zeros(len(ordered), dtype=bool)
    outside = np.zeros(len(ordered), dtype=bool)
    size = 2 * fps + 1
    for seq in sequences:
        x = pd.Series(np.where(seen[seq], ball_x[seq] * pitch[0], np.nan))
        y = pd.Series(np.where(seen[seq], ball_y[seq] * pitch[1], np.nan))
        count = pd.Series(seen[seq].astype(int)).rolling(size, center=True, min_periods=1).sum()
        dx = x.rolling(size, center=True, min_periods=1).max() - x.rolling(size, center=True, min_periods=1).min()
        dy = y.rolling(size, center=True, min_periods=1).max() - y.rolling(size, center=True, min_periods=1).min()
        still[seq] = ((count >= 0.8 * size) & (dx < params["still_distance_m"])
                      & (dy < params["still_distance_m"])).to_numpy()
        speed = medians.iloc[seq].reset_index(drop=True).rolling(
            size, center=True, min_periods=fps
        ).median()
        slow[seq] = (speed < params["slow_speed_mps"]).to_numpy()
        outside[seq] = _runs_at_least(seen[seq] & ~inside[seq], fps)

    since = numeric["sp_since"].to_numpy(dtype=float)
    event_window = np.zeros(len(ordered), dtype=bool)
    for kind, cutoff in cutoffs.items():
        matches = ordered["sp_kind"].eq(kind).fillna(False).to_numpy(dtype=bool)
        event_window |= matches & np.isfinite(since) & (since >= 0) & (since <= cutoff)
    dead = outside | (event_window & (still | slow | ~seen))
    dead = _filter_dead_runs(
        dead,
        sequences,
        int(round(params["dead_min_duration_s"] * fps)),
        int(round(params["dead_merge_gap_s"] * fps)),
    )

    a = numeric["min_team_a_dist"].to_numpy(dtype=float)
    b = numeric["min_team_b_dist"].to_numpy(dtype=float)
    has_distances = inside & np.isfinite(a) & np.isfinite(b)
    geometry = np.where(a <= b, "H", "A")
    geometry_labels = np.full(len(ordered), "X", dtype="<U1")
    geometry_labels[has_distances] = geometry[has_distances]
    geometry_labels = _smooth_known_labels(geometry_labels, sequences, window)
    clear = has_distances & (np.minimum(a, b) <= params["clear_distance_m"])
    home_tag = (
        (numeric["home_attacking"].to_numpy(dtype=int) == 1)
        | (numeric["away_defending"].to_numpy(dtype=int) == 1)
    )
    away_tag = (
        (numeric["away_attacking"].to_numpy(dtype=int) == 1)
        | (numeric["home_defending"].to_numpy(dtype=int) == 1)
    )
    tag_known = home_tag ^ away_tag
    tags = np.where(home_tag, "H", "A")
    labels = np.full(len(ordered), "X", dtype="<U1")
    labels[np.isin(hmm, ["H", "A"])] = hmm[np.isin(hmm, ["H", "A"])]
    labels[has_distances] = geometry_labels[has_distances]
    tag_priority = ~has_distances if geometry_first else ~clear
    labels[tag_known & tag_priority] = tags[tag_known & tag_priority]
    labels[clear] = geometry_labels[clear]
    labels[dead] = "X"
    labels = _smooth_possession_labels(
        labels, sequences, window, params["min_possession_s"] * fps
    )
    labels[dead] = "D"

    result = df.copy()
    restored = np.empty(len(df), dtype="<U1")
    restored[order] = labels
    result["hybrid_possession"] = restored
    log.info("Hybrid label counts %s; dead-ball evidence frames %d",
             pd.Series(restored).value_counts().to_dict(), int(dead.sum()))
    return result

"""Geometry-first possession estimates with independent dead-ball evidence."""
import logging
import re

import numpy as np
import pandas as pd

from stages.hmm_possession import _sequences, _smooth_labels

log = logging.getLogger(__name__)


def _runs_at_least(mask: np.ndarray, minimum: int) -> np.ndarray:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = np.zeros(len(mask), dtype=bool)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= minimum:
            result[start:end] = True
    return result


def add_hybrid_possession(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Keep the original HMM labels and add independent geometry/dead-ball labels."""
    required = [
        "period_id", "video_frame", "min_team_a_dist", "min_team_b_dist",
        "ball_x", "ball_y", "sp_kind", "sp_since", "home_attacking",
        "away_attacking", "hmm_possession",
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
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("Hybrid fps must be a positive integer")
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("Hybrid smoothing_half_window must be a non-negative integer")
    for key in ("clear_distance_m", "still_distance_m", "slow_speed_mps"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Hybrid {key} must be positive and finite")
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
        "home_attacking", "away_attacking", *speed_columns,
    ]
    numeric = ordered[numeric_columns].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Hybrid input numeric values must be finite when present")
    if (numeric[["min_team_a_dist", "min_team_b_dist", *speed_columns]] < 0).any().any():
        raise ValueError("Hybrid distances and player speeds must be non-negative")
    for column in ("home_attacking", "away_attacking"):
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

    a = numeric["min_team_a_dist"].to_numpy(dtype=float)
    b = numeric["min_team_b_dist"].to_numpy(dtype=float)
    has_distances = inside & np.isfinite(a) & np.isfinite(b)
    geometry = np.where(a <= b, "H", "A")
    clear = has_distances & (np.minimum(a, b) <= params["clear_distance_m"])
    home_tag = numeric["home_attacking"].to_numpy(dtype=int) == 1
    away_tag = numeric["away_attacking"].to_numpy(dtype=int) == 1
    tag_known = home_tag ^ away_tag
    tags = np.where(home_tag, "H", "A")
    labels = np.full(len(ordered), "X", dtype="<U1")
    labels[np.isin(hmm, ["H", "A"])] = hmm[np.isin(hmm, ["H", "A"])]
    labels[has_distances] = geometry[has_distances]
    labels[tag_known & ~clear] = tags[tag_known & ~clear]
    labels[clear] = geometry[clear]
    labels[dead] = "X"
    labels = _smooth_labels(labels, sequences, window)
    labels[dead] = "D"

    result = df.copy()
    restored = np.empty(len(df), dtype="<U1")
    restored[order] = labels
    result["hybrid_possession"] = restored
    log.info("Hybrid label counts %s; dead-ball evidence frames %d",
             pd.Series(restored).value_counts().to_dict(), int(dead.sum()))
    return result

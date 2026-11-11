"""Fit and decode an unsupervised, weakly initialised possession HMM."""
import itertools
import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger(__name__)

FEATURES = (
    "min_team_a_dist",
    "min_team_b_dist",
    "nearest_player_dist",
    "possession_advantage",
    "team_a_players_1m",
    "team_a_players_2m",
    "team_a_players_5m",
    "team_b_players_1m",
    "team_b_players_2m",
    "team_b_players_5m",
    "ball_dx",
    "ball_dy",
    "ball_displacement",
    "ball_speed",
    "ball_acceleration",
    "ball_direction_deg",
    "ball_direction_change_deg",
    "ball_direction_change_abs_deg",
    "team_a_nearest_player_ball_relative_velocity",
    "team_b_nearest_player_ball_relative_velocity",
    "nearest_player_ball_relative_velocity",
    "ball_velocity_x",
    "ball_velocity_toward_team_a_goal",
    "ball_velocity_toward_team_b_goal",
    "ball_moving_toward_team_a_goal",
    "ball_moving_toward_team_b_goal",
    "tag_att_signed",
    "tag_att_trans_signed",
    "tag_def_signed",
    "tag_n_active",
    "tag_box_players",
    "sp_team",
    "sp_since",
    "evt_t_to_restart",
    "home_attacking",
    "home_defending",
    "away_attacking",
    "away_defending",
    "team_centroid_x_all_players",
    "team_centroid_y_all_players",
    "team_centroid_x_barcelona",
    "team_centroid_y_barcelona",
    "team_centroid_x_atletic_club",
    "team_centroid_y_atletic_club",
    "team_dispersion_barcelona",
    "team_dispersion_atletic_club",
    "team_centroid_velocity_x_all_players",
    "team_centroid_velocity_y_all_players",
    "team_centroid_speed_all_players",
    "team_centroid_velocity_x_barcelona",
    "team_centroid_velocity_y_barcelona",
    "team_centroid_speed_barcelona",
    "team_centroid_velocity_x_atletic_club",
    "team_centroid_velocity_y_atletic_club",
    "team_centroid_speed_atletic_club",
    "team_average_radial_movement_barcelona",
    "team_average_radial_movement_atletic_club",
    "team_width_barcelona",
    "team_depth_barcelona",
    "team_tracked_players_barcelona",
    "team_width_atletic_club",
    "team_depth_atletic_club",
    "team_tracked_players_atletic_club",
    "team_movement_coherence_barcelona",
    "team_movement_coherence_players_barcelona",
    "team_movement_coherence_atletic_club",
    "team_movement_coherence_players_atletic_club",
    "opponent_team_centroid_distance",
    "opponent_team_closing_speed",
    "team_ball_longitudinal_distance_barcelona",
    "team_ball_lateral_distance_barcelona",
    "team_ball_longitudinal_distance_atletic_club",
    "team_ball_lateral_distance_atletic_club",
)
LABELS = np.array(["H", "A", "D"])


def _sequences(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return positional indices for consecutive frames, without crossing periods."""
    periods = frame["period_id"].to_numpy()
    frames = frame["video_frame"].to_numpy()
    boundaries = np.flatnonzero(
        (periods[1:] != periods[:-1]) | (frames[1:] != frames[:-1] + 1)
    ) + 1
    return [part for part in np.split(np.arange(len(frame)), boundaries) if len(part)]


def _viterbi(model: GaussianHMM, observations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Decode a sequence, marginalising features missing from each observation."""
    emissions = np.zeros((len(observations), model.n_components))
    observed = np.asarray(valid, dtype=bool)
    if observed.ndim == 1:
        observed = np.broadcast_to(observed[:, None], observations.shape)
    for i, mask in enumerate(observed):
        if mask.any():
            residual = observations[i, mask] - model.means_[:, mask]
            variances = model._covars_[:, mask]
            emissions[i] = -0.5 * (
                mask.sum() * np.log(2 * np.pi)
                + np.log(variances).sum(axis=1)
                + (residual**2 / variances).sum(axis=1)
            )
    with np.errstate(divide="ignore"):
        start = np.log(model.startprob_)
        transitions = np.log(model.transmat_)
    scores = start + emissions[0]
    back = np.zeros((len(observations), model.n_components), dtype=int)
    for i in range(1, len(observations)):
        candidates = scores[:, None] + transitions
        back[i] = np.argmax(candidates, axis=0)
        scores = candidates[back[i], np.arange(model.n_components)] + emissions[i]
    states = np.empty(len(observations), dtype=int)
    states[-1] = int(np.argmax(scores))
    for i in range(len(states) - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def add_hmm_possession(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add H/A/D estimates and X abstentions, preserving input order and columns.

    EM is fitted on contiguous runs after mean imputation of missing features.
    Missing features are marginalised during decoding, and frames with no observed
    model features are reported as X rather than invented labels.
    """
    required = ["period_id", "video_frame", "sp_kind", *FEATURES]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"HMM input is missing required columns: {missing}")
    if df.empty:
        raise ValueError("HMM requires at least one frame")
    if df[["period_id", "video_frame"]].isna().any().any():
        raise ValueError("HMM requires non-missing period_id and video_frame")
    if df.duplicated(["period_id", "video_frame"]).any():
        raise ValueError("HMM requires unique period_id/video_frame pairs")

    self_transition = params["self_transition"]
    if not 0 < self_transition < 1:
        raise ValueError("self_transition must be between 0 and 1")
    if params["n_iter"] < 1 or params["tol"] <= 0 or params["min_covar"] <= 0:
        raise ValueError("n_iter, tol and min_covar must be positive")
    if params["min_samples"] < 3 or params["min_anchor_samples"] < 1:
        raise ValueError("min_samples must be at least 3 and min_anchor_samples positive")
    video_frames = pd.to_numeric(df["video_frame"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(video_frames).all() or (video_frames != np.floor(video_frames)).any():
        raise ValueError("HMM requires finite integer video_frame values")
    order = np.lexsort((video_frames, df["period_id"].to_numpy()))
    ordered = df.iloc[order].copy()
    ordered["video_frame"] = video_frames[order]
    numeric = ordered.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("HMM features must be finite when present")
    observed = ~np.isnan(numeric)
    active = observed.any(axis=0)
    if not active.all():
        numeric = numeric[:, active]
        observed = observed[:, active]
        features = np.asarray(FEATURES)[active]
    else:
        features = np.asarray(FEATURES)
    valid = observed.any(axis=1)
    if valid.sum() < params["min_samples"]:
        raise ValueError(f"HMM needs at least {params['min_samples']} frames with observed features")

    transformed = np.sign(numeric[observed]) * np.log1p(np.abs(numeric[observed]))
    transformed_values = np.full_like(numeric, np.nan)
    transformed_values[observed] = transformed
    centre = np.nanmean(transformed_values, axis=0)
    scale = np.nanstd(transformed_values, axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.where(observed, (transformed_values - centre) / scale, 0.0)

    # Weak anchors are used only for initialisation, not as ground-truth labels.
    feature_indices = {name: i for i, name in enumerate(features)}
    missing_distances = [
        name for name in ("min_team_a_dist", "min_team_b_dist")
        if name not in feature_indices
    ]
    if missing_distances:
        raise ValueError(f"HMM has no observations for anchor features: {missing_distances}")
    distance_a = feature_indices["min_team_a_dist"]
    distance_b = feature_indices["min_team_b_dist"]
    speed_indices = [
        feature_indices[name]
        for name in ("team_centroid_speed_barcelona", "team_centroid_speed_atletic_club")
        if name in feature_indices
    ]
    if not speed_indices:
        raise ValueError("HMM has no observations for team centroid speed anchors")
    has_distances = observed[:, distance_a] & observed[:, distance_b]
    advantage = numeric[:, distance_b] - numeric[:, distance_a]
    has_set_piece = ordered["sp_kind"].notna().to_numpy()
    has_speed = observed[:, speed_indices].any(axis=1)
    motion = np.full(len(ordered), np.nan)
    speed_values = np.where(
        observed[:, speed_indices], numeric[:, speed_indices], 0.0
    )
    motion[has_speed] = (
        speed_values.sum(axis=1)[has_speed]
        / observed[:, speed_indices].sum(axis=1)[has_speed]
    )
    slow = np.zeros(len(ordered), dtype=bool)
    if (~np.isnan(motion)).any():
        slow[~np.isnan(motion)] = motion[~np.isnan(motion)] <= np.nanquantile(motion, 0.4)
    non_set_piece_advantage = advantage[has_distances & ~has_set_piece]
    if not len(non_set_piece_advantage):
        raise ValueError("Not enough HMM initialization examples for H/A/D: no distance anchors")
    anchors = [
        has_distances & (advantage >= np.quantile(non_set_piece_advantage, 0.7)) & ~has_set_piece,
        has_distances & (advantage <= np.quantile(non_set_piece_advantage, 0.3)) & ~has_set_piece,
        has_set_piece & has_speed & slow,
    ]
    if any(mask.sum() < params["min_anchor_samples"] for mask in anchors):
        counts = [int(mask.sum()) for mask in anchors]
        raise ValueError(f"Not enough HMM initialization examples for H/A/D: {counts}")
    seed_means = np.vstack([np.median(x[mask], axis=0) for mask in anchors])

    sequences = _sequences(ordered)
    training_runs = []
    for sequence in sequences:
        observed_indices = sequence[valid[sequence]]
        if len(observed_indices):
            breaks = np.flatnonzero(np.diff(observed_indices) != 1) + 1
            training_runs.extend(np.split(observed_indices, breaks))
    lengths = [len(run) for run in training_runs]
    train = np.concatenate([x[run] for run in training_runs])

    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=params["n_iter"],
        tol=params["tol"],
        min_covar=params["min_covar"],
        init_params="",
        random_state=params["random_state"],
    )
    model.startprob_ = np.full(3, 1 / 3)
    model.transmat_ = np.full((3, 3), (1 - self_transition) / 2)
    np.fill_diagonal(model.transmat_, self_transition)
    model.means_ = seed_means.copy()
    model.covars_ = np.tile(np.maximum(train.var(axis=0), params["min_covar"]), (3, 1))
    model.fit(train, lengths)
    if not model.monitor_.converged or model.monitor_.iter == model.n_iter:
        log.warning("HMM reached its iteration limit; estimates may not have converged")
    if not np.isfinite(model.means_).all() or not np.isfinite(model.transmat_).all():
        raise ValueError("HMM training produced non-finite parameters")

    # EM can permute or move states; map them back to the weak physical anchors.
    costs = ((model.means_[:, None, :] - seed_means[None, :, :]) ** 2).sum(axis=2)
    assignment = min(itertools.permutations(range(3)), key=lambda p: sum(costs[i, p[i]] for i in range(3)))
    state_labels = LABELS[list(assignment)]
    labels = np.full(len(ordered), "X", dtype="<U1")
    for sequence in sequences:
        states = _viterbi(model, x[sequence], observed[sequence])
        labels[sequence[valid[sequence]]] = state_labels[states[valid[sequence]]]
    result = df.copy()
    restored = np.empty(len(df), dtype="<U1")
    restored[order] = labels
    result["hmm_possession"] = restored
    log.info("HMM training: %d observed frames, %d runs, anchor counts %s, labels %s",
             int(valid.sum()), len(lengths), [int(mask.sum()) for mask in anchors],
             pd.Series(restored).value_counts().to_dict())
    return result

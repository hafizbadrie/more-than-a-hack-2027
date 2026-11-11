"""Fit and decode an unsupervised, weakly initialised possession HMM."""
import itertools
import logging
from pathlib import Path

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
    # "ball_speed",
    # "ball_acceleration",
    # "ball_direction_deg",
    # "ball_direction_change_deg",
    # "ball_direction_change_abs_deg",
    # "team_a_nearest_player_ball_relative_velocity",
    # "team_b_nearest_player_ball_relative_velocity",
    # "nearest_player_ball_relative_velocity",
    # "ball_velocity_x",
    # "ball_velocity_toward_team_a_goal",
    # "ball_velocity_toward_team_b_goal",
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
    # "team_centroid_velocity_x_all_players",
    # "team_centroid_velocity_y_all_players",
    # "team_centroid_speed_all_players",
    # "team_centroid_velocity_x_barcelona",
    # "team_centroid_velocity_y_barcelona",
    # "team_centroid_speed_barcelona",
    # "team_centroid_velocity_x_atletic_club",
    # "team_centroid_velocity_y_atletic_club",
    # "team_centroid_speed_atletic_club",
    "team_average_radial_movement_barcelona",
    "team_average_radial_movement_atletic_club",
    "team_width_barcelona",
    "team_depth_barcelona",
    "team_tracked_players_barcelona",
    "team_width_atletic_club",
    "team_depth_atletic_club",
    "team_tracked_players_atletic_club",
    # "team_movement_coherence_barcelona",
    # "team_movement_coherence_players_barcelona",
    # "team_movement_coherence_atletic_club",
    # "team_movement_coherence_players_atletic_club",
    "opponent_team_centroid_distance",
    # "opponent_team_closing_speed",
    "team_ball_longitudinal_distance_barcelona",
    "team_ball_lateral_distance_barcelona",
    "team_ball_longitudinal_distance_atletic_club",
    "team_ball_lateral_distance_atletic_club",
)
LABELS = np.array(["H", "A", "D"])
DISTANCE_FEATURES = ("min_team_a_dist", "min_team_b_dist")


def _sequences(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return positional indices for consecutive frames, without crossing periods."""
    periods = frame["period_id"].to_numpy()
    frames = frame["video_frame"].to_numpy()
    boundaries = np.flatnonzero(
        (periods[1:] != periods[:-1]) | (frames[1:] != frames[:-1] + 1)
    ) + 1
    return [part for part in np.split(np.arange(len(frame)), boundaries) if len(part)]


def _smooth_labels(labels: np.ndarray, sequences: list[np.ndarray], window: int) -> np.ndarray:
    """Apply a centred majority vote without crossing gaps, periods, or abstentions."""
    result = labels.copy()
    if window == 0:
        return result
    for sequence in sequences:
        available = sequence[labels[sequence] != "X"]
        if not len(available):
            continue
        breaks = np.flatnonzero(np.diff(available) != 1) + 1
        for run in np.split(available, breaks):
            original = labels[run]
            indices = np.arange(len(run))
            lo = np.maximum(0, indices - window)
            hi = np.minimum(len(run), indices + window + 1)
            votes = []
            for label in LABELS:
                cumulative = np.r_[0, np.cumsum(original == label)]
                votes.append(cumulative[hi] - cumulative[lo])
            counts = np.stack(votes, axis=1)
            maximum = counts.max(axis=1)
            winners = counts == maximum[:, None]
            unique = winners.sum(axis=1) == 1
            result[run[unique]] = LABELS[np.argmax(counts[unique], axis=1)]
    return result


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


def feature_importance(
    model: GaussianHMM,
    features: np.ndarray,
    state_labels: np.ndarray,
    observed: np.ndarray,
) -> pd.DataFrame:
    """Rank features by how well they separate the HMM states.

    For each pair of states, the symmetric KL divergence between the diagonal
    Gaussian emissions is computed per feature (features are standardised, so
    values are comparable). Importance is the mean over state pairs; the
    ``importance_pct`` column normalises it to sum to 100.
    """
    means = model.means_
    variances = model._covars_
    columns = {}
    for i, j in itertools.combinations(range(len(means)), 2):
        diff2 = (means[i] - means[j]) ** 2
        kl = 0.5 * (
            (variances[i] + diff2) / variances[j]
            + (variances[j] + diff2) / variances[i]
            - 2
        )
        columns[f"sep_{state_labels[i]}_{state_labels[j]}"] = kl
    report = pd.DataFrame(columns)
    report.insert(0, "feature", features)
    report["importance"] = report[list(columns)].mean(axis=1)
    total = report["importance"].sum()
    report["importance_pct"] = 100 * report["importance"] / total if total > 0 else 0.0
    report["observed_fraction"] = observed.mean(axis=0)
    report = report.sort_values("importance", ascending=False, ignore_index=True)
    report.insert(1, "rank", report.index + 1)
    return report


def _convergence_report(model: GaussianHMM) -> tuple[pd.DataFrame, str]:
    """Describe the recorded EM likelihoods without relying on monitor.converged."""
    history = np.asarray(model.monitor_.history, dtype=float)
    if not len(history) or not np.isfinite(history).all():
        raise ValueError("HMM training produced missing or non-finite likelihood history")
    deltas = np.concatenate(([np.nan], np.diff(history)))
    report = pd.DataFrame({
        "iteration": np.arange(1, len(history) + 1),
        "log_likelihood": history,
        "delta": deltas,
    })
    if len(history) >= 2 and deltas[-1] < -(np.finfo(float).eps ** 0.5):
        status = "likelihood_decreased"
    elif len(history) >= 2 and 0 <= deltas[-1] < model.tol:
        status = "converged"
    elif model.monitor_.iter >= model.n_iter:
        status = "iteration_limit"
    else:
        status = "not_converged"
    report["status"] = status
    report["tolerance"] = model.tol
    report["iteration_limit"] = model.n_iter
    return report, status


def add_hmm_possession(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add H/A/D estimates and X abstentions, preserving input order and columns.

    EM is fitted on contiguous runs after mean imputation of missing features.
    Missing features are marginalised during decoding, and frames with no observed
    model features are reported as X rather than invented labels.
    """
    selected = params.get("features", FEATURES)
    if not isinstance(selected, (tuple, list)) or not selected:
        raise ValueError("features must be a non-empty list or tuple of feature names")
    if any(not isinstance(name, str) or name not in FEATURES for name in selected):
        raise ValueError("features must contain only known HMM feature names")
    if len(set(selected)) != len(selected) or not set(DISTANCE_FEATURES).issubset(selected):
        raise ValueError("features must be unique and include both team distance features")
    coverage = params.get("min_feature_coverage", 0.0)
    if not np.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("min_feature_coverage must be between 0 and 1")
    missing_strategy = params.get("training_missing", "mean")
    if missing_strategy not in ("mean", "complete"):
        raise ValueError("training_missing must be 'mean' or 'complete'")
    mapping = params.get("state_mapping", "anchors")
    if mapping not in ("anchors", "distance"):
        raise ValueError("state_mapping must be 'anchors' or 'distance'")
    window = params.get("smoothing_half_window", 0)
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("smoothing_half_window must be a non-negative integer")
    freeze_transitions = params.get("freeze_transitions", False)
    if not isinstance(freeze_transitions, bool):
        raise ValueError("freeze_transitions must be a boolean")
    stationary_threshold = params.get("dead_anchor_max_displacement")
    if stationary_threshold is not None and (
        not np.isfinite(stationary_threshold) or stationary_threshold < 0
    ):
        raise ValueError("dead_anchor_max_displacement must be non-negative and finite")
    max_since = params.get("dead_anchor_max_since")
    if max_since is not None and (not np.isfinite(max_since) or max_since < 0):
        raise ValueError("dead_anchor_max_since must be non-negative and finite")
    required = ["period_id", "video_frame", "sp_kind", *selected]
    if stationary_threshold is not None and "ball_displacement" not in required:
        required.append("ball_displacement")
    if max_since is not None and "sp_since" not in required:
        required.append("sp_since")
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
    numeric = ordered.loc[:, selected].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("HMM features must be finite when present")
    observed = ~np.isnan(numeric)
    active = observed.any(axis=0) & (observed.mean(axis=0) >= coverage)
    removed = np.asarray(selected)[~active]
    if len(removed):
        log.info("HMM excluded features below coverage threshold: %s", removed.tolist())
    if not active.all():
        numeric = numeric[:, active]
        observed = observed[:, active]
        features = np.asarray(selected)[active]
    else:
        features = np.asarray(selected)
    missing_distances = [name for name in DISTANCE_FEATURES if name not in features]
    if missing_distances:
        raise ValueError(f"HMM has no observations for anchor features: {missing_distances}")
    valid = observed.any(axis=1)
    training_valid = observed.all(axis=1) if missing_strategy == "complete" else valid
    if training_valid.sum() < params["min_samples"]:
        raise ValueError(
            f"HMM needs at least {params['min_samples']} training frames with "
            f"training_missing={missing_strategy!r}; found {int(training_valid.sum())}"
        )

    transformed = np.sign(numeric[observed]) * np.log1p(np.abs(numeric[observed]))
    transformed_values = np.full_like(numeric, np.nan)
    transformed_values[observed] = transformed
    centre = np.nanmean(transformed_values[training_valid], axis=0)
    scale = np.nanstd(transformed_values[training_valid], axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.where(observed, (transformed_values - centre) / scale, 0.0)

    # Weak anchors are used only for initialisation, not as ground-truth labels.
    feature_indices = {name: i for i, name in enumerate(features)}
    distance_a = feature_indices["min_team_a_dist"]
    distance_b = feature_indices["min_team_b_dist"]
    has_distances = observed[:, distance_a] & observed[:, distance_b]
    advantage = numeric[:, distance_b] - numeric[:, distance_a]
    has_set_piece = ordered["sp_kind"].notna().to_numpy()
    dead_anchor = has_set_piece & training_valid
    if stationary_threshold is not None:
        displacement = pd.to_numeric(ordered["ball_displacement"], errors="raise").to_numpy(dtype=float)
        if np.isinf(displacement).any() or (displacement < 0).any():
            raise ValueError("ball_displacement must be non-negative and finite when present")
        dead_anchor &= np.isfinite(displacement) & (displacement <= stationary_threshold)
    if max_since is not None:
        since = pd.to_numeric(ordered["sp_since"], errors="raise").to_numpy(dtype=float)
        if np.isinf(since).any():
            raise ValueError("sp_since must be finite when present")
        dead_anchor &= np.isfinite(since) & (since >= 0) & (since <= max_since)
    non_set_piece_advantage = advantage[has_distances & ~has_set_piece & training_valid]
    if not len(non_set_piece_advantage):
        raise ValueError("Not enough HMM initialization examples for H/A/D: no distance anchors")
    anchors = [
        has_distances & (advantage >= np.quantile(non_set_piece_advantage, 0.7)) & ~has_set_piece & training_valid,
        has_distances & (advantage <= np.quantile(non_set_piece_advantage, 0.3)) & ~has_set_piece & training_valid,
        dead_anchor,
    ]
    if any(mask.sum() < params["min_anchor_samples"] for mask in anchors):
        counts = [int(mask.sum()) for mask in anchors]
        raise ValueError(f"Not enough HMM initialization examples for H/A/D: {counts}")
    seed_means = np.vstack([np.median(x[mask], axis=0) for mask in anchors])

    sequences = _sequences(ordered)
    training_runs = []
    for sequence in sequences:
        observed_indices = sequence[training_valid[sequence]]
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
        params="smc" if freeze_transitions else "stmc",
    )
    model.startprob_ = np.full(3, 1 / 3)
    model.transmat_ = np.full((3, 3), (1 - self_transition) / 2)
    np.fill_diagonal(model.transmat_, self_transition)
    model.means_ = seed_means.copy()
    model.covars_ = np.tile(np.maximum(train.var(axis=0), params["min_covar"]), (3, 1))
    model.fit(train, lengths)
    convergence, status = _convergence_report(model)
    if status != "converged":
        log.warning(
            "HMM training status: %s after %d iterations; final likelihood %.6f, delta %s",
            status, len(convergence), convergence["log_likelihood"].iloc[-1],
            convergence["delta"].iloc[-1],
        )
    else:
        log.info(
            "HMM converged after %d iterations; final likelihood %.6f, delta %.6f",
            len(convergence), convergence["log_likelihood"].iloc[-1],
            convergence["delta"].iloc[-1],
        )
    if not np.isfinite(model.means_).all() or not np.isfinite(model.transmat_).all():
        raise ValueError("HMM training produced non-finite parameters")

    # EM can permute or move states; map them back to the weak physical anchors.
    costs = ((model.means_[:, None, :] - seed_means[None, :, :]) ** 2).sum(axis=2)
    if mapping == "anchors":
        assignment = min(itertools.permutations(range(3)), key=lambda p: sum(costs[i, p[i]] for i in range(3)))
        state_labels = LABELS[list(assignment)]
    else:
        dead_state = int(np.argmin(costs[:, 2]))
        others = [i for i in range(3) if i != dead_state]
        means = model.means_ * scale + centre
        advantage_means = means[:, distance_b] - means[:, distance_a]
        home, away = sorted(others, key=lambda i: advantage_means[i], reverse=True)
        state_labels = np.empty(3, dtype="<U1")
        state_labels[home], state_labels[away], state_labels[dead_state] = "H", "A", "D"
    means = model.means_ * scale + centre
    states_report = pd.DataFrame({
        "state": np.arange(3),
        "label": state_labels,
        "mean_log_distance_advantage": means[:, distance_b] - means[:, distance_a],
        "distance_to_home_anchor": costs[:, 0],
        "distance_to_away_anchor": costs[:, 1],
        "distance_to_dead_anchor": costs[:, 2],
    })
    for label in LABELS:
        target = int(np.flatnonzero(state_labels == label)[0])
        states_report[f"transition_to_{label}"] = model.transmat_[:, target]
    labels = np.full(len(ordered), "X", dtype="<U1")
    for sequence in sequences:
        states = _viterbi(model, x[sequence], observed[sequence])
        labels[sequence[valid[sequence]]] = state_labels[states[valid[sequence]]]
    labels = _smooth_labels(labels, sequences, window)
    result = df.copy()
    restored = np.empty(len(df), dtype="<U1")
    restored[order] = labels
    result["hmm_possession"] = restored
    result.attrs["hmm_states"] = states_report
    states_path = params.get("state_diagnostics_path")
    if states_path:
        Path(states_path).parent.mkdir(parents=True, exist_ok=True)
        states_report.to_csv(states_path, index=False)
        log.info("HMM state diagnostics written to %s", states_path)
    result.attrs["hmm_convergence"] = convergence
    result.attrs["hmm_convergence_status"] = status
    convergence_path = params.get("convergence_path")
    if convergence_path:
        Path(convergence_path).parent.mkdir(parents=True, exist_ok=True)
        convergence.to_csv(convergence_path, index=False)
        log.info("HMM convergence history written to %s", convergence_path)
    importance = feature_importance(model, features, state_labels, observed)
    result.attrs["hmm_feature_importance"] = importance
    report_path = params.get("feature_importance_path")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(report_path, index=False)
        log.info("HMM feature importance report written to %s", report_path)
    log.info("HMM training: %d training frames, %d runs, anchor counts %s, labels %s",
             int(training_valid.sum()), len(lengths), [int(mask.sum()) for mask in anchors],
             pd.Series(restored).value_counts().to_dict())
    return result

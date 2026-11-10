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
    "team_centroid_speed_barcelona",
    "team_centroid_speed_atletic_club",
)
LABELS = np.array(["H", "A", "D"])


def _sequences(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return positional indices for consecutive frames, without crossing periods."""
    periods = frame["period_id"].to_numpy()
    frames = frame["frame_id"].to_numpy()
    boundaries = np.flatnonzero(
        (periods[1:] != periods[:-1]) | (frames[1:] != frames[:-1] + 1)
    ) + 1
    return [part for part in np.split(np.arange(len(frame)), boundaries) if len(part)]


def _viterbi(model: GaussianHMM, observations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Decode a sequence, marginalising observations missing any model feature."""
    emissions = np.zeros((len(observations), model.n_components))
    if valid.any():
        emissions[valid] = model._compute_log_likelihood(observations[valid])
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

    EM is fitted on fully observed contiguous runs. Missing observations have neutral
    emissions during decoding and are reported as X rather than invented labels.
    """
    required = ["period_id", "frame_id", "sp_kind", *FEATURES]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"HMM input is missing required columns: {missing}")
    if df.empty:
        raise ValueError("HMM requires at least one frame")
    if df[["period_id", "frame_id"]].isna().any().any():
        raise ValueError("HMM requires non-missing period_id and frame_id")
    if df.duplicated(["period_id", "frame_id"]).any():
        raise ValueError("HMM requires unique period_id/frame_id pairs")

    self_transition = params["self_transition"]
    if not 0 < self_transition < 1:
        raise ValueError("self_transition must be between 0 and 1")
    if params["n_iter"] < 1 or params["tol"] <= 0 or params["min_covar"] <= 0:
        raise ValueError("n_iter, tol and min_covar must be positive")
    if params["min_samples"] < 3 or params["min_anchor_samples"] < 1:
        raise ValueError("min_samples must be at least 3 and min_anchor_samples positive")
    frame_ids = pd.to_numeric(df["frame_id"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(frame_ids).all() or (frame_ids != np.floor(frame_ids)).any():
        raise ValueError("HMM requires finite integer frame_id values")
    order = np.lexsort((frame_ids, df["period_id"].to_numpy()))
    ordered = df.iloc[order].copy()
    ordered["frame_id"] = frame_ids[order]
    numeric = ordered.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(numeric).any() or (numeric < 0).any():
        raise ValueError("HMM distances and speeds must be non-negative and finite when present")
    valid = ~np.isnan(numeric).any(axis=1)
    if valid.sum() < params["min_samples"]:
        raise ValueError(f"HMM needs at least {params['min_samples']} fully observed frames")

    transformed = np.log1p(numeric[valid])
    centre = transformed.mean(axis=0)
    scale = transformed.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.zeros_like(numeric)
    x[valid] = (transformed - centre) / scale

    # Weak anchors are used only for initialisation, not as ground-truth labels.
    valid_rows = ordered.iloc[np.flatnonzero(valid)]
    advantage = numeric[valid, 1] - numeric[valid, 0]
    has_set_piece = valid_rows["sp_kind"].notna().to_numpy()
    motion = numeric[valid, 2:].mean(axis=1)
    slow = motion <= np.quantile(motion, 0.4)
    anchors = [
        (advantage >= np.quantile(advantage, 0.7)) & ~has_set_piece,
        (advantage <= np.quantile(advantage, 0.3)) & ~has_set_piece,
        has_set_piece & slow,
    ]
    if any(mask.sum() < params["min_anchor_samples"] for mask in anchors):
        counts = [int(mask.sum()) for mask in anchors]
        raise ValueError(f"Not enough HMM initialization examples for H/A/D: {counts}")
    seed_means = np.vstack([np.median(x[valid][mask], axis=0) for mask in anchors])

    sequences = _sequences(ordered)
    training_runs = []
    for sequence in sequences:
        observed = sequence[valid[sequence]]
        if len(observed):
            breaks = np.flatnonzero(np.diff(observed) != 1) + 1
            training_runs.extend(np.split(observed, breaks))
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
        states = _viterbi(model, x[sequence], valid[sequence])
        labels[sequence[valid[sequence]]] = state_labels[states[valid[sequence]]]
    result = df.copy()
    restored = np.empty(len(df), dtype="<U1")
    restored[order] = labels
    result["hmm_possession"] = restored
    log.info("HMM training: %d observed frames, %d runs, anchor counts %s, labels %s",
             int(valid.sum()), len(lengths), [int(mask.sum()) for mask in anchors],
             pd.Series(restored).value_counts().to_dict())
    return result

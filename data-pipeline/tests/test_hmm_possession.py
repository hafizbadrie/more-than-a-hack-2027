import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.hmm_possession import (
    FEATURES, _convergence_report, _sequences, _smooth_labels, _viterbi, add_hmm_possession,
)
from compare_hmm_runs import compare

PARAMS = {
    "self_transition": 0.99,
    "n_iter": 40,
    "tol": 0.01,
    "min_covar": 0.01,
    "random_state": 0,
    "min_samples": 100,
    "min_anchor_samples": 10,
}


def sample():
    rng = np.random.default_rng(42)
    centres = np.repeat([[1, 8], [8, 1], [5, 6]], 100, axis=0)
    values = np.maximum(centres + rng.normal(0, 0.15, centres.shape), 0.01)
    df = pd.DataFrame(values, columns=["min_team_a_dist", "min_team_b_dist"])
    df["period_id"] = 1
    df["video_frame"] = np.arange(len(df))
    df["sp_kind"] = [None] * 200 + ["free_kick"] * 100
    for feature in FEATURES:
        if feature not in df:
            df[feature] = 0.0
    df["possession_advantage"] = df["min_team_b_dist"] - df["min_team_a_dist"]
    df.loc[45:49, list(FEATURES)] = np.nan
    df.loc[40, "ball_dx"] = np.nan
    return df


class HMMTests(unittest.TestCase):
    def test_convergence_reports_distinguish_decrease_limit_and_tolerance(self):
        model = GaussianHMM(n_components=3, tol=0.01, n_iter=3)
        for history, expected in [
            ([-10.0, -9.0, -9.2], "likelihood_decreased"),
            ([-10.0, -9.0, -8.0], "iteration_limit"),
            ([-10.0, -9.0, -8.995], "converged"),
            ([-10.0], "not_converged"),
        ]:
            model.monitor_ = SimpleNamespace(history=history, iter=len(history))
            report, status = _convergence_report(model)
            self.assertEqual(status, expected)
            np.testing.assert_array_equal(report["iteration"], np.arange(1, len(history) + 1))
            np.testing.assert_allclose(report["log_likelihood"], history)
            self.assertTrue(np.isnan(report["delta"].iloc[0]))
        model.monitor_ = SimpleNamespace(history=[np.nan], iter=1)
        with self.assertRaisesRegex(ValueError, "likelihood history"):
            _convergence_report(model)

    def test_viterbi_matches_library_and_marginalizes_missing_frames(self):
        model = GaussianHMM(n_components=2, covariance_type="diag", init_params="")
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])
        model.means_ = np.array([[0.0], [5.0]])
        model.covars_ = np.array([[1.0], [1.0]])
        x = np.array([[0.0], [1.0], [5.0], [5.0]])
        np.testing.assert_array_equal(_viterbi(model, x, np.ones(4, dtype=bool)), model.predict(x))
        invalid = np.array([True, False, False, True])
        changed = x.copy()
        changed[~invalid] = 1000
        np.testing.assert_array_equal(_viterbi(model, x, invalid), _viterbi(model, changed, invalid))

    def test_sequence_boundaries(self):
        df = pd.DataFrame({"period_id": [1, 1, 1, 2, 2], "video_frame": [1, 2, 5, 6, 7]})
        self.assertEqual([x.tolist() for x in _sequences(df)], [[0, 1], [2], [3, 4]])

    def test_smoothing_respects_gaps_abstentions_and_ties(self):
        labels = np.array(["H", "A", "H", "X", "A", "H", "A", "H", "A"])
        sequences = [np.arange(7), np.arange(7, 9)]
        actual = _smooth_labels(labels, sequences, 1)
        np.testing.assert_array_equal(actual, ["H", "H", "H", "X", "A", "A", "A", "H", "A"])

    def test_experimental_features_and_complete_case_training(self):
        df = sample()
        df["nearest_player_dist"] = df["min_team_a_dist"]
        df.loc[:240, "nearest_player_dist"] = np.nan
        params = {
            **PARAMS,
            "features": ["min_team_a_dist", "min_team_b_dist", "nearest_player_dist"],
            "min_feature_coverage": 0.5,
            "training_missing": "complete",
            "state_mapping": "distance",
            "freeze_transitions": True,
            "smoothing_half_window": 2,
        }
        result = add_hmm_possession(df, params)
        self.assertEqual(set(result.attrs["hmm_feature_importance"]["feature"]),
                         {"min_team_a_dist", "min_team_b_dist"})
        self.assertTrue((result.loc[45:49, "hmm_possession"] == "X").all())
        self.assertGreater((result.loc[0:39, "hmm_possession"] == "H").sum(), 30)
        self.assertGreater((result.loc[100:199, "hmm_possession"] == "A").sum(), 80)

    def test_complete_case_training_excludes_missing_rows(self):
        df = sample()
        df.loc[0:240, "min_team_b_dist"] = np.nan
        params = {
            **PARAMS,
            "features": ["min_team_a_dist", "min_team_b_dist"],
            "training_missing": "complete",
        }
        with self.assertRaisesRegex(ValueError, "training_missing='complete'; found 59"):
            add_hmm_possession(df, params)

    def test_complete_training_breaks_at_missing_rows_and_freezes_transitions(self):
        df = sample()
        df.loc[20, "min_team_b_dist"] = np.nan
        params = {
            **PARAMS,
            "features": ["min_team_a_dist", "min_team_b_dist"],
            "training_missing": "complete",
            "freeze_transitions": True,
        }
        fit = GaussianHMM.fit
        fitted = {}

        def capture_fit(model, observations, lengths):
            fitted["lengths"] = lengths.copy()
            fitted["observations"] = observations.copy()
            fitted["transition_before"] = model.transmat_.copy()
            result = fit(model, observations, lengths)
            fitted["transition_after"] = model.transmat_.copy()
            return result

        with patch.object(GaussianHMM, "fit", capture_fit):
            result = add_hmm_possession(df, params)
        self.assertEqual(fitted["lengths"], [20, 24, 250])
        self.assertEqual(fitted["observations"].shape, (294, 2))
        np.testing.assert_array_equal(fitted["transition_before"], fitted["transition_after"])
        self.assertNotEqual(result.loc[20, "hmm_possession"], "X")

    def test_stationary_and_timed_dead_anchor_are_required_when_enabled(self):
        df = sample()
        df["ball_displacement"] = 1.0
        df["sp_since"] = 10.0
        params = {
            **PARAMS,
            "features": ["min_team_a_dist", "min_team_b_dist"],
            "dead_anchor_max_displacement": 0.05,
            "dead_anchor_max_since": 3.0,
        }
        with self.assertRaisesRegex(ValueError, "initialization"):
            add_hmm_possession(df, params)
        df.loc[200:249, "ball_displacement"] = 0.01
        df.loc[200:249, "sp_since"] = 1.0
        result = add_hmm_possession(df, params)
        self.assertEqual(len(result), len(df))

    def test_invalid_experimental_configuration_fails(self):
        df = sample()
        for overrides in [
            {"features": ["min_team_a_dist"]},
            {"features": ["min_team_a_dist", "min_team_b_dist", "unknown"]},
            {"min_feature_coverage": 1.1},
            {"training_missing": "marginal"},
            {"state_mapping": "invalid"},
            {"smoothing_half_window": -1},
            {"dead_anchor_max_displacement": -1},
            {"freeze_transitions": "yes"},
        ]:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    add_hmm_possession(df, {**PARAMS, **overrides})

    def test_viterbi_uses_transition_history_for_ambiguous_frames(self):
        model = GaussianHMM(n_components=2, covariance_type="diag", init_params="")
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])
        model.means_ = np.array([[0.0], [5.0]])
        model.covars_ = np.array([[1.0], [1.0]])
        valid = np.ones(2, dtype=bool)
        np.testing.assert_array_equal(_viterbi(model, np.array([[0.0], [3.0]]), valid), [0, 0])
        np.testing.assert_array_equal(_viterbi(model, np.array([[5.0], [3.0]]), valid), [1, 1])

    def test_training_preserves_order_input_and_missing_abstentions(self):
        df = sample().sample(frac=1, random_state=5)
        original = df.copy(deep=True)
        result = add_hmm_possession(df, PARAMS)
        pd.testing.assert_frame_equal(df, original)
        pd.testing.assert_frame_equal(result.drop(columns="hmm_possession"), original)
        missing_frames = df.loc[df.loc[:, FEATURES].isna().all(axis=1), "video_frame"]
        missing_labels = result.loc[
            result.video_frame.isin(missing_frames), "hmm_possession"
        ]
        self.assertTrue((missing_labels == "X").all())
        self.assertNotEqual(result.loc[result.video_frame == 40, "hmm_possession"].iloc[0], "X")
        observed_labels = result.loc[
            ~result.video_frame.isin(missing_frames), "hmm_possession"
        ]
        self.assertFalse((observed_labels == "X").any())
        for start, label in [(0, "H"), (100, "A"), (200, "D")]:
            section = result[(result.video_frame >= start) & (result.video_frame < start + 100)]
            self.assertGreater((section.hmm_possession == label).sum(), 80)
        pd.testing.assert_series_equal(
            result.hmm_possession, add_hmm_possession(df, PARAMS).hmm_possession
        )

    def test_rejects_bad_input_and_insufficient_anchors(self):
        df = sample()
        with self.assertRaisesRegex(ValueError, "missing required"):
            add_hmm_possession(df.drop(columns="sp_kind"), PARAMS)
        with self.assertRaisesRegex(ValueError, "unique"):
            add_hmm_possession(pd.concat([df, df.iloc[:1]]), PARAMS)
        df["sp_kind"] = None
        with self.assertRaisesRegex(ValueError, "initialization"):
            add_hmm_possession(df, PARAMS)

    def test_removed_features_do_not_affect_training_or_report(self):
        excluded = (
            "ball_speed", "ball_acceleration", "ball_direction_deg",
            "ball_direction_change_deg", "ball_direction_change_abs_deg",
            "team_a_nearest_player_ball_relative_velocity",
            "team_b_nearest_player_ball_relative_velocity",
            "nearest_player_ball_relative_velocity", "ball_velocity_x",
            "ball_velocity_toward_team_a_goal", "ball_velocity_toward_team_b_goal",
            "team_centroid_velocity_x_all_players", "team_centroid_velocity_y_all_players",
            "team_centroid_speed_all_players", "team_centroid_velocity_x_barcelona",
            "team_centroid_velocity_y_barcelona", "team_centroid_speed_barcelona",
            "team_centroid_velocity_x_atletic_club", "team_centroid_velocity_y_atletic_club",
            "team_centroid_speed_atletic_club", "team_movement_coherence_barcelona",
            "team_movement_coherence_players_barcelona", "team_movement_coherence_atletic_club",
            "team_movement_coherence_players_atletic_club", "opponent_team_closing_speed",
        )
        self.assertTrue(set(excluded).isdisjoint(FEATURES))
        df = sample()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports" / "importance.csv"
            baseline = add_hmm_possession(df, {**PARAMS, "feature_importance_path": path})
            report = pd.read_csv(path)
            self.assertEqual(set(report["feature"]), set(FEATURES))
            pd.testing.assert_frame_equal(
                baseline.attrs["hmm_feature_importance"], report, check_exact=False
            )
            for name in excluded:
                df[name] = np.inf
            changed = add_hmm_possession(df, PARAMS)
            pd.testing.assert_series_equal(baseline["hmm_possession"], changed["hmm_possession"])
            pd.testing.assert_frame_equal(
                baseline.attrs["hmm_feature_importance"],
                changed.attrs["hmm_feature_importance"],
            )

    def test_convergence_history_is_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reports" / "convergence.csv"
            state_path = Path(directory) / "reports" / "states.csv"
            result = add_hmm_possession(
                sample(), {**PARAMS, "convergence_path": path, "state_diagnostics_path": state_path}
            )
            pd.testing.assert_frame_equal(
                result.attrs["hmm_convergence"], pd.read_csv(path), check_exact=False
            )
            pd.testing.assert_frame_equal(
                result.attrs["hmm_states"], pd.read_csv(state_path), check_exact=False
            )
            self.assertEqual(set(result.attrs["hmm_states"]["label"]), {"H", "A", "D"})
            np.testing.assert_allclose(
                result.attrs["hmm_states"][["transition_to_H", "transition_to_A", "transition_to_D"]]
                .sum(axis=1),
                1.0,
            )
            self.assertIn(result.attrs["hmm_convergence_status"], {
                "converged", "likelihood_decreased", "iteration_limit", "not_converged",
            })

    def test_likelihood_decrease_is_reported_even_if_library_marks_converged(self):
        fit = GaussianHMM.fit

        def fit_with_decrease(model, *args, **kwargs):
            fit(model, *args, **kwargs)
            model.monitor_.history.append(model.monitor_.history[-1] - 2.78)
            model.monitor_.iter += 1
            return model

        with patch.object(GaussianHMM, "fit", fit_with_decrease):
            with self.assertLogs("stages.hmm_possession", level="WARNING") as logs:
                result = add_hmm_possession(sample(), PARAMS)
        self.assertIn("likelihood_decreased", logs.output[0])
        self.assertEqual(result.attrs["hmm_convergence_status"], "likelihood_decreased")
        self.assertAlmostEqual(result.attrs["hmm_convergence"]["delta"].iloc[-1], -2.78)

    def test_unobserved_set_piece_frames_do_not_count_as_anchors(self):
        df = sample()
        df.loc[209:299, list(FEATURES)] = np.nan
        with self.assertRaisesRegex(ValueError, r"initialization.*\[.*9\]"):
            add_hmm_possession(df, PARAMS)


class ComparisonTests(unittest.TestCase):
    def test_compares_predictions_and_rejects_bad_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt.csv"
            first = root / "first.csv"
            second = root / "second.csv"
            for path, labels in [
                (gt, ["H", "A", "D", "X"]),
                (first, ["H", "A", "D", "H"]),
                (second, ["H", "D", "D", "A"]),
            ]:
                pd.DataFrame({"frame": range(4), "label": labels}).to_csv(path, index=False)
            result = compare(first, second, gt)
            self.assertEqual(result["changed_frames"], 2)
            self.assertEqual(result["changed_graded_frames"], 1)
            self.assertEqual(result["graded_frames"], 3)
            self.assertEqual(result["runs"][0]["accuracy"], 1.0)
            self.assertAlmostEqual(result["runs"][1]["accuracy"], 2 / 3)
            pd.DataFrame({"frame": [0, 0, 2, 3], "label": ["H"] * 4}).to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "frame.*in order"):
                compare(first, second, gt)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.ball_focused_features_extraction import (
    extract_player_density, mask_unobserved_player_density,
)
from stages.hybrid_possession import add_hybrid_possession
from stages.export import export_hybrid_evaluation


PARAMS = {
    "fps": 2, "pitch_m": (105.3, 68.0), "clear_distance_m": 3.0,
    "still_distance_m": 0.5, "slow_speed_mps": 0.7,
    "min_tracked_players": 1, "smoothing_half_window": 0,
    "dead_min_duration_s": 0, "dead_merge_gap_s": 0, "min_possession_s": 0,
    "set_piece_cutoffs_s": {"free_kick": 2.0},
}


def frames():
    return pd.DataFrame({
        "period_id": [1] * 8,
        "video_frame": np.arange(8),
        "min_team_a_dist": [1, 5, np.nan, np.nan, 2, 2, 2, 2],
        "min_team_b_dist": [5, 4, np.nan, np.nan, 5, 5, 5, 5],
        "ball_x": [.1, .2, np.nan, np.nan, .5, .5, .5, .5],
        "ball_y": [.5] * 8,
        "sp_kind": [None, None, "free_kick", None, "free_kick", "free_kick", None, None],
        "sp_since": [np.nan, np.nan, 1, np.nan, 1, 1, np.nan, np.nan],
        "home_attacking": [0, 1, 0, 0, 0, 0, 0, 0],
        "away_attacking": [0, 0, 0, 0, 0, 0, 0, 0],
        "home_defending": [0] * 8,
        "away_defending": [0] * 8,
        "1_s": [2, 2, 2, 2, 0, 0, 0, 0],
        "hmm_possession": ["A", "A", "H", "D", "A", "A", "A", "A"],
    })


class DensityTests(unittest.TestCase):
    def test_missing_distance_is_not_zero_density(self):
        source = pd.DataFrame({
            "1_ball_dist_m": [np.nan, 4, 1, np.nan],
            "1_team": ["Barcelona"] * 4,
            "2_ball_dist_m": [1, 4, np.nan, np.nan],
            "2_team": ["Atletic Club"] * 4,
        })
        result = extract_player_density(source)
        self.assertTrue(np.isnan(result.loc[0, "team_a_players_1m"]))
        self.assertEqual(result.loc[1, "team_a_players_1m"], 0)
        self.assertEqual(result.loc[2, "team_a_players_1m"], 1)
        self.assertTrue(np.isnan(result.loc[2, "team_b_players_1m"]))
        self.assertTrue(np.isnan(result.loc[3, "team_a_players_1m"]))

    def test_cached_density_corrected_per_team_without_mutation(self):
        source = pd.DataFrame({"min_team_a_dist": [np.nan, 4], "min_team_b_dist": [1, np.nan]})
        for side in ("a", "b"):
            for radius in (1, 2, 5):
                source[f"team_{side}_players_{radius}m"] = 0
        original = source.copy()
        corrected = mask_unobserved_player_density(source)
        pd.testing.assert_frame_equal(source, original)
        self.assertTrue(np.isnan(corrected.loc[0, "team_a_players_1m"]))
        self.assertEqual(corrected.loc[0, "team_b_players_1m"], 0)
        self.assertTrue(np.isnan(corrected.loc[1, "team_b_players_2m"]))


class HybridTests(unittest.TestCase):
    def test_geometry_tags_dead_and_unknown_fallback(self):
        source = frames().sample(frac=1, random_state=4)
        original = source.copy()
        result = add_hybrid_possession(source, PARAMS)
        pd.testing.assert_frame_equal(source, original)
        labels = result.sort_values("video_frame")["hybrid_possession"].to_list()
        self.assertEqual(labels, ["H", "H", "D", "X", "D", "D", "H", "H"])
        self.assertEqual(export_hybrid_evaluation(result).columns.tolist(), ["frame", "label"])

    def test_missing_ball_is_not_dead_outside_a_set_piece(self):
        source = frames()
        source.loc[3, "hmm_possession"] = "A"
        result = add_hybrid_possession(source, PARAMS)
        self.assertEqual(result.loc[3, "hybrid_possession"], "A")

    def test_defending_tag_assigns_opponent_possession(self):
        source = frames()
        source.loc[3, "away_defending"] = 1
        result = add_hybrid_possession(source, PARAMS)
        self.assertEqual(result.loc[3, "hybrid_possession"], "H")

    def test_geometry_first_prefers_geometry_and_preserves_fallbacks(self):
        source = frames()
        source.loc[0, "away_attacking"] = 1
        source.loc[3, "away_defending"] = 1
        source.loc[6, "min_team_b_dist"] = np.nan
        source.loc[6, "away_attacking"] = 1
        source.loc[7, "ball_x"] = np.nan
        source.loc[7, "hmm_possession"] = "H"

        baseline = add_hybrid_possession(source, PARAMS)
        explicit_default = add_hybrid_possession(source, {**PARAMS, "geometry_first": False})
        geometry_first = add_hybrid_possession(source, {**PARAMS, "geometry_first": True})

        self.assertEqual(
            baseline["hybrid_possession"].to_list(),
            ["H", "H", "D", "H", "D", "D", "A", "H"],
        )
        pd.testing.assert_series_equal(
            baseline["hybrid_possession"], explicit_default["hybrid_possession"]
        )
        self.assertEqual(
            geometry_first["hybrid_possession"].to_list(),
            ["H", "A", "D", "H", "D", "D", "A", "H"],
        )

    def test_geometry_first_requires_boolean(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                add_hybrid_possession(frames(), {**PARAMS, "geometry_first": value})

    def test_outside_signal_does_not_cross_gap_or_period(self):
        source = frames().iloc[:4].copy()
        source["ball_x"] = 1.2
        source["sp_kind"] = None
        source["hmm_possession"] = "A"
        source["video_frame"] = [0, 2, 3, 4]
        source["period_id"] = [1, 1, 1, 2]
        params = {**PARAMS, "dead_min_duration_s": 1.0}
        result = add_hybrid_possession(source, params)
        self.assertEqual(result["hybrid_possession"].to_list(), ["A", "D", "D", "A"])

    def test_short_dead_gap_is_merged_before_minimum_duration_filter(self):
        source = frames()
        source["ball_x"] = [1.2, 1.2, 0.5, 0.5, 1.2, 1.2, 0.5, 0.5]
        source["ball_y"] = 0.5
        source["hmm_possession"] = "H"
        params = {
            **PARAMS,
            "dead_min_duration_s": 2.0,
            "dead_merge_gap_s": 1.0,
        }
        result = add_hybrid_possession(source, params)
        self.assertEqual(
            result["hybrid_possession"].to_list(),
            ["D", "D", "D", "D", "D", "D", "H", "H"],
        )

    def test_short_possession_is_merged_but_unknown_frames_remain_unknown(self):
        source = frames()
        source["min_team_a_dist"] = [1, 1, 1, 5, 5, 1, 1, 1]
        source["min_team_b_dist"] = [5, 5, 5, 1, 1, 5, 5, 5]
        source["ball_x"] = 0.5
        source["ball_y"] = 0.5
        source["sp_kind"] = None
        source["sp_since"] = np.nan
        source["home_attacking"] = 0
        source["away_attacking"] = 0
        source["hmm_possession"] = "X"
        source.loc[3, ["min_team_a_dist", "min_team_b_dist"]] = [5, 1]
        source.loc[4, ["min_team_a_dist", "min_team_b_dist"]] = np.nan
        source.loc[4, "ball_x"] = np.nan
        params = {**PARAMS, "min_possession_s": 1.5}
        result = add_hybrid_possession(source, params)
        self.assertEqual(
            result["hybrid_possession"].to_list(),
            ["H", "H", "H", "H", "X", "H", "H", "H"],
        )

    def test_invalid_input_rejected(self):
        source = frames()
        for broken in [
            source.drop(columns="1_s"),
            source.assign(home_attacking=2),
            source.assign(video_frame=0),
            source.assign(hmm_possession="invalid"),
        ]:
            with self.assertRaises(ValueError):
                add_hybrid_possession(broken, PARAMS)


if __name__ == "__main__":
    unittest.main()

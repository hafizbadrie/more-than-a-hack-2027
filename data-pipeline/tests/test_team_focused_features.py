import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.team_focused_features_extraction import (
    team_average_radial_movement,
    team_position_relative_to_ball,
)
from stages.ball_focused_features_extraction import extract_ball_trajectory_toward_goal


class BallCoordinateUnitTests(unittest.TestCase):
    def test_radial_movement_uses_ball_coordinates_in_metres(self):
        source = pd.DataFrame({
            "period_id": [1, 1],
            "video_frame": [0, 1],
            "ball_x": [0.5, 0.5],
            "ball_y": [0.5, 0.5],
            "1_x": [40.0 / 105.3, 41.0 / 105.3],
            "1_y": [0.5, 0.5],
            "1_x_m": [40.0, 41.0],
            "1_y_m": [34.0, 34.0],
            "1_team": ["Barcelona", "Barcelona"],
            "2_y": [0.6, 0.6],
            "2_y_m": [40.8, 40.8],
        })

        result = team_average_radial_movement(source, {"fps": 25})

        self.assertAlmostEqual(
            result.loc[1, "team_average_radial_movement_barcelona"], 25.0
        )

    def test_team_ball_offsets_are_in_metres(self):
        source = pd.DataFrame({
            "period_id": [1],
            "video_frame": [0],
            "ball_x": [0.5],
            "ball_y": [0.5],
            "1_x": [40.0 / 105.3],
            "1_y": [30.0 / 68.0],
            "1_x_m": [40.0],
            "1_y_m": [30.0],
            "1_team": ["Barcelona"],
            "2_x_m": [50.0],
            "2_y_m": [38.0],
            "2_x": [50.0 / 105.3],
            "2_y": [38.0 / 68.0],
            "2_team": ["Atletic Club"],
        })

        result = team_position_relative_to_ball(
            source,
            {
                "attack_directions": {"Barcelona": 1, "Atletic Club": -1},
            },
        )

        self.assertAlmostEqual(
            result.loc[0, "team_ball_longitudinal_distance_barcelona"], 12.65
        )
        self.assertAlmostEqual(
            result.loc[0, "team_ball_lateral_distance_barcelona"], 4.0
        )
        self.assertAlmostEqual(
            result.loc[0, "team_ball_longitudinal_distance_atletic_club"], -2.65
        )
        self.assertAlmostEqual(
            result.loc[0, "team_ball_lateral_distance_atletic_club"], -4.0
        )

    def test_unidentifiable_conversion_is_rejected(self):
        source = pd.DataFrame({
            "video_frame": [0],
            "ball_x": [0.5],
            "ball_y": [0.5],
            "1_x": [0.4],
            "1_y": [0.4],
            "1_x_m": [40.0],
            "1_y_m": [30.0],
            "1_team": ["Barcelona"],
        })

        with self.assertRaisesRegex(ValueError, "affine.*x"):
            team_average_radial_movement(source)

    def test_missing_or_nonfinite_calibration_is_rejected(self):
        source = pd.DataFrame({
            "video_frame": [0, 1],
            "ball_x": [0.5, 0.5],
            "ball_y": [0.5, 0.5],
            "1_x": [0.2, 0.3],
            "1_x_m": [20.0, 30.0],
            "1_y_m": [10.0, 20.0],
            "1_team": ["Barcelona", "Barcelona"],
        })
        params = {"attack_directions": {"Barcelona": 1}}
        with self.assertRaisesRegex(ValueError, "affine.*y"):
            team_position_relative_to_ball(source, params)
        source["1_y"] = [float("nan"), float("inf")]
        with self.assertRaisesRegex(ValueError, "affine.*y"):
            team_position_relative_to_ball(source, params)

    def test_affine_conversion_used_by_both_features(self):
        source = pd.DataFrame({
            "video_frame": [0, 1],
            "custom_x": [0.5, 0.5],
            "custom_y": [0.5, 0.5],
            "4_x": [0.3, 0.31],
            "4_y": [0.3, 0.3],
            "4_x_m": [40.0, 41.0],
            "4_y_m": [22.0, 22.0],
            "4_team": ["Barcelona", "Barcelona"],
            "5_x": [0.4, float("nan")],
            "5_y": [0.5, float("nan")],
            "5_x_m": [50.0, float("nan")],
            "5_y_m": [34.0, float("nan")],
            "5_team": ["Atletic Club", "Atletic Club"],
        })
        params = {
            "fps": 25,
            "ball_x_col": "custom_x",
            "ball_y_col": "custom_y",
            "attack_directions": {"Barcelona": 1, "Atletic Club": -1},
        }

        radial = team_average_radial_movement(source, params)
        self.assertAlmostEqual(
            radial.loc[1, "team_average_radial_movement_barcelona"],
            25 * 19 / (19**2 + 12**2) ** 0.5,
        )
        positions = team_position_relative_to_ball(source, params)
        self.assertAlmostEqual(
            positions.loc[0, "team_ball_longitudinal_distance_barcelona"], 20.0
        )
        self.assertAlmostEqual(
            positions.loc[0, "team_ball_lateral_distance_barcelona"], 12.0
        )

    def test_ball_goal_directions_reverse_by_period(self):
        source = pd.DataFrame({
            "period_id": [1, 2],
            "ball_dx": [1.0, 1.0],
            "ball_displacement": [1.0, 1.0],
            "ball_speed": [2.0, 2.0],
        })

        result = extract_ball_trajectory_toward_goal(
            source,
            {
                "team_a_attack_direction": 1,
                "team_b_attack_direction": -1,
                "period_direction_multipliers": {1: 1, 2: -1},
            },
        )

        self.assertEqual(
            result["ball_velocity_toward_team_a_goal"].tolist(), [2.0, -2.0]
        )
        self.assertEqual(
            result["ball_velocity_toward_team_b_goal"].tolist(), [-2.0, 2.0]
        )

    def test_team_ball_longitudinal_direction_reverses_by_period(self):
        source = pd.DataFrame({
            "period_id": [1, 2],
            "video_frame": [10, 20],
            "ball_x": [0.5, 0.5],
            "ball_y": [0.5, 0.5],
            "1_x": [40.0 / 105.3, 40.0 / 105.3],
            "1_y": [30.0 / 68.0, 30.0 / 68.0],
            "1_x_m": [40.0, 40.0],
            "1_y_m": [30.0, 30.0],
            "1_team": ["Barcelona", "Barcelona"],
            "2_x_m": [50.0, 50.0],
            "2_y_m": [38.0, 38.0],
            "2_x": [50.0 / 105.3, 50.0 / 105.3],
            "2_y": [38.0 / 68.0, 38.0 / 68.0],
            "2_team": ["Atletic Club", "Atletic Club"],
        })

        result = team_position_relative_to_ball(
            source,
            {
                "attack_directions": {"Barcelona": 1, "Atletic Club": -1},
                "period_direction_multipliers": {1: 1, 2: -1},
            },
        )

        for actual, expected in zip(
            result["team_ball_longitudinal_distance_barcelona"], [12.65, -12.65]
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            result["team_ball_longitudinal_distance_atletic_club"], [-2.65, 2.65]
        ):
            self.assertAlmostEqual(actual, expected)

    def test_unmapped_period_is_rejected(self):
        source = pd.DataFrame({
            "period_id": [3],
            "ball_dx": [1.0],
            "ball_displacement": [1.0],
            "ball_speed": [2.0],
        })

        with self.assertRaisesRegex(ValueError, "Missing period direction"):
            extract_ball_trajectory_toward_goal(
                source,
                {
                    "team_a_attack_direction": 1,
                    "team_b_attack_direction": -1,
                    "period_direction_multipliers": {1: 1, 2: -1},
                },
            )


if __name__ == "__main__":
    unittest.main()

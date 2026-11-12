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
            "1_x_m": [40.0, 41.0],
            "1_y_m": [34.0, 34.0],
            "1_team": ["Barcelona", "Barcelona"],
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
            "1_x_m": [40.0],
            "1_y_m": [30.0],
            "1_team": ["Barcelona"],
            "2_x_m": [50.0],
            "2_y_m": [38.0],
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

    def test_invalid_pitch_dimensions_are_rejected(self):
        source = pd.DataFrame({
            "video_frame": [0],
            "ball_x": [0.5],
            "ball_y": [0.5],
            "1_x_m": [40.0],
            "1_y_m": [30.0],
            "1_team": ["Barcelona"],
        })

        with self.assertRaisesRegex(ValueError, "pitch_m"):
            team_average_radial_movement(source, {"pitch_m": (105.3, 0)})

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
            "1_x_m": [40.0, 40.0],
            "1_y_m": [30.0, 30.0],
            "1_team": ["Barcelona", "Barcelona"],
            "2_x_m": [50.0, 50.0],
            "2_y_m": [38.0, 38.0],
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

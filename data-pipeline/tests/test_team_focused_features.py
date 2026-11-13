import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.team_focused_features_extraction import (
    team_average_radial_movement,
    team_position_relative_to_ball,
)
from stages.ball_focused_features_extraction import (
    extract_ball_acceleration,
    extract_ball_direction_change,
    extract_ball_speed,
    extract_ball_trajectory_toward_goal,
)
from stages.direction import infer_attack_directions


class BallCoordinateUnitTests(unittest.TestCase):
    def test_inferred_directions_drive_both_features(self):
        source = pd.DataFrame({
            "period_id": [1, 1, 2, 2],
            "video_frame": [0, 1, 2, 3],
            "ball_x": [0.5] * 4,
            "ball_y": [0.5] * 4,
            "1_x": [0.7, 0.6, 0.2, 0.3],
            "1_y": [0.4] * 4,
            "1_x_m": [70.0, 60.0, 20.0, 30.0],
            "1_y_m": [40.0] * 4,
            "1_team": ["Barcelona"] * 4,
            "2_x": [0.3, 0.4, 0.8, 0.7],
            "2_y": [0.6] * 4,
            "2_x_m": [30.0, 40.0, 80.0, 70.0],
            "2_y_m": [60.0] * 4,
            "2_team": ["Atletic Club"] * 4,
            "ball_dx": [1.0] * 4,
            "ball_displacement": [1.0] * 4,
            "ball_speed_calc": [2.0] * 4,
        })
        params = {"team_a": "Barcelona", "team_b": "Atletic Club"}
        directions = infer_attack_directions(source, params)
        self.assertEqual(directions, {
            1: {"Barcelona": -1, "Atletic Club": 1},
            2: {"Barcelona": 1, "Atletic Club": -1},
        })
        trajectory = extract_ball_trajectory_toward_goal(source, params, directions)
        self.assertEqual(
            trajectory["ball_velocity_toward_team_a_goal"].tolist(),
            [-2.0, -2.0, 2.0, 2.0],
        )
        self.assertEqual(
            trajectory["ball_velocity_toward_team_b_goal"].tolist(),
            [2.0, 2.0, -2.0, -2.0],
        )
        positions = team_position_relative_to_ball(source, {}, directions)
        self.assertAlmostEqual(
            positions.loc[0, "team_ball_longitudinal_distance_barcelona"], 20.0
        )
        self.assertAlmostEqual(
            positions.loc[2, "team_ball_longitudinal_distance_barcelona"], 30.0
        )
        self.assertAlmostEqual(
            positions.loc[0, "team_ball_longitudinal_distance_atletic_club"], 20.0
        )
        self.assertAlmostEqual(
            positions.loc[2, "team_ball_longitudinal_distance_atletic_club"], 30.0
        )
        with self.assertRaisesRegex(ValueError, "period"):
            extract_ball_trajectory_toward_goal(source, params, {1: directions[1]})
        with self.assertRaisesRegex(ValueError, "period"):
            team_position_relative_to_ball(source, {}, {1: directions[1]})

    def test_inference_uses_mean_frame_centroids_and_rejects_missing_or_tied_teams(self):
        source = pd.DataFrame({
            "period_id": [1, 1],
            "video_frame": [0, 1],
            "1_x_m": [0.0, 100.0],
            "1_team": ["Barcelona"] * 2,
            "2_x_m": [0.0, float("nan")],
            "2_team": ["Barcelona"] * 2,
            "3_x_m": [40.0, 40.0],
            "3_team": ["Atletic Club"] * 2,
        })
        params = {"team_a": "Barcelona", "team_b": "Atletic Club"}
        self.assertEqual(infer_attack_directions(source, params), {
            1: {"Barcelona": -1, "Atletic Club": 1},
        })
        missing = source.copy()
        missing["3_x_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Atletic Club"):
            infer_attack_directions(missing, params)
        tied = source.copy()
        tied["3_x_m"] = 50.0
        with self.assertRaisesRegex(ValueError, "equal"):
            infer_attack_directions(tied, params)

    def test_ball_motion_uses_metres_and_preserves_source_speed(self):
        source = pd.DataFrame({
            "video_frame": [10, 12, 14, 16],
            "ball_x": [0.20, 0.22, float("nan"), 0.25],
            "ball_y": [0.30, 0.32, float("nan"), 0.40],
            "ball_speed": [70.0, 71.0, 72.0, 73.0],
            "1_x": [0.1, 0.2, 0.3, 0.4],
            "1_y": [0.1, 0.2, 0.3, 0.4],
            "1_x_m": [20.0, 30.0, 40.0, 50.0],
            "1_y_m": [12.0, 17.0, 22.0, 27.0],
        })
        params = {
            "frame_col": "video_frame", "ball_x_col": "ball_x",
            "ball_y_col": "ball_y", "fps": 10,
            "team_a_attack_direction": 1, "team_b_attack_direction": -1,
        }

        speed = extract_ball_speed(source, params)
        pd.testing.assert_series_equal(speed["ball_speed"], source["ball_speed"])
        without_source_speed = extract_ball_speed(source.drop(columns="ball_speed"), params)
        self.assertNotIn("ball_speed", without_source_speed)
        pd.testing.assert_series_equal(
            speed["ball_speed_calc"], without_source_speed["ball_speed_calc"]
        )
        pd.testing.assert_frame_equal(extract_ball_speed(speed, params), speed)
        self.assertAlmostEqual(speed.loc[1, "ball_dx"], 2.0)
        self.assertAlmostEqual(speed.loc[1, "ball_dy"], 1.0)
        self.assertAlmostEqual(speed.loc[1, "ball_displacement"], 5 ** 0.5)
        self.assertAlmostEqual(speed.loc[1, "ball_speed_calc"], 5 ** 0.5 / 0.2)
        self.assertTrue(pd.isna(speed.loc[2, "ball_speed_calc"]))
        self.assertAlmostEqual(speed.loc[3, "ball_displacement"], 5.0)
        self.assertAlmostEqual(speed.loc[3, "ball_speed_calc"], 12.5)
        self.assertTrue(pd.isna(speed.loc[0, "ball_speed_calc"]))

        acceleration = extract_ball_acceleration(speed, params)
        self.assertAlmostEqual(
            acceleration.loc[3, "ball_acceleration"],
            (12.5 - 5 ** 0.5 / 0.2) / 0.4,
        )
        direction = extract_ball_direction_change(acceleration, params)
        self.assertAlmostEqual(direction.loc[1, "ball_direction_deg"], 26.565051177)
        trajectory = extract_ball_trajectory_toward_goal(direction, params)
        self.assertAlmostEqual(trajectory.loc[1, "ball_velocity_x"], 10.0)
        self.assertAlmostEqual(trajectory.loc[3, "ball_velocity_x"], 7.5)

    def test_ball_speed_requires_coordinate_calibration(self):
        source = pd.DataFrame({
            "video_frame": [0, 1], "ball_x": [0.2, 0.3], "ball_y": [0.2, 0.3],
        })
        with self.assertRaisesRegex(ValueError, "affine.*x"):
            extract_ball_speed(source)

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
            "ball_speed_calc": [2.0, 2.0],
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
            "ball_speed_calc": [2.0],
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

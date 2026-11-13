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
    extract_nearest_player_ball_distance,
    extract_nearest_player_ball_relative_velocity,
)
from stages.direction import infer_attack_directions


class BallCoordinateUnitTests(unittest.TestCase):
    def test_nearest_player_ties_do_not_select_a_team(self):
        source = pd.DataFrame({
            "video_frame": list(range(7)),
            "1_ball_dist_m": [8.0, 6.0, 6.0, 5.0, float("nan"), 7.0, float("nan")],
            "1_team": ["Barcelona"] * 7,
            "2_ball_dist_m": [8.0, 6.0, 5.0, 6.0, 7.0, float("nan"), float("nan")],
            "2_team": ["Atletic Club"] * 7,
        })
        params = {
            "frame_col": "video_frame", "fps": 1,
            "distance_suffix": "_ball_dist_m", "team_suffix": "_team",
            "team_a": "Barcelona", "team_b": "Atletic Club",
        }

        distance = extract_nearest_player_ball_distance(source, params)
        relative = extract_nearest_player_ball_relative_velocity(source, params)
        for result, column in (
            (distance, "nearest_player_team"),
            (relative, "nearest_player_ball_relative_velocity_team"),
        ):
            with self.subTest(column=column):
                for row in (0, 1, 6):
                    self.assertTrue(pd.isna(result.loc[row, column]))
                self.assertEqual(result.loc[2, column], "Atletic Club")
                self.assertEqual(result.loc[3, column], "Barcelona")
                self.assertEqual(result.loc[4, column], "Atletic Club")
                self.assertEqual(result.loc[5, column], "Barcelona")

        self.assertEqual(distance.loc[1, "nearest_player_dist"], 6.0)
        self.assertEqual(relative.loc[1, "team_a_nearest_player_ball_relative_velocity"], 1.0)
        self.assertEqual(relative.loc[1, "team_b_nearest_player_ball_relative_velocity"], 1.5)
        self.assertTrue(pd.isna(relative.loc[1, "nearest_player_ball_relative_velocity"]))

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
        self.assertTrue(speed["ball_speed_calc"].isna().all())

    def test_ball_speed_requires_coordinate_calibration(self):
        source = pd.DataFrame({
            "video_frame": [0, 1], "ball_x": [0.2, 0.3], "ball_y": [0.2, 0.3],
        })
        with self.assertRaisesRegex(ValueError, "affine.*x"):
            extract_ball_speed(source)

    def test_ball_motion_resets_at_period_boundaries(self):
        source = pd.DataFrame({
            "period_id": [1] * 4 + [2] * 5,
            "video_frame": [0, 1, 2, 3, 0, 1, 2, 3, 4],
            "ball_x": [0.1, 0.2, 0.4, 0.4, float("nan"), 0.8, 0.7, 0.5, 0.5],
            "ball_y": [0.1, 0.1, 0.1, 0.2, float("nan"), 0.8, 0.8, 0.8, 0.7],
            "1_x": [0.0, 1.0] + [0.0] * 7,
            "1_y": [0.0, 1.0] + [0.0] * 7,
            "1_x_m": [0.0, 10.0] + [0.0] * 7,
            "1_y_m": [0.0, 10.0] + [0.0] * 7,
        })
        params = {"frame_col": "video_frame", "ball_x_col": "ball_x",
                  "ball_y_col": "ball_y", "fps": 1}

        speed = extract_ball_speed(source, params)
        acceleration = extract_ball_acceleration(speed, params)
        result = extract_ball_direction_change(acceleration, params)

        for row in (0, 4, 5, 8):
            for column in ("ball_dx", "ball_dy", "ball_displacement", "ball_speed_calc",
                           "ball_acceleration", "ball_direction_deg",
                           "ball_direction_change_deg", "ball_direction_change_abs_deg"):
                with self.subTest(row=row, column=column):
                    self.assertTrue(pd.isna(result.loc[row, column]))
        self.assertTrue(result["ball_acceleration"].isna().all())
        self.assertTrue(result["ball_direction_change_deg"].isna().all())
        self.assertAlmostEqual(result.loc[1, "ball_speed_calc"], 1.5)
        self.assertAlmostEqual(result.loc[6, "ball_speed_calc"], 1.5)

    def test_acceleration_and_direction_reset_with_precomputed_motion(self):
        source = pd.DataFrame({
            "period_id": [1, 1, 2, 2, 2],
            "video_frame": [0, 1, 10, 11, 12],
            "ball_speed_calc": [1.0, 2.0, float("nan"), 4.0, 5.0],
            "ball_dx": [1.0, 1.0, float("nan"), 0.0, -1.0],
            "ball_dy": [0.0, 0.0, float("nan"), 1.0, 0.0],
        })
        params = {"frame_col": "video_frame", "fps": 1}

        result = extract_ball_direction_change(extract_ball_acceleration(source, params), params)

        self.assertTrue(pd.isna(result.loc[3, "ball_acceleration"]))
        self.assertTrue(pd.isna(result.loc[3, "ball_direction_change_deg"]))
        self.assertTrue(pd.isna(result.loc[4, "ball_acceleration"]))
        self.assertTrue(pd.isna(result.loc[4, "ball_direction_change_deg"]))

    def test_ball_relative_velocity_resets_at_period_boundaries(self):
        source = pd.DataFrame({
            "period_id": [1, 1, 2, 2, 2],
            "video_frame": [0, 2, 0, 1, 3],
            "1_ball_dist_m": [5.0, 3.0, float("nan"), 8.0, 6.0],
            "1_team": ["Barcelona"] * 5,
            "2_ball_dist_m": [8.0, 6.0, float("nan"), 4.0, 2.0],
            "2_team": ["Atletic Club"] * 5,
        })
        params = {"frame_col": "video_frame", "fps": 1,
                  "distance_suffix": "_ball_dist_m", "team_suffix": "_team",
                  "team_a": "Barcelona", "team_b": "Atletic Club"}

        result = extract_nearest_player_ball_relative_velocity(source, params)

        for column in ("team_a_nearest_player_ball_relative_velocity",
                       "team_b_nearest_player_ball_relative_velocity",
                       "nearest_player_ball_relative_velocity"):
            for row in (0, 2, 3):
                with self.subTest(column=column, row=row):
                    self.assertTrue(pd.isna(result.loc[row, column]))
            self.assertTrue(pd.isna(result.loc[1, column]))
            self.assertTrue(pd.isna(result.loc[4, column]))

        without_period = extract_nearest_player_ball_relative_velocity(
            source.iloc[:2].drop(columns="period_id"), params
        )
        self.assertTrue(pd.isna(without_period.loc[0, "nearest_player_ball_relative_velocity"]))
        self.assertTrue(pd.isna(without_period.loc[1, "nearest_player_ball_relative_velocity"]))

    def test_centered_motion_uses_frame_lag_and_metres(self):
        frames = list(range(17))
        source = pd.DataFrame({
            "video_frame": frames,
            "ball_x": [t * t / 100 for t in frames],
            "ball_y": [0.0] * len(frames),
            "1_x": [t / 100 for t in frames],
            "1_y": [t / 100 for t in frames],
            "1_x_m": frames,
            "1_y_m": frames,
            "1_ball_dist_m": [100.0 - t for t in frames],
            "1_team": ["Barcelona"] * len(frames),
            "2_ball_dist_m": [200.0 - 2 * t for t in frames],
            "2_team": ["Atletic Club"] * len(frames),
        })
        params = {
            "frame_col": "video_frame", "ball_x_col": "ball_x", "ball_y_col": "ball_y",
            "fps": 10, "lag": 2, "max_gap_frames": 1,
            "distance_suffix": "_ball_dist_m", "team_suffix": "_team",
            "team_a": "Barcelona", "team_b": "Atletic Club",
            "team_a_attack_direction": 1, "team_b_attack_direction": -1,
        }
        speed = extract_ball_speed(source, params)
        self.assertAlmostEqual(speed.loc[8, "ball_dx"], 64.0)
        self.assertAlmostEqual(speed.loc[8, "ball_speed_calc"], 160.0)
        self.assertTrue(pd.isna(speed.loc[1, "ball_speed_calc"]))
        self.assertTrue(pd.isna(speed.loc[15, "ball_speed_calc"]))
        acceleration = extract_ball_acceleration(speed, params)
        self.assertAlmostEqual(acceleration.loc[8, "ball_acceleration"], 200.0)
        direction = extract_ball_direction_change(acceleration, params)
        self.assertAlmostEqual(direction.loc[8, "ball_direction_change_deg"], 0.0)
        trajectory = extract_ball_trajectory_toward_goal(direction, params)
        self.assertAlmostEqual(trajectory.loc[8, "ball_velocity_x"], 160.0)
        relative = extract_nearest_player_ball_relative_velocity(source, params)
        self.assertAlmostEqual(relative.loc[8, "team_a_nearest_player_ball_relative_velocity"], 10.0)
        self.assertAlmostEqual(relative.loc[8, "team_b_nearest_player_ball_relative_velocity"], 20.0)
        self.assertEqual(relative.loc[8, "nearest_player_ball_relative_velocity_team"], "Barcelona")
        wider = extract_ball_speed(source, {**params, "lag": 6})
        self.assertAlmostEqual(wider.loc[8, "ball_dx"], 192.0)
        self.assertAlmostEqual(wider.loc[8, "ball_speed_calc"], 160.0)
        self.assertTrue(pd.isna(wider.loc[5, "ball_speed_calc"]))

    def test_centered_windows_reject_large_gaps_and_missing_endpoints(self):
        source = pd.DataFrame({
            "video_frame": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "ball_x": [t / 100 for t in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
            "ball_y": [0.0] * 12,
            "1_x": [0.0, 1.0] + [0.0] * 10,
            "1_y": [0.0, 1.0] + [0.0] * 10,
            "1_x_m": [0.0, 100.0] + [0.0] * 10,
            "1_y_m": [0.0, 100.0] + [0.0] * 10,
            "1_ball_dist_m": [20.0 - t for t in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
            "1_team": ["Barcelona"] * 12,
            "2_ball_dist_m": [30.0 - t for t in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
            "2_team": ["Atletic Club"] * 12,
        })
        params = {
            "frame_col": "video_frame", "ball_x_col": "ball_x", "ball_y_col": "ball_y",
            "fps": 1, "lag": 2, "max_gap_frames": 1,
            "distance_suffix": "_ball_dist_m", "team_suffix": "_team",
            "team_a": "Barcelona", "team_b": "Atletic Club",
        }
        strict = extract_ball_speed(source, params).set_index("video_frame")
        loose = extract_ball_speed(source, {**params, "max_gap_frames": 2}).set_index("video_frame")
        self.assertTrue(pd.isna(strict.loc[2, "ball_speed_calc"]))
        self.assertAlmostEqual(loose.loc[2, "ball_speed_calc"], 1.0)
        self.assertTrue(pd.isna(loose.loc[5, "ball_speed_calc"]))  # missing t-lag
        self.assertAlmostEqual(strict.loc[8, "ball_speed_calc"], 1.0)
        relative = extract_nearest_player_ball_relative_velocity(source, params).set_index("video_frame")
        self.assertTrue(pd.isna(relative.loc[2, "nearest_player_ball_relative_velocity"]))
        self.assertAlmostEqual(relative.loc[8, "nearest_player_ball_relative_velocity"], 1.0)
        for extractor in (extract_ball_speed, extract_ball_acceleration,
                          extract_ball_direction_change, extract_nearest_player_ball_relative_velocity):
            with self.subTest(extractor=extractor.__name__):
                with self.assertRaisesRegex(ValueError, "lag"):
                    extractor(source, {**params, "lag": 0})
                with self.assertRaisesRegex(ValueError, "max_gap_frames"):
                    extractor(source, {**params, "max_gap_frames": 0})

    def test_centered_direction_wrap_and_gaps_in_derived_features(self):
        source = pd.DataFrame({
            "video_frame": list(range(7)),
            "ball_speed_calc": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ball_dx": [-1.0] * 7,
            "ball_dy": [0.0, 0.01, 0.0, -0.01, 0.0, 0.0, 0.0],
        })
        params = {"frame_col": "video_frame", "fps": 2, "lag": 1, "max_gap_frames": 1}
        result = extract_ball_direction_change(extract_ball_acceleration(source, params), params)
        self.assertAlmostEqual(result.loc[2, "ball_acceleration"], 2.0)
        self.assertGreater(result.loc[2, "ball_direction_change_deg"], 0)
        self.assertLess(result.loc[2, "ball_direction_change_deg"], 2)
        source.loc[2, ["ball_speed_calc", "ball_dx"]] = float("nan")
        result = extract_ball_direction_change(extract_ball_acceleration(source, params), params)
        self.assertTrue(pd.isna(result.loc[3, "ball_acceleration"]))
        self.assertTrue(pd.isna(result.loc[3, "ball_direction_change_deg"]))

    def test_centered_windows_never_cross_periods(self):
        source = pd.DataFrame({
            "period_id": [1, 1, 1, 2, 2, 2],
            "video_frame": [0, 1, 2, 3, 4, 5],
            "ball_speed_calc": [0., 1., 2., 3., 4., 5.],
            "ball_dx": [1., 1., 1., 1., 1., 1.],
            "ball_dy": [0., 0., 0., 0., 0., 0.],
            "1_ball_dist_m": [10., 9., 8., 7., 6., 5.],
            "1_team": ["Barcelona"] * 6,
            "2_ball_dist_m": [20., 19., 18., 17., 16., 15.],
            "2_team": ["Atletic Club"] * 6,
        })
        params = {
            "frame_col": "video_frame", "fps": 1, "lag": 1, "max_gap_frames": 5,
            "distance_suffix": "_ball_dist_m", "team_suffix": "_team",
            "team_a": "Barcelona", "team_b": "Atletic Club",
        }
        result = extract_ball_direction_change(extract_ball_acceleration(source, params), params)
        result = extract_nearest_player_ball_relative_velocity(result, params)
        for column in ("ball_acceleration", "ball_direction_change_deg",
                       "nearest_player_ball_relative_velocity"):
            self.assertTrue(pd.isna(result.loc[2, column]))
            self.assertTrue(pd.isna(result.loc[3, column]))
            self.assertTrue(pd.notna(result.loc[1, column]))
            self.assertTrue(pd.notna(result.loc[4, column]))

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

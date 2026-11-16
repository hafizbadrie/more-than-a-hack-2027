import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as pipeline_main
from data_preparation import (
    convert_atd_to_dataframe,
    convert_smart_tagging_to_dataframe,
)
from hybrid_pipeline import prepare_cached_density
from stages.hmm_possession import FEATURES, add_hmm_possession
from stages.team_focused_features_extraction import _ball_coordinates_in_metres


class PipelineTests(unittest.TestCase):
    def test_pipeline_boundaries_and_default(self):
        preparation = pipeline_main.build_data_preparation_pipeline()
        cleansing = pipeline_main.build_features_pipeline()
        hybrid = pipeline_main.build_hybrid_pipeline()
        combined = pipeline_main.build_pipeline()
        self.assertIn("raw_tactical", preparation.outputs())
        self.assertIn("raw_smart_tag", preparation.outputs())
        self.assertIn("raw_tactical", cleansing.inputs())
        self.assertNotIn("final", cleansing.inputs())
        self.assertIn("final", hybrid.inputs())
        self.assertNotIn("raw_tactical", hybrid.inputs())
        self.assertNotIn("raw_tactical", combined.inputs())
        self.assertNotIn("final", combined.inputs())
        self.assertEqual(combined.nodes, (preparation + cleansing + hybrid).nodes)

        for args, expected in [
            ([], combined),
            (["--pipeline", "all"], combined),
            (["--pipeline", "cleansing"], cleansing),
            (["--pipeline", "hmm"], hmm),
        ]:
            with self.subTest(args=args), patch.object(sys, "argv", ["main.py", *args]), \
                    patch.object(pipeline_main, "SequentialRunner") as runner:
                pipeline_main.main()
                actual = runner.return_value.run.call_args.args[0]
                self.assertEqual(actual.nodes, expected.nodes)

    def test_data_preparation_cli_selection(self):
        with patch.object(sys, "argv", ["main.py", "--pipeline", "data_preparation"]), \
                patch.object(pipeline_main, "SequentialRunner") as runner:
            pipeline_main.main()
            actual = runner.return_value.run.call_args.args[0]
            self.assertEqual(actual.nodes, pipeline_main.build_data_preparation_pipeline().nodes)

    def test_data_preparation_atd_conversion_preserves_notebook_conversion(self):
        home = SimpleNamespace(name="Home")
        away = SimpleNamespace(name="Away")
        players = [
            SimpleNamespace(player_id="p1", team=home),
            SimpleNamespace(player_id="p2", team=away),
        ]
        dataset = SimpleNamespace(
            metadata=SimpleNamespace(
                pitch_dimensions=SimpleNamespace(pitch_length=100, pitch_width=50),
                teams=[
                    SimpleNamespace(players=[players[0]]),
                    SimpleNamespace(players=[players[1]]),
                ],
            ),
            to_df=lambda: pd.DataFrame({
                "frame_id": [110808, 110809, 110810],
                "timestamp": ["00:00:00", "00:00:01", "00:00:02"],
                "period_id": [1, 1, 1],
                "ball_x": [0.2, 0.3, np.nan],
                "ball_y": [0.4, 0.4, 0.5],
                "p1_x": [0.5, 0.3, 0.1],
                "p1_y": [0.4, 0.6, 0.5],
                "p2_x": [0.2, np.nan, 0.4],
                "p2_y": [0.4, np.nan, 0.5],
            }),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "tracking.txt"
            raw.write_text("110808:players:0.2,0.4,3.5\n110809:players:0.3,0.4,NaN\n")
            paths = {"atd_meta": str(root / "meta.xml"), "atd_raw": str(raw)}
            output = root / "output.csv"
            catalog = DataCatalog({
                "params:atd_paths": MemoryDataset(paths),
                "params:second_half_start": MemoryDataset(110808),
                "raw_tactical": CSVDataset(filepath=str(output), save_args={"index": False}),
                "ingested": MemoryDataset(),
            })
            with patch("data_preparation.metrica.load_tracking_epts", return_value=dataset) as load:
                SequentialRunner().run(
                    pipeline_main.build_pipeline().only_nodes("atd_to_csv", "ingest"),
                    catalog,
                )
            load.assert_called_once_with(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
            result = pd.read_csv(output)
            self.assertEqual(catalog.load("ingested")["video_frame"].tolist(), [110807, 110808, 110809])
            self.assertEqual(result["video_frame"].tolist(), [110807, 110808, 110809])
            self.assertEqual(result["period_id"].tolist(), [1, 2, 2])
            np.testing.assert_allclose(result["ball_speed"], [3.5, np.nan, np.nan], equal_nan=True)
            np.testing.assert_allclose(result["p1_x_m"], [50, 30, 10])
            np.testing.assert_allclose(result["p1_y_m"], [20, 30, 25])
            np.testing.assert_allclose(result["p1_ball_dist_m"], [30, 10, np.nan], equal_nan=True)
            np.testing.assert_allclose(result["p2_ball_dist_m"], [0, np.nan, np.nan], equal_nan=True)
            self.assertEqual(result["p1_team"].tolist(), ["Home"] * 3)
            self.assertEqual(result["p2_team"].tolist(), ["Away"] * 3)
            self.assertNotIn("frame_id", result.columns)
            self.assertFalse(any(column.startswith("Unnamed:") for column in result.columns))

    def test_atd_conversion_rejects_malformed_raw_data(self):
        dataset = SimpleNamespace(to_df=lambda: pd.DataFrame({
            "frame_id": [1], "period_id": [1]
        }))
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "tracking.txt"
            raw.write_text("malformed\n")
            with patch("data_preparation.metrica.load_tracking_epts", return_value=dataset):
                with self.assertRaises(ValueError):
                    convert_atd_to_dataframe({"atd_meta": "metadata.xml", "atd_raw": str(raw)}, 110808)

    def test_smart_tagging_conversion_adds_video_frame_columns(self):
        events = pd.DataFrame({
            "timestamp": pd.to_timedelta(["00:00:01.2"]),
            "end_timestamp": pd.to_timedelta(["00:00:02.4"]),
            "label": ["Pass"],
        })
        paths = {"smart_tagging": "events.xml"}
        with patch("data_preparation.sportscode.load") as load:
            load.return_value.to_df.return_value = events
            result = convert_smart_tagging_to_dataframe(paths, 25)
        load.assert_called_once_with(paths["smart_tagging"])
        self.assertEqual(result["start_frame"].tolist(), [30])
        self.assertEqual(result["end_frame"].tolist(), [60])
        self.assertEqual(result["label"].tolist(), ["Pass"])

    def test_density_option_preserves_input_and_rejects_non_boolean(self):
        source = pd.DataFrame({
            "min_team_a_dist": [np.nan],
            "min_team_b_dist": [1.0],
            "team_a_players_1m": [0],
            "team_b_players_1m": [1],
        })
        ball_params = {"density_radii": [1]}
        result = prepare_cached_density(source, ball_params, {"correct_cached_density": False})
        pd.testing.assert_frame_equal(result, source)
        self.assertIsNot(result, source)
        corrected = prepare_cached_density(source, ball_params, {"correct_cached_density": True})
        self.assertTrue(pd.isna(corrected.loc[0, "team_a_players_1m"]))
        self.assertEqual(source.loc[0, "team_a_players_1m"], 0)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            prepare_cached_density(source, ball_params, {"correct_cached_density": "false"})

    def test_cleansing_runs_without_hmm_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cleansed.csv"
            sample = pd.read_csv(
                pipeline_main.REPO / "data/tactical/BAR-ATH_tactical_sample.csv",
                nrows=100,
            )
            second_period = sample.copy()
            second_period["period_id"] = 2
            second_period["video_frame"] += 100
            for column in second_period.columns:
                if column.endswith("_team"):
                    second_period[column] = second_period[column].replace({
                        "Barcelona": "Atletic Club", "Atletic Club": "Barcelona"
                    })
            sample = pd.concat([sample, second_period], ignore_index=True)
            sample["ball_speed"] = np.arange(len(sample), dtype=float)
            tags = pd.read_csv(
                pipeline_main.REPO / "data/tactical/BAR-ATH_tactical_smart_tagging_sample.csv"
            )
            catalog = DataCatalog({
                "raw_tactical": MemoryDataset(sample),
                "raw_smart_tag": MemoryDataset(tags),
                "params:features": MemoryDataset(pipeline_main.PARAMS),
                "params:ball_focused_features": MemoryDataset(pipeline_main.BALL_FOCUSED_PARAMS),
                "params:team_focused_features": MemoryDataset(pipeline_main.TEAM_FOCUSED_PARAMS),
                "match_info": MemoryDataset(pipeline_main.MATCH_INFO),
                "final": CSVDataset(filepath=str(output), save_args={"index": False}),
                "without_hmm": CSVDataset(filepath=str(Path(directory) / "missing.csv")),
            })
            SequentialRunner().run(pipeline_main.build_features_pipeline(), catalog)
            result = pd.read_csv(output)
            self.assertEqual(len(result), len(sample))
            self.assertIn("min_team_a_dist", result.columns)
            self.assertIn("ball_speed_calc", result.columns)
            pd.testing.assert_series_equal(result["ball_speed"], sample["ball_speed"])
            directions = pipeline_main.infer_attack_directions(
                sample, pipeline_main.BALL_FOCUSED_PARAMS
            )
            ball_x_m, _ = _ball_coordinates_in_metres(sample, "ball_x", "ball_y")
            self.assertEqual(directions[1]["Barcelona"], -directions[2]["Barcelona"])
            for period, group in result.groupby("period_id"):
                direction = directions[period]["Barcelona"]
                pd.testing.assert_series_equal(
                    group["ball_velocity_toward_team_a_goal"],
                    group["ball_velocity_x"] * direction,
                    check_names=False,
                )
                expected = (
                    ball_x_m.loc[group.index] - group["team_centroid_x_barcelona"]
                ) * direction
                np.testing.assert_allclose(
                    group["team_ball_longitudinal_distance_barcelona"], expected,
                    equal_nan=True, atol=1e-6,
                )
            self.assertFalse((Path(directory) / "missing.csv").exists())

    def test_hmm_requires_calculated_speed_instead_of_source_speed(self):
        frame = pd.DataFrame({feature: [0.0] for feature in FEATURES})
        frame["period_id"] = 1
        frame["video_frame"] = 0
        frame["sp_kind"] = None
        frame["ball_speed"] = 1.0
        self.assertIn("ball_speed_calc", FEATURES)
        self.assertNotIn("ball_speed", FEATURES)
        with self.assertRaisesRegex(ValueError, "ball_speed_calc"):
            add_hmm_possession(frame.drop(columns="ball_speed_calc"), {})

    def test_hmm_runs_from_cache_without_raw_data(self):
        rng = np.random.default_rng(42)
        centres = np.repeat([[1, 8], [8, 1], [5, 6]], 100, axis=0)
        frame = pd.DataFrame(
            np.maximum(centres + rng.normal(0, 0.15, centres.shape), 0.01),
            columns=["min_team_a_dist", "min_team_b_dist"],
        )
        frame["video_frame"] = np.arange(len(frame))
        frame["period_id"] = 1
        frame["sp_kind"] = [None] * 200 + ["free_kick"] * 100
        for feature in FEATURES:
            if feature not in frame:
                frame[feature] = 0.0
        frame["possession_advantage"] = frame["min_team_b_dist"] - frame["min_team_a_dist"]
        frame["ball_x"] = 0.5
        frame["ball_y"] = 0.5
        frame["sp_since"] = 1.0
        frame["home_attacking"] = 0
        frame["away_attacking"] = 0
        for player in range(1, 6):
            frame[f"{player}_s"] = 1.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.csv"
            output = root / "hybrid.csv"
            frame.to_csv(cache, index=False)
            original = cache.read_bytes()
            params = {
                **pipeline_main.HMM_PARAMS,
                "n_iter": 10,
                "feature_importance_path": str(root / "importance.csv"),
                "convergence_path": str(root / "convergence.csv"),
                "state_diagnostics_path": str(root / "states.csv"),
            }
            catalog = DataCatalog({
                "without_hmm": CSVDataset(filepath=str(cache)),
                "raw_tactical": CSVDataset(filepath=str(root / "missing.csv")),
                "params:ball_focused_features": MemoryDataset(pipeline_main.BALL_FOCUSED_PARAMS),
                "params:hmm": MemoryDataset(params),
                "params:hybrid": MemoryDataset(pipeline_main.HYBRID_PARAMS),
                "hybrid_evaluation": CSVDataset(filepath=str(output), save_args={"index": False}),
            })
            SequentialRunner().run(pipeline_main.build_hybrid_pipeline(), catalog)
            result = pd.read_csv(output)
            self.assertEqual(result.columns.tolist(), ["frame", "label"])
            self.assertEqual(result["frame"].tolist(), list(range(len(frame))))
            self.assertTrue(set(result["label"]).issubset({"H", "A", "D", "X"}))
            self.assertEqual(cache.read_bytes(), original)
            for name in ("importance.csv", "convergence.csv", "states.csv"):
                self.assertTrue((root / name).exists())


if __name__ == "__main__":
    unittest.main()

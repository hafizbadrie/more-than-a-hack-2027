"""Pipeline orchestrator (Kedro). Run: python data-pipeline/main.py"""
from pathlib import Path

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset
from stages.ball_focused_approach_features_extraction import (
    extract_ball_acceleration, extract_ball_direction_change,
    extract_ball_speed, extract_ball_trajectory_toward_goal,
    extract_nearest_player_ball_distance,
    extract_nearest_player_ball_relative_velocity, extract_player_density,
    extract_possession_advantage)
from stages.clean import clean
from stages.export import export
from stages.ingest import ingest
from stages.smart_tags import build_phase_flags, build_tag_features
from stages.team_focused_features_extraction import *

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

PARAMS = {"smoothing_window": 5}
BALL_FOCUSED_APPROACH_PARAMS = {
    "team_a": "Barcelona",
    "team_b": "Atletic Club",
    "distance_suffix": "_ball_dist_m",
    "team_suffix": "_team",
    "density_radii": [1.0, 2.0, 5.0],
    "frame_col": "frame_id",
    "ball_x_col": "ball_x",
    "ball_y_col": "ball_y",
    "fps": 25,
    "team_a_attack_direction": 1,
    "team_b_attack_direction": -1,
}
MATCH_INFO = {"match_id": "BAR-ATH", "home_team": "BAR", "away_team": "ATH"}


def build_pipeline():
    return pipeline(
        [
            node(
                ingest,
                "raw_tactical",
                "ingested",
                name="ingest",
            ),
            node(
                clean,
                "ingested",
                "cleaned",
                name="clean",
            ),
            # Feature 1: Nearest player-ball distance
            node(
                extract_nearest_player_ball_distance,
                ["cleaned", "params:feature_extraction_dani"],
                "features_nearest_player_ball_distance",
                name="nearest_player_ball_distance",
            ),
            # Feature 2: Possession advantage
            node(
                extract_possession_advantage,
                [
                    "features_nearest_player_ball_distance",
                    "params:feature_extraction_dani",
                ],
                "features_possession_advantage",
                name="possession_advantage",
            ),
            # Feature 3: Player density around the ball
            node(
                extract_player_density,
                [
                    "features_possession_advantage",
                    "params:feature_extraction_dani",
                ],
                "features_player_density",
                name="player_density",
            ),
            # Feature 4: Ball speed
            node(
                extract_ball_speed,
                [
                    "features_player_density",
                    "params:feature_extraction_dani",
                ],
                "features_ball_speed",
                name="ball_speed",
            ),
            # Feature 5: Ball acceleration
            node(
                extract_ball_acceleration,
                [
                    "features_ball_speed",
                    "params:feature_extraction_dani",
                ],
                "features_ball_acceleration",
                name="ball_acceleration",
            ),
            # Feature 6: Ball direction change
            node(
                extract_ball_direction_change,
                [
                    "features_ball_acceleration",
                    "params:feature_extraction_dani",
                ],
                "features_ball_direction",
                name="ball_direction_change",
            ),
            # Feature 7: Nearest player-ball relative velocity
            node(
                extract_nearest_player_ball_relative_velocity,
                [
                    "features_ball_direction",
                    "params:feature_extraction_dani",
                ],
                "features_relative_velocity",
                name="nearest_player_ball_relative_velocity",
            ),
            # Feature 8: Ball trajectory toward goal
            node(
                extract_ball_trajectory_toward_goal,
                "features_relative_velocity",
                "features_ball_trajectory",
                name="ball_trajectory_toward_goal",
            ),
            # Smart Tagging: Join event information to tracking features
            node(
                build_tag_features,
                [
                    "features_ball_trajectory",
                    "raw_smart_tag",
                    "params:features",
                    "match_info",
                ],
                "features_tagged",
                name="tag_features",
            ),
            node(build_phase_flags, 
                [
                    "features_tagged", 
                    "raw_smart_tag", 
                    "params:features", 
                    "match_info"
                ],
                "features_phased", 
                name="phase_flags"
            ),
            node(
                team_centroids,
                "features_phased",
                "with_team_centroids",
                name="hbl_features",
            ),
            node(
                team_dispersion,
                "with_team_centroids",
                "with_team_dispersion",
                name="team_dispersion",
            ),
            # Export final processed dataset
            node(
                export,
                "with_team_dispersion",
                "final",
                name="export",
            ),
        ]
    )



def build_catalog():
    catalog = DataCatalog(
        {
            "raw_tactical": CSVDataset(
                filepath=str(REPO / "data/tactical/BAR-ATH_tactical_sample.csv")
            ),
            "raw_smart_tag": CSVDataset(
                filepath=str(REPO / "data/tactical/BAR-ATH_tactical_smart_tagging_sample.csv")
            ),
            "ingested": MemoryDataset(),
            "cleaned": MemoryDataset(),
            "features_nearest_player_ball_distance": MemoryDataset(),
            "features_possession_advantage": MemoryDataset(),
            "features_player_density": MemoryDataset(),
            "features_ball_speed": MemoryDataset(),
            "features_ball_acceleration": MemoryDataset(),
            "features_ball_direction": MemoryDataset(),
            "features_relative_velocity": MemoryDataset(),
            "features_ball_trajectory": MemoryDataset(),
            "features_tagged": MemoryDataset(),
            "with_team_centroids": MemoryDataset(),
            "with_team_dispersion": MemoryDataset(),
            "final": CSVDataset(
                filepath=str(ROOT / "output/tactical_processed.csv"),
                save_args={"index": False},
            ),
            "params:features": MemoryDataset(PARAMS),
            "params:feature_extraction_dani": MemoryDataset(
                BALL_FOCUSED_APPROACH_PARAMS
            ),
            "match_info": MemoryDataset(MATCH_INFO),
        }
    )
    return catalog


def main():
    (ROOT / "output").mkdir(exist_ok=True)
    SequentialRunner().run(build_pipeline(), build_catalog())


if __name__ == "__main__":
    main()

"""Run the cleansing pipeline, the cached HMM pipeline, or both."""
import argparse
from pathlib import Path

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset
from hmm_pipeline import build_hmm_pipeline
from stages.ball_focused_features_extraction import (
    extract_ball_acceleration,
    extract_ball_direction_change,
    extract_ball_speed,
    extract_ball_trajectory_toward_goal,
    extract_nearest_player_ball_distance,
    extract_nearest_player_ball_relative_velocity,
    extract_player_density,
    extract_possession_advantage,
)
from stages.clean import clean
from stages.export import export
from stages.ingest import ingest
from stages.smart_tags import build_phase_flags, build_tag_features
from stages.team_focused_features_extraction import (
    opponent_team_proximity,
    team_average_radial_movement,
    team_centroid_velocity,
    team_centroids,
    team_dispersion,
    team_movement_coherence,
    team_position_relative_to_ball,
    team_width_and_depth,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

PARAMS = {"smoothing_window": 5}
BALL_FOCUSED_PARAMS = {
    "team_a": "Barcelona",
    "team_b": "Atletic Club",
    "distance_suffix": "_ball_dist_m",
    "team_suffix": "_team",
    "density_radii": [1.0, 2.0, 5.0],
    "frame_col": "video_frame",
    "ball_x_col": "ball_x",
    "ball_y_col": "ball_y",
    "fps": 25,
    "team_a_attack_direction": 1,
    "team_b_attack_direction": -1,
    "period_direction_multipliers": {1: 1, 2: -1},
}
TEAM_FOCUSED_PARAMS = {
    "fps": 25,
    "ball_x_col": "ball_x",
    "ball_y_col": "ball_y",
    "pitch_m": (105.3, 68.0),
    "period_direction_multipliers": {1: 1, 2: -1},
    "attack_directions": {
        "Barcelona": 1,
        "Atletic Club": -1,
    },
}
MATCH_INFO = {"match_id": "BAR-ATH", "home_team": "BAR", "away_team": "ATH"}
HMM_PARAMS = {
    "self_transition": 0.997,
    "n_iter": 100,
    "tol": 0.01,
    "min_covar": 0.01,
    "random_state": 0,
    "min_samples": 100,
    "min_anchor_samples": 10,
    "feature_importance_path": str(ROOT / "output/hmm_feature_importance.csv"),
    "convergence_path": str(ROOT / "output/hmm_convergence.csv"),
    "state_diagnostics_path": str(ROOT / "output/hmm_states.csv"),
    "smoothing_half_window": 50,
    "correct_cached_density": False,
    # Optional experiment settings are documented in the HMM section of README.md.
}
HYBRID_PARAMS = {
    "fps": 25,
    "pitch_m": (105.3, 68.0),
    "clear_distance_m": 3.0,
    "geometry_first": True,
    "still_distance_m": 0.5,
    "slow_speed_mps": 0.7,
    "min_tracked_players": 5,
    "smoothing_half_window": 50,
    "dead_min_duration_s": 2.0,
    "dead_merge_gap_s": 2.0,
    "min_possession_s": 3.0,
    "set_piece_cutoffs_s": {
        "corner_kick": 5.6,
        "free_kick": 5.2,
        "throw_in": 5.8,
        "kick_off_after_goal": 4.0,
    },
}


def build_cleansing_pipeline():
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
                ["cleaned", "params:ball_focused_features"],
                "features_nearest_player_ball_distance",
                name="nearest_player_ball_distance",
            ),
            # Feature 2: Possession advantage
            node(
                extract_possession_advantage,
                [
                    "features_nearest_player_ball_distance",
                    "params:ball_focused_features",
                ],
                "features_possession_advantage",
                name="possession_advantage",
            ),
            # Feature 3: Player density around the ball
            node(
                extract_player_density,
                [
                    "features_possession_advantage",
                    "params:ball_focused_features",
                ],
                "features_player_density",
                name="player_density",
            ),
            # Feature 4: Ball speed
            node(
                extract_ball_speed,
                [
                    "features_player_density",
                    "params:ball_focused_features",
                ],
                "features_ball_speed",
                name="ball_speed",
            ),
            # Feature 5: Ball acceleration
            node(
                extract_ball_acceleration,
                [
                    "features_ball_speed",
                    "params:ball_focused_features",
                ],
                "features_ball_acceleration",
                name="ball_acceleration",
            ),
            # Feature 6: Ball direction change
            node(
                extract_ball_direction_change,
                [
                    "features_ball_acceleration",
                    "params:ball_focused_features",
                ],
                "features_ball_direction",
                name="ball_direction_change",
            ),
            # Feature 7: Nearest player-ball relative velocity
            node(
                extract_nearest_player_ball_relative_velocity,
                [
                    "features_ball_direction",
                    "params:ball_focused_features",
                ],
                "features_relative_velocity",
                name="nearest_player_ball_relative_velocity",
            ),
            # Feature 8: Ball trajectory toward goal
            node(
                extract_ball_trajectory_toward_goal,
                [
                    "features_relative_velocity",
                    "params:ball_focused_features",
                ],
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
            node(
                team_centroid_velocity,
                ["with_team_dispersion", "params:team_focused_features"],
                "with_team_centroid_velocity",
                name="team_centroid_velocity",
            ),
            node(
                team_average_radial_movement,
                [
                    "with_team_centroid_velocity",
                    "params:team_focused_features",
                ],
                "with_team_average_radial_movement",
                name="team_average_radial_movement",
            ),
            node(
                team_width_and_depth,
                "with_team_average_radial_movement",
                "with_team_width_and_depth",
                name="team_width_and_depth",
            ),
            node(
                team_movement_coherence,
                [
                    "with_team_width_and_depth",
                    "params:team_focused_features",
                ],
                "with_team_movement_coherence",
                name="team_movement_coherence",
            ),
            node(
                opponent_team_proximity,
                [
                    "with_team_movement_coherence",
                    "params:team_focused_features",
                ],
                "with_opponent_team_proximity",
                name="opponent_team_proximity",
            ),
            node(
                team_position_relative_to_ball,
                [
                    "with_opponent_team_proximity",
                    "params:team_focused_features",
                ],
                "with_team_position_relative_to_ball",
                name="team_position_relative_to_ball",
            ),
            # Export final processed dataset
            node(
                export,
                "with_team_position_relative_to_ball",
                "final",
                name="export",
            ),
        ]
    )


def build_pipeline():
    return build_cleansing_pipeline() + build_hmm_pipeline()


def build_catalog():
    catalog = DataCatalog(
        {
            "raw_tactical": CSVDataset(
                filepath=str(REPO / "data/tactical/BAR-ATH_tactical_ds.csv")
            ),
            "raw_smart_tag": CSVDataset(
                filepath=str(REPO / "data/tactical/BAR-ATH_smart_tagging_ds.csv")
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
            "features_phased": MemoryDataset(),
            "with_team_centroids": MemoryDataset(),
            "with_team_dispersion": MemoryDataset(),
            "with_team_centroid_velocity": MemoryDataset(),
            "with_team_average_radial_movement": MemoryDataset(),
            "with_team_width_and_depth": MemoryDataset(),
            "with_team_movement_coherence": MemoryDataset(),
            "with_opponent_team_proximity": MemoryDataset(),
            "with_team_position_relative_to_ball": MemoryDataset(),
            "with_hmm_possession": MemoryDataset(),
            "density_corrected": MemoryDataset(),
            "with_hybrid_possession": MemoryDataset(),
            "final": CSVDataset(
                filepath=str(ROOT / "output/tactical_processed.csv"),
                save_args={"index": False},
            ),
            "hybrid_evaluation": CSVDataset(
                filepath=str(ROOT / "output/hybrid_results.csv"),
                save_args={"index": False},
            ),
            "params:features": MemoryDataset(PARAMS),
            "params:ball_focused_features": MemoryDataset(
                BALL_FOCUSED_PARAMS
            ),
            "params:team_focused_features": MemoryDataset(
                TEAM_FOCUSED_PARAMS
            ),
            "params:hmm": MemoryDataset(HMM_PARAMS),
            "params:hybrid": MemoryDataset(HYBRID_PARAMS),
            "match_info": MemoryDataset(MATCH_INFO),
        }
    )
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        choices=("all", "cleansing", "hmm"),
        default="all",
        help="pipeline to run (default: all; HMM reads the existing feature cache)",
    )
    args = parser.parse_args()
    builders = {
        "all": build_pipeline,
        "cleansing": build_cleansing_pipeline,
        "hmm": build_hmm_pipeline,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    SequentialRunner().run(builders[args.pipeline](), build_catalog())


if __name__ == "__main__":
    main()

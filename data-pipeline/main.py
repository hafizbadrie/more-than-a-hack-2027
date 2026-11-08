"""Pipeline orchestrator (Kedro). Run: python data-pipeline/main.py"""
from pathlib import Path

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner import SequentialRunner
from kedro_datasets.pandas import CSVDataset

from stages.clean import clean
from stages.export import export
from stages.feature_extraction_dani import extract_nearest_player_ball_distance
from stages.features import build_features
from stages.ingest import ingest

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

PARAMS = {"smoothing_window": 5}
FEATURE_EXTRACTION_DANI_PARAMS = {
    "team_a": "Barcelona",
    "team_b": "Atletic Club",
    "distance_suffix": "_ball_dist_m",
    "team_suffix": "_team",
}
MATCH_INFO = {"match_id": "BAR-ATH", "home_team": "BAR", "away_team": "ATH"}


def build_pipeline():
    return pipeline(
        [
            node(ingest, "raw_tactical", "ingested", name="ingest"),
            node(clean, "ingested", "cleaned", name="clean"),
            # One stage, several inputs: "cleaned" comes from a MemoryDataset
            # (output of the previous stage), "params:features" and "match_info"
            # come from plain dicts (MemoryDataset / direct dict assignment).
            node(
                build_features,
                ["cleaned", "params:features", "match_info"],
                "features",
                name="features",
            ),
            node(
                extract_nearest_player_ball_distance,
                ["features", "params:feature_extraction_dani"],
                "features_nearest_player_ball_distance",
                name="nearest_player_ball_distance",
            ),
            node(
                export,
                "features_nearest_player_ball_distance",
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
            "features": MemoryDataset(),
            "features_nearest_player_ball_distance": MemoryDataset(),
            "final": CSVDataset(
                filepath=str(ROOT / "output/tactical_processed.csv"),
                save_args={"index": False},
            ),
            "params:features": MemoryDataset(PARAMS),
            "params:feature_extraction_dani": MemoryDataset(
                FEATURE_EXTRACTION_DANI_PARAMS
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

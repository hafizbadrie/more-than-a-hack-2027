"""Kedro pipeline for cached HMM and hybrid possession predictions."""

from kedro.pipeline import node, pipeline

from stages.ball_focused_features_extraction import mask_unobserved_player_density
from stages.export import export_hybrid_evaluation
from stages.hmm_possession import add_hmm_possession
from stages.hybrid_possession import add_hybrid_possession


def prepare_cached_density(df, ball_params, hmm_params):
    correct = hmm_params["correct_cached_density"]
    if not isinstance(correct, bool):
        raise ValueError("correct_cached_density must be a boolean")
    if correct:
        return mask_unobserved_player_density(df, ball_params)
    return df.copy()


def build_hmm_pipeline():
    return pipeline(
        [
            node(
                prepare_cached_density,
                ["final", "params:ball_focused_features", "params:hmm"],
                "density_corrected",
                name="correct_cached_density",
            ),
            node(
                add_hmm_possession,
                ["density_corrected", "params:hmm"],
                "with_hmm_possession",
                name="hmm_possession",
            ),
            node(
                add_hybrid_possession,
                ["with_hmm_possession", "params:hybrid"],
                "with_hybrid_possession",
                name="hybrid_possession",
            ),
            node(
                export_hybrid_evaluation,
                "with_hybrid_possession",
                "hybrid_evaluation",
                name="export_hybrid_evaluation",
            ),
        ]
    )

"""Stage 4: export - final shaping of the pipeline output."""
import pandas as pd


def export(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

def export_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """
        I want to export the data in a structure suitable for evaluation.
        It only needs 2 columns: which is the `video_frame` (rename the column as `frame`) 
        and hmm_possession (rename the column as `label`)
    """
    return df[["video_frame", "hmm_possession"]].rename(columns={"video_frame": "frame", "hmm_possession": "label"})


def export_hybrid_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    return df[["video_frame", "hybrid_possession"]].rename(
        columns={"video_frame": "frame", "hybrid_possession": "label"}
    )
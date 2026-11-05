"""Stage 4: export - final shaping of the pipeline output."""
import pandas as pd


def export(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

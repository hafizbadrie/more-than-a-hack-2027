"""Stage 2: clean - drop unusable rows and duplicates."""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["period_id", "frame_id"])
    # Drop columns that are entirely empty (e.g. players not on the pitch)
    return df.dropna(axis=1, how="all").reset_index(drop=True)

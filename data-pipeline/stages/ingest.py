"""Stage 1: ingest - basic typing and sorting of the raw tactical frames."""
import pandas as pd


def ingest(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp"] = pd.to_timedelta(df["timestamp"])
    return df.sort_values(["period_id", "frame_id"]).reset_index(drop=True)

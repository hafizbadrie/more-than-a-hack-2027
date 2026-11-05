"""Stage 3: features - derive per-frame features. Extend with your own."""
import pandas as pd


def build_features(df: pd.DataFrame, params: dict, match_info: dict) -> pd.DataFrame:
    out = df.copy()
    out["match_id"] = match_info["match_id"]
    window = params.get("smoothing_window", 5)
    if "ball_speed" in out:
        out["ball_speed_smooth"] = (
            out["ball_speed"].rolling(window, min_periods=1, center=True).mean()
        )
    out["seconds"] = out["timestamp"].dt.total_seconds()
    return out

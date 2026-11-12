"""Helpers for period-dependent attacking directions."""
import numpy as np
import pandas as pd


def period_direction_multipliers(
    df: pd.DataFrame,
    params: dict,
) -> pd.Series:
    """Return each frame's configured attack-direction multiplier."""
    configured = params.get("period_direction_multipliers")
    if configured is None:
        return pd.Series(1, index=df.index, dtype=np.int8)
    if "period_id" not in df:
        raise ValueError(
            "period_direction_multipliers requires a period_id column"
        )
    if not isinstance(configured, dict) or not configured:
        raise ValueError("period_direction_multipliers must be a non-empty mapping")
    if any(
        isinstance(multiplier, bool) or multiplier not in (-1, 1)
        for multiplier in configured.values()
    ):
        raise ValueError("period direction multipliers must be either -1 or +1")

    multipliers = df["period_id"].map(configured)
    if multipliers.isna().any():
        missing = df.loc[multipliers.isna(), "period_id"].drop_duplicates().tolist()
        raise ValueError(
            f"Missing period direction multipliers for period_id values: {missing}"
        )
    return multipliers.astype(np.int8)

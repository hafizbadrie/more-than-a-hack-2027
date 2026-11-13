"""Helpers for period-dependent attacking directions."""
import numpy as np
import pandas as pd


def infer_attack_directions(df: pd.DataFrame, params: dict) -> dict:
    """Infer opposing directions from the mean per-frame team centroid in each period."""
    if df.empty or "period_id" not in df or df["period_id"].isna().any():
        raise ValueError("Attack direction inference requires non-missing period_id values")
    teams = (params["team_a"], params["team_b"])
    if teams[0] == teams[1]:
        raise ValueError("Attack direction inference requires two distinct teams")
    slots = [
        column[:-4] for column in df.columns
        if column.endswith("_x_m") and column[:-4] + "_team" in df.columns
    ]
    if not slots:
        raise ValueError("Attack direction inference requires player x and team columns")

    means = {}
    for team in teams:
        total = pd.Series(0.0, index=df.index)
        count = pd.Series(0, index=df.index)
        for slot in slots:
            x = pd.to_numeric(df[f"{slot}_x_m"], errors="coerce")
            valid = df[f"{slot}_team"].eq(team) & np.isfinite(x)
            total += x.where(valid, 0)
            count += valid.astype(int)
        centroid = total.div(count.where(count > 0))
        means[team] = centroid.groupby(df["period_id"]).mean()

    directions = {}
    for period in df["period_id"].unique():
        team_a_x, team_b_x = (means[team].loc[period] for team in teams)
        for team, x in zip(teams, (team_a_x, team_b_x)):
            if not np.isfinite(x):
                raise ValueError(
                    f"Cannot infer attack direction for team '{team}' in period {period}: "
                    "no finite player positions"
                )
        if team_a_x == team_b_x:
            raise ValueError(f"Cannot infer attack directions in period {period}: equal centroids")
        direction = 1 if team_a_x < team_b_x else -1
        directions[period] = {teams[0]: direction, teams[1]: -direction}
    return directions


def attack_direction_for_team(
    df: pd.DataFrame, directions: dict, team: str
) -> pd.Series:
    """Select and validate a team's direction for each frame's period."""
    if "period_id" not in df:
        raise ValueError("Per-period attack directions require a period_id column")
    values = {}
    for period in df["period_id"].unique():
        if period not in directions or team not in directions[period]:
            raise ValueError(f"Missing attack direction for team '{team}' in period {period}")
        direction = directions[period][team]
        if isinstance(direction, bool) or direction not in (-1, 1):
            raise ValueError(f"Invalid attack direction for team '{team}' in period {period}")
        values[period] = direction
    result = df["period_id"].map(values)
    if result.isna().any():
        raise ValueError("Per-period attack directions require non-missing period_id values")
    return result.astype(np.int8)


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

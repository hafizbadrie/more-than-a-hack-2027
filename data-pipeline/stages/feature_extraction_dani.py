"""Feature extraction stages contributed by Dani."""
import pandas as pd


PARAMS = {
    "team_a": "Barcelona",
    "team_b": "Atletic Club",
    "distance_suffix": "_ball_dist_m",
    "team_suffix": "_team",
}


def extract_nearest_player_ball_distance(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """Add each team's nearest tracked player-to-ball distance per frame."""
    out = df.copy()
    distance_columns = [
        column
        for column in out.columns
        if column.endswith(params["distance_suffix"])
        and f"{column.removesuffix(params['distance_suffix'])}{params['team_suffix']}"
        in out.columns
    ]

    def nearest_distance(team: str) -> pd.Series:
        distances = {
            column: out[column].where(
                out[
                    f"{column.removesuffix(params['distance_suffix'])}"
                    f"{params['team_suffix']}"
                ].eq(team)
            )
            for column in distance_columns
        }
        return pd.DataFrame(distances, index=out.index).min(axis=1)

    out["min_team_a_dist"] = nearest_distance(params["team_a"])
    out["min_team_b_dist"] = nearest_distance(params["team_b"])
    out["nearest_player_dist"] = out[
        ["min_team_a_dist", "min_team_b_dist"]
    ].min(axis=1)
    out["nearest_player_team"] = pd.NA

    team_a_is_nearest = out["min_team_a_dist"].notna() & (
        out["min_team_b_dist"].isna()
        | (out["min_team_a_dist"] <= out["min_team_b_dist"])
    )
    team_b_is_nearest = out["min_team_b_dist"].notna() & (
        out["min_team_a_dist"].isna()
        | (out["min_team_b_dist"] < out["min_team_a_dist"])
    )
    out.loc[team_a_is_nearest, "nearest_player_team"] = params["team_a"]
    out.loc[team_b_is_nearest, "nearest_player_team"] = params["team_b"]
    return out

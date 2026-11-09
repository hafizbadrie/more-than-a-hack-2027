"""Feature extraction stages contributed by Dani."""
import pandas as pd


PARAMS = {
    "team_a": "Barcelona",
    "team_b": "Atletic Club",
    "distance_suffix": "_ball_dist_m",
    "team_suffix": "_team",
    "density_radii": [1.0, 2.0, 5.0],
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


def extract_possession_advantage(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """Add the signed closest-player distance gap from team A's perspective."""
    out = df.copy()
    out["possession_advantage"] = (
        out["min_team_b_dist"] - out["min_team_a_dist"]
    )
    return out

def extract_player_density(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 3: Count the number of players around the ball.

    Calculates player density for multiple radii (default: 1m, 2m, 5m).

    Output:
        team_a_players_1m
        team_b_players_1m
        team_a_players_2m
        team_b_players_2m
        team_a_players_5m
        team_b_players_5m
    """
    out = df.copy()

    distance_suffix = params["distance_suffix"]
    team_suffix = params["team_suffix"]

    teams = {
        "team_a": params["team_a"],
        "team_b": params["team_b"],
    }

    radii = params["density_radii"]

    # Identify player distance columns and their corresponding team columns
    player_columns = {}

    for column in out.columns:
        if not column.endswith(distance_suffix):
            continue

        team_column = (
            f"{column.removesuffix(distance_suffix)}{team_suffix}"
        )

        if team_column in out.columns:
            player_columns[column] = team_column

    if not player_columns:
        raise ValueError(
            "No valid player-ball distance columns found. "
            "Cannot calculate player density."
        )

    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("density_radii must contain positive values.")

    # Calculate density for each team and radius
    for team_key, team_name in teams.items():

        team_distances = {}

        for distance_column, team_column in player_columns.items():

            team_distances[distance_column] = (
                pd.to_numeric(
                    out[distance_column], errors="coerce"
                )
                .where(out[team_column].eq(team_name))
            )

        team_distances_df = pd.DataFrame(
            team_distances,
            index=out.index
        )

        for radius in radii:

            count = (
                team_distances_df
                .le(radius)
                .fillna(False)
                .sum(axis=1)
            )

            out[f"{team_key}_players_{radius:g}m"] = count

    return out
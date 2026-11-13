"""Feature extraction stages contributed by Dani."""
import pandas as pd
import numpy as np
from stages.direction import attack_direction_for_team, period_direction_multipliers
from stages.team_focused_features_extraction import _ball_coordinates_in_metres

PARAMS = {
    "team_a": "Barcelona",
    "team_b": "Atletic Club",
    "distance_suffix": "_ball_dist_m",
    "team_suffix": "_team",
    # Feature 3: Player density
    "density_radii": [1.0, 2.0, 5.0],

    # Feature 4: Ball speed
    "frame_col": "video_frame",
    "ball_x_col": "ball_x",
    "ball_y_col": "ball_y",
    "fps": 25,
    "lag": 1,
    "max_gap_frames": 1,

    # Feature 8: Attacking directions
    "team_a_attack_direction": 1,
    "team_b_attack_direction": -1,
}


def extract_nearest_player_ball_distance(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """Add nearest player distances; leave the overall team missing for ties."""
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
        | (out["min_team_a_dist"] < out["min_team_b_dist"])
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

            out[f"{team_key}_players_{radius:g}m"] = count.where(
                team_distances_df.notna().any(axis=1)
            )

    return out


def mask_unobserved_player_density(df: pd.DataFrame, params: dict = PARAMS) -> pd.DataFrame:
    """Correct cached density values where no player-ball distance was observed."""
    required = ["min_team_a_dist", "min_team_b_dist"]
    required += [
        f"team_{team}_players_{radius:g}m"
        for team in ("a", "b") for radius in params["density_radii"]
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Cannot correct cached density; missing columns: {missing}")
    out = df.copy()
    for team in ("a", "b"):
        columns = [f"team_{team}_players_{radius:g}m" for radius in params["density_radii"]]
        out.loc[out[f"min_team_{team}_dist"].isna(), columns] = np.nan
    return out



def _motion_settings(params: dict) -> tuple[int, int, float]:
    lag = params.get("lag", 1)
    max_gap = params.get("max_gap_frames", 1)
    fps = params["fps"]
    for name, value in (("lag", lag), ("max_gap_frames", max_gap)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be greater than zero and finite.")
    return lag, max_gap, fps


def _centered_difference(
    frames: pd.DataFrame, values: pd.DataFrame, frame_col: str, lag: int, max_gap: int
) -> pd.DataFrame:
    """Return endpoint differences, rejecting missing endpoints and excessive gaps."""
    result = pd.DataFrame(np.nan, index=frames.index, columns=values.columns)
    groups = (
        frames.groupby("period_id", sort=False, dropna=False).indices.values()
        if "period_id" in frames.columns else [np.arange(len(frames))]
    )
    for positions in groups:
        frame_numbers = frames.iloc[positions][frame_col].to_numpy(dtype=float, na_value=np.nan)
        data = values.iloc[positions].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(data).all(axis=1) & np.isfinite(frame_numbers)
        observed_frames = frame_numbers[valid]
        if not len(observed_frames):
            continue
        observed_data = data[valid]
        centres = frame_numbers
        left = np.searchsorted(observed_frames, centres - lag)
        right = np.searchsorted(observed_frames, centres + lag)
        in_bounds = (left < len(observed_frames)) & (right < len(observed_frames))
        left_safe = np.minimum(left, len(observed_frames) - 1)
        right_safe = np.minimum(right, len(observed_frames) - 1)
        bad_gaps = np.r_[0, np.cumsum(np.diff(observed_frames) > max_gap)]
        usable = (
            valid & in_bounds
            & (observed_frames[left_safe] == centres - lag)
            & (observed_frames[right_safe] == centres + lag)
            & (bad_gaps[right_safe] == bad_gaps[left_safe])
        )
        result.iloc[positions[usable]] = (
            observed_data[right_safe[usable]] - observed_data[left_safe[usable]]
        )
    return result


def extract_ball_speed(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 4: Calculate centred ball displacement and speed in metres.

    Calculates:
        ball_dx           : Horizontal displacement (m)
        ball_dy           : Vertical displacement (m)
        ball_displacement : Euclidean displacement (m)
        ball_speed_calc   : Average ball speed (m/s)

    Notes:
        - Uses observations at t-lag and t+lag within the same period.
        - Requires a current observation and no gap longer than max_gap_frames.
        - Converts normalized ball coordinates using player coordinate pairs.
    """

    out = df.copy()

    frame_col = params["frame_col"]
    ball_x = params["ball_x_col"]
    ball_y = params["ball_y_col"]
    lag, max_gap, fps = _motion_settings(params)
    frame_cols = (["period_id"] if "period_id" in out.columns else []) + [frame_col]

    required_columns = [frame_col, ball_x, ball_y]

    missing_columns = [
        col for col in required_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required ball tracking columns: {missing_columns}"
        )

    # Extract ball coordinates from the original dataframe.
    # Keep the first valid X/Y coordinate pair for each frame.
    ball_x_m, ball_y_m = _ball_coordinates_in_metres(out, ball_x, ball_y)
    ball_positions = out[frame_cols].copy()
    ball_positions["ball_x_m"] = ball_x_m
    ball_positions["ball_y_m"] = ball_y_m

    ball_positions[frame_col] = pd.to_numeric(ball_positions[frame_col], errors="raise")
    ball_positions["_missing_ball"] = ball_positions[["ball_x_m", "ball_y_m"]].isna().any(axis=1)
    ball_positions = (
        ball_positions
        .sort_values("_missing_ball", kind="stable")
        .drop_duplicates(subset=frame_cols, keep="first")
        .sort_values(frame_cols)
        .reset_index(drop=True)
    )

    differences = _centered_difference(
        ball_positions[frame_cols], ball_positions[["ball_x_m", "ball_y_m"]],
        frame_col, lag, max_gap,
    )
    ball_positions["ball_dx"] = differences["ball_x_m"]
    ball_positions["ball_dy"] = differences["ball_y_m"]

    # Euclidean displacement
    ball_positions["ball_displacement"] = np.hypot(
        ball_positions["ball_dx"],
        ball_positions["ball_dy"],
    )

    ball_positions["ball_speed_calc"] = ball_positions["ball_displacement"] * fps / (2 * lag)

    # Select extracted features
    result = ball_positions[
        frame_cols + [
            "ball_dx",
            "ball_dy",
            "ball_displacement",
            "ball_speed_calc",
        ]
    ]

    # Drop pre-existing feature columns (e.g. already present in the raw
    # data) so the merge doesn't create _x/_y suffixed duplicates.
    out = out.drop(
        columns=[c for c in result.columns if c not in frame_cols and c in out.columns]
    )

    # Merge extracted features back into the original dataframe.
    # Missing ball observations retain NaN values.
    out = out.merge(
        result,
        on=frame_cols,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


def extract_ball_acceleration(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 5: Centred rate of change of ball speed (m/s²).
    """

    out = df.copy()

    frame_col = params["frame_col"]
    lag, max_gap, fps = _motion_settings(params)
    frame_cols = (["period_id"] if "period_id" in out.columns else []) + [frame_col]

    required_columns = [frame_col, "ball_speed_calc"]

    missing_columns = [
        col for col in required_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for ball acceleration: "
            f"{missing_columns}. Run extract_ball_speed() first."
        )

    speed_data = (
        out[frame_cols + ["ball_speed_calc"]]
        .drop_duplicates(subset=frame_cols, keep="first")
        .sort_values(frame_cols)
        .reset_index(drop=True)
    )

    differences = _centered_difference(
        speed_data[frame_cols], speed_data[["ball_speed_calc"]], frame_col, lag, max_gap
    )
    speed_data["ball_acceleration"] = differences["ball_speed_calc"] * fps / (2 * lag)

    # Keep extracted feature
    result = speed_data[
        frame_cols + ["ball_acceleration"]
    ]

    # Merge into original dataframe
    out = out.drop(columns=["ball_acceleration"], errors="ignore").merge(
        result,
        on=frame_cols,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


def extract_ball_direction_change(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 6: Calculate ball movement direction and trajectory changes.

    Direction change compares the movement directions at t-lag and t+lag.

    Calculates:
        ball_direction_deg:
            Ball movement direction relative to the positive X-axis (degrees).

        ball_direction_change_deg:
            Signed angular change between the endpoint movement vectors
            (degrees, range -180 to +180).

        ball_direction_change_abs_deg:
            Absolute angular change between the endpoint movement vectors
            (degrees, range 0 to 180).

    Notes:
        - Requires ball_dx and ball_dy from extract_ball_speed().
        - Requires an observed current direction and limits intervening gaps.
        - Zero-displacement observations have undefined direction.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    frame_col = params["frame_col"]
    lag, max_gap, _ = _motion_settings(params)
    frame_cols = (["period_id"] if "period_id" in out.columns else []) + [frame_col]

    required_columns = [
        frame_col,
        "ball_dx",
        "ball_dy",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for ball direction: "
            f"{missing_columns}. Run extract_ball_speed() first."
        )

    movement = (
        out[frame_cols + ["ball_dx", "ball_dy"]]
        .drop_duplicates(subset=frame_cols, keep="first")
        .sort_values(frame_cols)
        .reset_index(drop=True)
    )

    # Calculate movement direction relative to positive X-axis
    movement["ball_direction_deg"] = np.degrees(
        np.arctan2(
            movement["ball_dy"],
            movement["ball_dx"],
        )
    )

    # A zero-displacement vector has no defined direction
    zero_movement = (
        movement["ball_dx"].eq(0)
        & movement["ball_dy"].eq(0)
    )

    movement.loc[
        zero_movement, "ball_direction_deg"
    ] = np.nan

    changes = _centered_difference(
        movement[frame_cols], movement[["ball_direction_deg"]], frame_col, lag, max_gap
    )
    angle_delta = np.radians(changes["ball_direction_deg"])
    movement["ball_direction_change_deg"] = np.degrees(
        np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
    )

    # Absolute direction change
    movement["ball_direction_change_abs_deg"] = (
        movement["ball_direction_change_deg"].abs()
    )

    # Keep extracted features
    result = movement[
        frame_cols + [
            "ball_direction_deg",
            "ball_direction_change_deg",
            "ball_direction_change_abs_deg",
        ]
    ]

    # Merge into original dataframe
    out = out.drop(columns=[
        "ball_direction_deg", "ball_direction_change_deg", "ball_direction_change_abs_deg"
    ], errors="ignore").merge(
        result,
        on=frame_cols,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


def extract_nearest_player_ball_relative_velocity(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 7: Calculate the relative velocity between each player and the ball.

    Calculates the centred rate of change in player-to-ball distance.

    Positive value:
        Player is approaching the ball.

    Negative value:
        Player is moving away from the ball.

    Zero:
        Player maintains approximately the same distance.

    Outputs:
        team_a_nearest_player_ball_relative_velocity
        team_b_nearest_player_ball_relative_velocity
        nearest_player_ball_relative_velocity
        nearest_player_ball_relative_velocity_team

    Notes:
        - Reuses existing player-to-ball distance columns.
        - Requires distances at t-lag, t and t+lag and limits intervening gaps.
        - Calculates changes independently within each period when available.
        - Selects the relative velocity of the nearest player in each team.
        - Leaves the overall team and velocity missing for ties between teams.
        - Does not interpolate missing distances.
        - Does not estimate player possession.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    frame_col = params["frame_col"]
    lag, max_gap, fps = _motion_settings(params)
    frame_cols = (["period_id"] if "period_id" in out.columns else []) + [frame_col]

    distance_suffix = params["distance_suffix"]
    team_suffix = params["team_suffix"]

    required_columns = [frame_col]

    missing_columns = [
        col for col in required_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

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
            "No valid player-ball distance columns found."
        )

    # Create unique, chronologically sorted frame index
    frame_data = out[frame_cols].copy()
    frame_data[frame_col] = pd.to_numeric(frame_data[frame_col], errors="coerce")
    frame_data = (
        frame_data.dropna(subset=[frame_col])
        .drop_duplicates(subset=frame_cols)
        .sort_values(frame_cols)
        .reset_index(drop=True)
    )
    frame_index = frame_data.set_index(frame_cols).index

    distance_data = {}
    velocity_data = {}

    # Calculate relative velocity for every tracked player
    for distance_column in player_columns:

        player_distance = (
            out[frame_cols + [distance_column]]
            .copy()
        )

        player_distance[frame_col] = pd.to_numeric(
            player_distance[frame_col], errors="coerce"
        )
        player_distance[distance_column] = pd.to_numeric(
            player_distance[distance_column],
            errors="coerce"
        )

        player_distance = (
            player_distance
            .drop_duplicates(subset=frame_cols, keep="first")
            .set_index(frame_cols)[distance_column]
            .reindex(frame_index)
        )

        distance_change = _centered_difference(
            frame_data[frame_cols],
            player_distance.reset_index(drop=True).to_frame(),
            frame_col, lag, max_gap,
        ).iloc[:, 0]
        relative_velocity = pd.Series(
            (-distance_change * fps / (2 * lag)).to_numpy(), index=frame_index
        )

        distance_data[distance_column] = player_distance
        velocity_data[distance_column] = relative_velocity

    distance_matrix = pd.DataFrame(
        distance_data, index=frame_index
    )

    velocity_matrix = pd.DataFrame(
        velocity_data, index=frame_index
    )

    # Select the nearest player's relative velocity for each team
    team_results = {}

    for team_key, team_name in [
        ("team_a", params["team_a"]),
        ("team_b", params["team_b"]),
    ]:

        team_columns = [
            distance_column
            for distance_column, team_column in player_columns.items()
            if out[team_column].eq(team_name).any()
        ]

        if not team_columns:
            raise ValueError(
                f"No player distance columns found for {team_name}."
            )

        team_distances = distance_matrix[team_columns]
        team_velocities = velocity_matrix[team_columns]

        # Find nearest player in each frame
        distance_values = team_distances.to_numpy()

        valid = np.isfinite(distance_values)

        nearest_indices = np.where(
            valid.any(axis=1),
            np.where(valid, distance_values, np.inf).argmin(axis=1),
            -1,
        )

        # Select the relative velocity of that same player
        velocity_values = team_velocities.to_numpy()

        nearest_velocity = np.full(len(frame_index), np.nan)

        valid_rows = nearest_indices >= 0

        nearest_velocity[valid_rows] = velocity_values[
            np.arange(len(frame_index))[valid_rows],
            nearest_indices[valid_rows],
        ]

        team_results[
            f"{team_key}_nearest_player_ball_relative_velocity"
        ] = nearest_velocity

    # Add team-specific results
    for column, values in team_results.items():
        frame_data[column] = values

    # Determine which team's player is closest to the ball
    team_a_dist = distance_matrix[
        [
            col for col, team_col in player_columns.items()
            if out[team_col].eq(params["team_a"]).any()
        ]
    ].min(axis=1)

    team_b_dist = distance_matrix[
        [
            col for col, team_col in player_columns.items()
            if out[team_col].eq(params["team_b"]).any()
        ]
    ].min(axis=1)

    team_a_nearest = (
        team_a_dist.notna()
        & (
            team_b_dist.isna()
            | (team_a_dist < team_b_dist)
        )
    )

    team_b_nearest = (
        team_b_dist.notna()
        & (
            team_a_dist.isna()
            | (team_b_dist < team_a_dist)
        )
    )

    frame_data["nearest_player_ball_relative_velocity"] = np.nan
    frame_data["nearest_player_ball_relative_velocity_team"] = pd.NA

    frame_data.loc[
        team_a_nearest.to_numpy(),
        "nearest_player_ball_relative_velocity",
    ] = frame_data.loc[
        team_a_nearest.to_numpy(),
        "team_a_nearest_player_ball_relative_velocity",
    ].to_numpy()

    frame_data.loc[
        team_b_nearest.to_numpy(),
        "nearest_player_ball_relative_velocity",
    ] = frame_data.loc[
        team_b_nearest.to_numpy(),
        "team_b_nearest_player_ball_relative_velocity",
    ].to_numpy()

    frame_data.loc[
        team_a_nearest.to_numpy(),
        "nearest_player_ball_relative_velocity_team",
    ] = params["team_a"]

    frame_data.loc[
        team_b_nearest.to_numpy(),
        "nearest_player_ball_relative_velocity_team",
    ] = params["team_b"]

    # Merge extracted features back into original dataframe
    result = frame_data.reset_index(drop=True)

    out = out.drop(columns=[
        "team_a_nearest_player_ball_relative_velocity",
        "team_b_nearest_player_ball_relative_velocity",
        "nearest_player_ball_relative_velocity",
        "nearest_player_ball_relative_velocity_team",
    ], errors="ignore").merge(
        result,
        on=frame_cols,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


def extract_ball_trajectory_toward_goal(
    df: pd.DataFrame, params: dict = PARAMS, attack_directions: dict | None = None
) -> pd.DataFrame:
    """
    Feature 8: Calculate ball movement relative to each team's
    attacking goal direction.

    Horizontal ball velocity is projected onto each team's
    attacking direction.

    Attack direction:
        +1 : Attacking toward increasing X coordinates.
        -1 : Attacking toward decreasing X coordinates.

    Calculates:
        ball_velocity_x:
            Horizontal ball velocity (m/s).

        ball_velocity_toward_team_a_goal:
            Ball velocity projected onto Team A's attacking direction.

        ball_velocity_toward_team_b_goal:
            Ball velocity projected onto Team B's attacking direction.

        ball_moving_toward_team_a_goal:
            Whether the ball is moving toward Team A's attacking goal.

        ball_moving_toward_team_b_goal:
            Whether the ball is moving toward Team B's attacking goal.

    Notes:
        - Requires ball_dx, ball_displacement and ball_speed_calc.
        - Uses the actual average ball speed between valid observations.
        - Does not assume the nearest player is the possession team.
        - Calculates both teams' attacking perspectives independently.
        - Per-period directions take precedence over legacy direction parameters.
        - Does not interpolate missing ball observations.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    required_columns = [
        "ball_dx",
        "ball_displacement",
        "ball_speed_calc",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for ball trajectory: "
            f"{missing_columns}. Run extract_ball_speed() first."
        )

    if attack_directions is not None:
        team_a_direction = attack_direction_for_team(out, attack_directions, params["team_a"])
        team_b_direction = attack_direction_for_team(out, attack_directions, params["team_b"])
        if (team_a_direction == team_b_direction).any():
            raise ValueError("The two opposing teams must attack in opposite directions.")
    else:
        team_a_direction = params["team_a_attack_direction"]
        team_b_direction = params["team_b_attack_direction"]
        if team_a_direction not in (-1, 1):
            raise ValueError("team_a_attack_direction must be either -1 or +1.")
        if team_b_direction not in (-1, 1):
            raise ValueError("team_b_attack_direction must be either -1 or +1.")
        if team_a_direction == team_b_direction:
            raise ValueError("The two opposing teams should attack in opposite directions during the same half.")
        direction_multiplier = period_direction_multipliers(out, params)
        team_a_direction = team_a_direction * direction_multiplier
        team_b_direction = team_b_direction * direction_multiplier

    # Calculate horizontal ball velocity.
    # ball_speed_calc contains the magnitude of the velocity.
    # ball_dx / ball_displacement provides its horizontal direction.
    out["ball_velocity_x"] = (
        out["ball_speed_calc"]
        * out["ball_dx"]
        / out["ball_displacement"].where(
            out["ball_displacement"] > 0
        )
    )

    # Project ball velocity onto Team A's attacking direction
    out["ball_velocity_toward_team_a_goal"] = (
        out["ball_velocity_x"] * team_a_direction
    )

    # Project ball velocity onto Team B's attacking direction
    out["ball_velocity_toward_team_b_goal"] = (
        out["ball_velocity_x"] * team_b_direction
    )

    # Determine whether the ball moves toward each team's goal.
    # Preserve NaN when ball velocity is unavailable.
    out["ball_moving_toward_team_a_goal"] = (
        out["ball_velocity_toward_team_a_goal"]
        .gt(0)
        .where(
            out["ball_velocity_toward_team_a_goal"].notna()
        )
    )

    out["ball_moving_toward_team_b_goal"] = (
        out["ball_velocity_toward_team_b_goal"]
        .gt(0)
        .where(
            out["ball_velocity_toward_team_b_goal"].notna()
        )
    )

    return out

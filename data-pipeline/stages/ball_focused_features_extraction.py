"""Feature extraction stages contributed by Dani."""
import pandas as pd
import numpy as np

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

    # Feature 8: Attacking directions
    "team_a_attack_direction": 1,
    "team_b_attack_direction": -1,
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



def extract_ball_speed(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 4: Calculate ball movement and speed between available observations.

    Calculates ball displacement and average speed using consecutive
    valid ball observations, based on their actual frame-number difference.

    Missing frames or ball detections are not interpolated or estimated.
    The previous valid observation is used when intermediate observations
    have missing ball coordinates.

    Calculates:
        ball_dx           : Horizontal displacement (m)
        ball_dy           : Vertical displacement (m)
        ball_displacement : Euclidean displacement (m)
        ball_speed        : Average ball speed (m/s)

    Notes:
        - Uses only observations where both ball coordinates are available.
        - Uses the previous valid observation, not necessarily the
          immediately preceding frame.
        - Accounts for missing or skipped frames using actual elapsed time.
        - Does not interpolate missing coordinates or smooth ball movement.
        - The first valid observation has NaN displacement and speed.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    frame_col = params["frame_col"]
    ball_x = params["ball_x_col"]
    ball_y = params["ball_y_col"]
    fps = params["fps"]

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

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
    ball_positions = (
        out[[frame_col, ball_x, ball_y]]
        .copy()
    )

    ball_positions[ball_x] = pd.to_numeric(
        ball_positions[ball_x], errors="coerce"
    )

    ball_positions[ball_y] = pd.to_numeric(
        ball_positions[ball_y], errors="coerce"
    )

    # Keep only observations with both coordinates available
    ball_positions = ball_positions.dropna(
        subset=[ball_x, ball_y]
    )

    # One valid ball observation per frame
    ball_positions = (
        ball_positions
        .drop_duplicates(subset=[frame_col], keep="first")
        .sort_values(frame_col)
        .copy()
    )

    # Frame-to-frame displacement between consecutive valid observations
    ball_positions["ball_dx"] = (
        ball_positions[ball_x].diff()
    )

    ball_positions["ball_dy"] = (
        ball_positions[ball_y].diff()
    )

    # Euclidean displacement
    ball_positions["ball_displacement"] = np.hypot(
        ball_positions["ball_dx"],
        ball_positions["ball_dy"],
    )

    # Actual elapsed time between valid observations
    elapsed_time = (
        ball_positions[frame_col].diff() / fps
    )

    # Average ball speed (m/s)
    ball_positions["ball_speed"] = (
        ball_positions["ball_displacement"] / elapsed_time
    ).where(elapsed_time > 0)

    # Select extracted features
    result = ball_positions[
        [
            frame_col,
            "ball_dx",
            "ball_dy",
            "ball_displacement",
            "ball_speed",
        ]
    ]

    # Drop pre-existing feature columns (e.g. already present in the raw
    # data) so the merge doesn't create _x/_y suffixed duplicates.
    out = out.drop(
        columns=[c for c in result.columns if c != frame_col and c in out.columns]
    )

    # Merge extracted features back into the original dataframe.
    # Missing ball observations retain NaN values.
    out = out.merge(
        result,
        on=frame_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


def extract_ball_acceleration(
    df: pd.DataFrame, params: dict = PARAMS
) -> pd.DataFrame:
    """
    Feature 5: Calculate ball acceleration from consecutive valid speed
    observations.

    Acceleration is calculated from the change in average ball speed
    divided by the actual elapsed time between observations.

    Calculates:
        ball_acceleration : Change in ball speed over time (m/s²)

    Notes:
        - Requires ball_speed from extract_ball_speed().
        - Uses actual frame-number differences to calculate elapsed time.
        - Uses only available valid speed observations.
        - Does not interpolate missing ball positions or speeds.
        - The first valid speed observation has NaN acceleration.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    frame_col = params["frame_col"]
    fps = params["fps"]

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    required_columns = [frame_col, "ball_speed"]

    missing_columns = [
        col for col in required_columns
        if col not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for ball acceleration: "
            f"{missing_columns}. Run extract_ball_speed() first."
        )

    # Extract valid speed observations
    speed_data = (
        out[[frame_col, "ball_speed"]]
        .dropna(subset=["ball_speed"])
        .drop_duplicates(subset=[frame_col], keep="first")
        .sort_values(frame_col)
        .copy()
    )

    # Calculate actual elapsed time between speed observations
    elapsed_time = speed_data[frame_col].diff() / fps

    # Calculate change in speed
    speed_change = speed_data["ball_speed"].diff()

    # Calculate acceleration (m/s²)
    speed_data["ball_acceleration"] = (
        speed_change / elapsed_time
    ).where(elapsed_time > 0)

    # Keep extracted feature
    result = speed_data[
        [frame_col, "ball_acceleration"]
    ]

    # Merge into original dataframe
    out = out.merge(
        result,
        on=frame_col,
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

    Direction is calculated from consecutive valid ball displacement
    vectors using their horizontal and vertical components.

    Calculates:
        ball_direction_deg:
            Ball movement direction relative to the positive X-axis (degrees).

        ball_direction_change_deg:
            Signed angular change between consecutive movement vectors
            (degrees, range -180 to +180).

        ball_direction_change_abs_deg:
            Absolute angular change between consecutive movement vectors
            (degrees, range 0 to 180).

    Notes:
        - Requires ball_dx and ball_dy from extract_ball_speed().
        - Uses only available valid movement observations.
        - Missing ball positions are not interpolated.
        - A direction change requires two consecutive valid movement vectors.
        - The first valid movement vector has NaN direction change.
        - Zero-displacement observations have undefined direction.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    frame_col = params["frame_col"]

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

    # Extract valid movement observations
    movement = (
        out[[frame_col, "ball_dx", "ball_dy"]]
        .dropna(subset=["ball_dx", "ball_dy"])
        .drop_duplicates(subset=[frame_col], keep="first")
        .sort_values(frame_col)
        .copy()
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

    # Convert movement vectors into radians
    dx_previous = movement["ball_dx"].shift(1)
    dy_previous = movement["ball_dy"].shift(1)

    dx_current = movement["ball_dx"]
    dy_current = movement["ball_dy"]

    # Calculate signed angular change using cross and dot products
    cross_product = (
        dx_previous * dy_current
        - dy_previous * dx_current
    )

    dot_product = (
        dx_previous * dx_current
        + dy_previous * dy_current
    )

    movement["ball_direction_change_deg"] = np.degrees(
        np.arctan2(cross_product, dot_product)
    )

    # Exclude direction changes when either movement vector is invalid
    previous_valid = (
        dx_previous.notna()
        & dy_previous.notna()
        & ~(
            dx_previous.eq(0)
            & dy_previous.eq(0)
        )
    )

    current_valid = ~zero_movement

    movement.loc[
        ~(previous_valid & current_valid),
        "ball_direction_change_deg",
    ] = np.nan

    # Absolute direction change
    movement["ball_direction_change_abs_deg"] = (
        movement["ball_direction_change_deg"].abs()
    )

    # Keep extracted features
    result = movement[
        [
            frame_col,
            "ball_direction_deg",
            "ball_direction_change_deg",
            "ball_direction_change_abs_deg",
        ]
    ]

    # Merge into original dataframe
    out = out.merge(
        result,
        on=frame_col,
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

    Calculates the rate of change in player-to-ball distance using
    consecutive available observations.

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
        - Uses actual elapsed time between available observations.
        - Selects the relative velocity of the nearest player in each team.
        - Does not interpolate missing distances.
        - Does not estimate player possession.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    frame_col = params["frame_col"]
    fps = params["fps"]

    distance_suffix = params["distance_suffix"]
    team_suffix = params["team_suffix"]

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

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
    frames = (
        pd.to_numeric(out[frame_col], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    frame_data = pd.DataFrame({
        frame_col: frames.to_numpy()
    })

    frame_index = pd.Index(frames.to_numpy())

    distance_data = {}
    velocity_data = {}

    # Calculate relative velocity for every tracked player
    for distance_column in player_columns:

        player_distance = (
            out[[frame_col, distance_column]]
            .copy()
        )

        player_distance[distance_column] = pd.to_numeric(
            player_distance[distance_column],
            errors="coerce"
        )

        player_distance = (
            player_distance
            .drop_duplicates(subset=[frame_col], keep="first")
            .set_index(frame_col)[distance_column]
            .reindex(frame_index)
        )

        # Calculate relative velocity between consecutive valid observations
        valid_distance = player_distance.dropna()

        elapsed_time = (
            valid_distance.index.to_series().diff() / fps
        )

        valid_relative_velocity = (
            -valid_distance.diff() / elapsed_time
        ).where(elapsed_time > 0)

        # Restore original frame index, keeping missing observations as NaN
        relative_velocity = valid_relative_velocity.reindex(frame_index)

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
            | (team_a_dist <= team_b_dist)
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

    out = out.merge(
        result,
        on=frame_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return out


def extract_ball_trajectory_toward_goal(
    df: pd.DataFrame, params: dict = PARAMS
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
        - Requires ball_dx, ball_displacement and ball_speed.
        - Uses the actual average ball speed between valid observations.
        - Does not assume the nearest player is the possession team.
        - Calculates both teams' attacking perspectives independently.
        - Attack directions must reflect the current playing half.
        - Does not interpolate missing ball observations.
        - Preserves all existing columns and features.
    """

    out = df.copy()

    required_columns = [
        "ball_dx",
        "ball_displacement",
        "ball_speed",
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

    # Retrieve attacking directions
    team_a_direction = params["team_a_attack_direction"]
    team_b_direction = params["team_b_attack_direction"]

    # Validate attack directions
    if team_a_direction not in (-1, 1):
        raise ValueError(
            "team_a_attack_direction must be either -1 or +1."
        )

    if team_b_direction not in (-1, 1):
        raise ValueError(
            "team_b_attack_direction must be either -1 or +1."
        )

    if team_a_direction == team_b_direction:
        raise ValueError(
            "The two opposing teams should attack in opposite "
            "directions during the same half."
        )

    # Calculate horizontal ball velocity.
    # ball_speed contains the magnitude of the velocity.
    # ball_dx / ball_displacement provides its horizontal direction.
    out["ball_velocity_x"] = (
        out["ball_speed"]
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

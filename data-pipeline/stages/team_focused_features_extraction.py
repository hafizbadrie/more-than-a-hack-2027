"""Functions to build team-focused features from player tracking data."""
import re
import pandas as pd
import numpy as np
from stages.direction import period_direction_multipliers


def _clean_team_name(team) -> str:
	"""Lowercase and replace any non-alphanumeric run with an underscore."""
	return re.sub(r"[^0-9a-z]+", "_", str(team).strip().lower()).strip("_") or "unknown"


def _build_player_data(features_hbl: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	"""Build a long-form table of player coordinates and their teams."""
	frame_cols = ["period_id", "video_frame"] if "period_id" in features_hbl else ["video_frame"]
	if "video_frame" not in features_hbl:
		raise ValueError("Cannot calculate team features without a 'video_frame' column.")

	player_ids = [
		column[:-4]
		for column in features_hbl.columns
		if column.endswith("_x_m")
		and column[:-4] + "_y_m" in features_hbl
		and column[:-4] + "_team" in features_hbl
	]
	if not player_ids:
		raise ValueError(
			"Cannot calculate team features: no matching player _x_m, _y_m, "
			"and _team columns found."
		)

	player_data = pd.concat(
		[
			features_hbl[frame_cols].assign(
				player_id=player_id,
				team=features_hbl[f"{player_id}_team"],
				x=pd.to_numeric(features_hbl[f"{player_id}_x_m"], errors="coerce"),
				y=pd.to_numeric(features_hbl[f"{player_id}_y_m"], errors="coerce"),
			)
			for player_id in player_ids
		],
		ignore_index=True,
	)
	return player_data, frame_cols


def _ball_coordinates_in_metres(
    df: pd.DataFrame,
    ball_x_col: str,
    ball_y_col: str,
) -> tuple[pd.Series, pd.Series]:
    converted = []
    for axis, ball_col in (("x", ball_x_col), ("y", ball_y_col)):
        suffix = f"_{axis}_m"
        slots = [
            column[:-len(suffix)]
            for column in df.columns
            if column.endswith(suffix)
            and column[:-len(suffix)].isdigit()
            and f"{column[:-len(suffix)]}_{axis}" in df.columns
        ]
        if not slots:
            raise ValueError(
                f"Cannot fit affine ball coordinate conversion for {axis}: "
                "no player coordinate pairs."
            )
        normalized = np.concatenate([
            pd.to_numeric(df[f"{slot}_{axis}"], errors="coerce").to_numpy(dtype=float)
            for slot in slots
        ])
        metres = np.concatenate([
            pd.to_numeric(df[f"{slot}_{axis}_m"], errors="coerce").to_numpy(dtype=float)
            for slot in slots
        ])
        valid = np.isfinite(normalized) & np.isfinite(metres)
        if valid.sum() < 2 or np.ptp(normalized[valid]) == 0:
            raise ValueError(
                f"Cannot fit affine ball coordinate conversion for {axis}: "
                "insufficient distinct finite player coordinates."
            )
        coefficients, _, rank, _, _ = np.polyfit(
            normalized[valid], metres[valid], 1, full=True
        )
        if rank < 2 or not np.isfinite(coefficients).all():
            raise ValueError(
                f"Cannot fit affine ball coordinate conversion for {axis}: "
                "degenerate player coordinates."
            )
        converted.append(
            pd.to_numeric(df[ball_col], errors="coerce") * coefficients[0] + coefficients[1]
        )
    return converted[0], converted[1]


def team_centroids(feature_phased: pd.DataFrame) -> pd.DataFrame:
	"""Feature set 1: Add per-team frame centroids from all tracked player coordinates."""
	df = feature_phased.copy()
	player_data, frame_cols = _build_player_data(df)
	overall_centroids = (
		player_data.dropna(subset=["x", "y"])
		.groupby(frame_cols)[["x", "y"]]
		.mean()
		.rename(
			columns={
				"x": "team_centroid_x_all_players",
				"y": "team_centroid_y_all_players",
			}
		)
		.reset_index()
	)
	df = df.merge(overall_centroids, on=frame_cols, how="left")

	teams = player_data["team"].dropna().unique()
	for team in teams:
		team_name = _clean_team_name(team)
		team_centroids = (
			player_data.loc[player_data["team"].eq(team)]
			.dropna(subset=["x", "y"])
			.groupby(frame_cols)[["x", "y"]]
			.mean()
			.rename(
				columns={
					"x": f"team_centroid_x_{team_name}",
					"y": f"team_centroid_y_{team_name}",
				}
			)
			.reset_index()
		)
		df = df.merge(team_centroids, on=frame_cols, how="left")

	return df

def team_dispersion(with_team_centroids: pd.DataFrame) -> pd.DataFrame:
	# Feature set 2: Team dispersion. It's defined as the average distance for each player from the team centroid.
	df = with_team_centroids.copy()
	player_data, frame_cols = _build_player_data(df)
	teams = player_data["team"].dropna().unique()

	for team in teams:
		team_name = _clean_team_name(team)
		centroid_columns = [
			f"team_centroid_x_{team_name}",
			f"team_centroid_y_{team_name}",
		]
		team_distances = (
			player_data.loc[player_data["team"].eq(team)]
			.dropna(subset=["x", "y"])
			.merge(df[frame_cols + centroid_columns], on=frame_cols, how="left")
		)
		team_distances[f"team_dispersion_{team_name}"] = (
			(team_distances["x"] - team_distances[centroid_columns[0]]) ** 2
			+ (team_distances["y"] - team_distances[centroid_columns[1]]) ** 2
		) ** 0.5
		team_dispersion = (
			team_distances.groupby(frame_cols)[f"team_dispersion_{team_name}"]
			.mean()
			.reset_index()
		)
		df = df.merge(team_dispersion, on=frame_cols, how="left")
	return df


def team_centroid_velocity(
    with_team_centroids: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:

    params = params or {}
    fps = params.get("fps", 25.0)
    """
    Feature set 3: Team centroid velocity.

    Calculate the velocity of each team's centroid between consecutive
    valid observations.

    Positive/negative X and Y velocities indicate movement along the
    respective pitch axes. Velocity magnitude represents the overall
    speed of the team's centroid in metres per second.

    Args:
        with_team_centroids: DataFrame containing team centroid coordinates.
        fps: Tracking frequency in frames per second.

    Returns:
        Original DataFrame with additional team centroid velocity features.
    """
    if fps <= 0:
        raise ValueError("'fps' must be greater than zero.")

    df = with_team_centroids.copy()

    if "video_frame" not in df.columns:
        raise ValueError(
            "Cannot calculate team centroid velocity without a 'video_frame' column."
        )

    teams = [
        column.removeprefix("team_centroid_x_")
        for column in df.columns
        if column.startswith("team_centroid_x_")
        and f"team_centroid_y_{column.removeprefix('team_centroid_x_')}" in df.columns
    ]

    if not teams:
        raise ValueError(
            "Cannot calculate team centroid velocity: no matching "
            "team centroid X/Y columns found."
        )

    # Use period_id as part of the grouping when available.
    frame_cols = (
        ["period_id", "video_frame"]
        if "period_id" in df.columns
        else ["video_frame"]
    )

    for team_name in teams:
        x_col = f"team_centroid_x_{team_name}"
        y_col = f"team_centroid_y_{team_name}"

        # Keep one centroid observation per frame.
        centroid_data = (
            df[frame_cols + [x_col, y_col]]
            .copy()
            .assign(
                **{
                    x_col: lambda x: pd.to_numeric(x[x_col], errors="coerce"),
                    y_col: lambda x: pd.to_numeric(x[y_col], errors="coerce"),
                }
            )
            .dropna(subset=[x_col, y_col])
            .drop_duplicates(subset=frame_cols, keep="first")
            .sort_values(frame_cols)
            .copy()
        )

        # Calculate velocity independently within each period.
        if "period_id" in frame_cols:
            grouped = centroid_data.groupby("period_id", sort=False)
        else:
            grouped = [(None, centroid_data)]

        velocity_results = []

        for period, group in grouped:
            group = group.copy()

            elapsed_time = group["video_frame"].diff() / fps

            group[f"team_centroid_velocity_x_{team_name}"] = (
                group[x_col].diff() / elapsed_time
            ).where(elapsed_time > 0)

            group[f"team_centroid_velocity_y_{team_name}"] = (
                group[y_col].diff() / elapsed_time
            ).where(elapsed_time > 0)

            group[f"team_centroid_speed_{team_name}"] = (
                group[
                    [
                        f"team_centroid_velocity_x_{team_name}",
                        f"team_centroid_velocity_y_{team_name}",
                    ]
                ]
                .pow(2)
                .sum(axis=1)
                .pow(0.5)
            )

            velocity_results.append(
                group[
                    frame_cols
                    + [
                        f"team_centroid_velocity_x_{team_name}",
                        f"team_centroid_velocity_y_{team_name}",
                        f"team_centroid_speed_{team_name}",
                    ]
                ]
            )

        team_velocity = pd.concat(velocity_results, ignore_index=True)

        df = df.merge(
            team_velocity,
            on=frame_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return df


def team_average_radial_movement(
    features: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:

    params = params or {}
    fps = params.get("fps", 25.0)
    ball_x_col = params.get("ball_x_col", "ball_x")
    ball_y_col = params.get("ball_y_col", "ball_y")
    """
    Feature set 4: Team average radial movement relative to the ball.

    Calculate each player's velocity projected onto the direction
    from the player towards the ball.

    Positive values indicate movement towards the ball.
    Negative values indicate movement away from the ball.

    The team feature is the mean radial velocity of its tracked players.

    Args:
        features: DataFrame containing player coordinates, team labels,
                  ball coordinates, and video_frame.
        fps: Tracking frequency in frames per second.
        ball_x_col: Ball X-coordinate column.
        ball_y_col: Ball Y-coordinate column.

    Returns:
        Original DataFrame with one average radial movement feature
        per team.
    """
    if fps <= 0:
        raise ValueError("'fps' must be greater than zero.")

    df = features.copy()

    required_columns = ["video_frame", ball_x_col, ball_y_col]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for radial movement: {missing_columns}"
        )
    ball_x_m, ball_y_m = _ball_coordinates_in_metres(
        df, ball_x_col, ball_y_col
    )

    # Identify players with matching X, Y and team columns.
    player_ids = [
        column[:-4]
        for column in df.columns
        if column.endswith("_x_m")
        and column[:-4] + "_y_m" in df.columns
        and column[:-4] + "_team" in df.columns
        and column not in {ball_x_col}
    ]

    if not player_ids:
        raise ValueError(
            "Cannot calculate radial movement: no matching player "
            "_x_m, _y_m and _team columns found."
        )

    frame_cols = (
        ["period_id", "video_frame"]
        if "period_id" in df.columns
        else ["video_frame"]
    )

    # Build long-form player tracking data.
    player_data = pd.concat(
        [
            df[frame_cols].assign(
                player_id=player_id,
                team=df[f"{player_id}_team"],
                x=pd.to_numeric(df[f"{player_id}_x_m"], errors="coerce"),
                y=pd.to_numeric(df[f"{player_id}_y_m"], errors="coerce"),
                ball_x=ball_x_m,
                ball_y=ball_y_m,
            )
            for player_id in player_ids
        ],
        ignore_index=True,
    )

    player_data = player_data.sort_values(
        ["player_id"] + frame_cols
    ).copy()

    # Calculate player velocity using consecutive valid coordinate pairs.
    valid_positions = player_data.dropna(
        subset=["x", "y"]
    ).copy()

    grouped = valid_positions.groupby(
        ["player_id"] + (["period_id"] if "period_id" in frame_cols else []),
        sort=False,
    )

    valid_positions["previous_x"] = grouped["x"].shift(1)
    valid_positions["previous_y"] = grouped["y"].shift(1)
    valid_positions["previous_frame"] = grouped["video_frame"].shift(1)

    elapsed_time = (
        valid_positions["video_frame"] - valid_positions["previous_frame"]
    ) / fps

    valid_positions["player_vx"] = (
        (valid_positions["x"] - valid_positions["previous_x"]) / elapsed_time
    ).where(elapsed_time > 0)

    valid_positions["player_vy"] = (
        (valid_positions["y"] - valid_positions["previous_y"]) / elapsed_time
    ).where(elapsed_time > 0)

    # Vector from player position towards the ball.
    direction_x = valid_positions["ball_x"] - valid_positions["x"]
    direction_y = valid_positions["ball_y"] - valid_positions["y"]

    distance_to_ball = (
        direction_x.pow(2) + direction_y.pow(2)
    ).pow(0.5)

    # Unit vector pointing towards the ball.
    unit_x = direction_x / distance_to_ball.where(distance_to_ball > 0)
    unit_y = direction_y / distance_to_ball.where(distance_to_ball > 0)

    # Project player velocity onto the player-to-ball direction.
    valid_positions["radial_movement"] = (
        valid_positions["player_vx"] * unit_x
        + valid_positions["player_vy"] * unit_y
    )

    # Average radial movement per team and frame.
    team_features = (
        valid_positions.dropna(subset=["radial_movement", "team"])
        .groupby(frame_cols + ["team"])["radial_movement"]
        .mean()
        .reset_index()
    )

    # Create one output column per team.
    for team in team_features["team"].dropna().unique():
        team_name = _clean_team_name(team)

        team_result = (
            team_features.loc[team_features["team"].eq(team)]
            [frame_cols + ["radial_movement"]]
            .rename(
                columns={
                    "radial_movement":
                        f"team_average_radial_movement_{team_name}"
                }
            )
        )

        df = df.merge(
            team_result,
            on=frame_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return df


def team_width_and_depth(features: pd.DataFrame) -> pd.DataFrame:
    """
    Feature set 5: Team width and depth.

    Calculate the horizontal and longitudinal extent of each team's
    player positions per frame.

    Team width = max(Y) - min(Y)
    Team depth = max(X) - min(X)

    Args:
        features: DataFrame containing player X/Y coordinates and team labels.

    Returns:
        Original DataFrame with additional team width and depth features.
    """
    df = features.copy()

    player_data, frame_cols = _build_player_data(df)

    teams = player_data["team"].dropna().unique()

    for team in teams:
        team_name = _clean_team_name(team)

        team_positions = (
            player_data.loc[player_data["team"].eq(team)]
            .dropna(subset=["x", "y"])
        )

        team_dimensions = (
            team_positions.groupby(frame_cols)
            .agg(
                **{
                    f"team_width_{team_name}": ("y", lambda x: x.max() - x.min()),
                    f"team_depth_{team_name}": ("x", lambda x: x.max() - x.min()),
                    f"team_tracked_players_{team_name}": ("player_id", "count"),
                }
            )
            .reset_index()
        )

        df = df.merge(
            team_dimensions,
            on=frame_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return df


def team_movement_coherence(
    features: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    params = params or {}
    fps = params.get("fps", 25.0)
    """
    Feature set 6: Team movement coherence.

    Measures how similarly players within a team are moving.
    Uses mean resultant length of player movement direction vectors.

    Output range: 0 to 1.
    """
    if fps <= 0:
        raise ValueError("'fps' must be greater than zero.")

    df = features.copy()
    player_data, frame_cols = _build_player_data(df)

    grouping_cols = ["player_id"] + (
        ["period_id"] if "period_id" in frame_cols else []
    )

    player_data = player_data.sort_values(
        grouping_cols + ["video_frame"]
    ).copy()

    grouped = player_data.groupby(grouping_cols, sort=False)

    player_data["previous_x"] = grouped["x"].shift(1)
    player_data["previous_y"] = grouped["y"].shift(1)
    player_data["previous_frame"] = grouped["video_frame"].shift(1)

    elapsed_time = (
        player_data["video_frame"] - player_data["previous_frame"]
    ) / fps

    player_data["vx"] = (
        (player_data["x"] - player_data["previous_x"]) / elapsed_time
    ).where(elapsed_time > 0)

    player_data["vy"] = (
        (player_data["y"] - player_data["previous_y"]) / elapsed_time
    ).where(elapsed_time > 0)

    speed = np.hypot(player_data["vx"], player_data["vy"])

    # Exclude stationary players and invalid velocity observations.
    valid = (
        speed.notna()
        & speed.gt(0)
        & player_data["team"].notna()
    )

    player_data["direction_x"] = (
        player_data["vx"] / speed
    ).where(valid)

    player_data["direction_y"] = (
        player_data["vy"] / speed
    ).where(valid)

    # Average the unit movement vectors per team and frame.
    coherence = (
        player_data.dropna(
            subset=["direction_x", "direction_y"]
        )
        .groupby(frame_cols + ["team"])
        .agg(
            mean_direction_x=("direction_x", "mean"),
            mean_direction_y=("direction_y", "mean"),
            tracked_players=("player_id", "count"),
        )
        .reset_index()
    )

    coherence["movement_coherence"] = np.hypot(
        coherence["mean_direction_x"],
        coherence["mean_direction_y"],
    ).clip(0, 1)

    for team in coherence["team"].dropna().unique():
        team_name = _clean_team_name(team)

        team_result = (
            coherence.loc[coherence["team"].eq(team)]
            [frame_cols + ["movement_coherence", "tracked_players"]]
            .rename(
                columns={
                    "movement_coherence":
                        f"team_movement_coherence_{team_name}",
                    "tracked_players":
                        f"team_movement_coherence_players_{team_name}",
                }
            )
        )

        df = df.merge(
            team_result,
            on=frame_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return df


def opponent_team_proximity(
    features: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    params = params or {}
    fps = params.get("fps", 25.0)
    """
    Feature set 7: Opponent team proximity and closing movement.

    Calculates the distance between the two team centroids and
    the rate at which that distance changes.

    Positive closing speed = teams approaching.
    Negative closing speed = teams separating.
    """
    if fps <= 0:
        raise ValueError("'fps' must be greater than zero.")

    df = features.copy()
    player_data, frame_cols = _build_player_data(df)

    teams = player_data["team"].dropna().unique()

    if len(teams) != 2:
        raise ValueError(
            "Opponent proximity requires exactly two distinct teams. "
            f"Found {len(teams)} teams: {list(teams)}"
        )

    centroids = (
        player_data.dropna(subset=["x", "y"])
        .groupby(frame_cols + ["team"])[["x", "y"]]
        .mean()
        .reset_index()
    )

    centroids = centroids.pivot(
        index=frame_cols,
        columns="team",
        values=["x", "y"],
    )

    team_a, team_b = teams

    proximity = pd.DataFrame(index=centroids.index)

    proximity["opponent_team_centroid_distance"] = np.hypot(
        centroids[("x", team_a)] - centroids[("x", team_b)],
        centroids[("y", team_a)] - centroids[("y", team_b)],
    )

    proximity = proximity.reset_index().sort_values(frame_cols)

    grouping_cols = ["period_id"] if "period_id" in frame_cols else []

    if grouping_cols:
        grouped = proximity.groupby(grouping_cols, sort=False)
        elapsed_time = grouped["video_frame"].diff() / fps
        distance_change = grouped[
            "opponent_team_centroid_distance"
        ].diff()
    else:
        elapsed_time = proximity["video_frame"].diff() / fps
        distance_change = proximity[
            "opponent_team_centroid_distance"
        ].diff()

    proximity["opponent_team_closing_speed"] = (
        -distance_change / elapsed_time
    ).where(elapsed_time > 0)

    df = df.merge(
        proximity[
            frame_cols
            + [
                "opponent_team_centroid_distance",
                "opponent_team_closing_speed",
            ]
        ],
        on=frame_cols,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    return df


def team_position_relative_to_ball(
    features: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    params = params or {}
    ball_x_col = params.get("ball_x_col", "ball_x")
    ball_y_col = params.get("ball_y_col", "ball_y")
    attack_directions = params.get("attack_directions")
    """
    Feature set 8: Team positional relationship to ball.

    Calculate the longitudinal and lateral distance between each
    team's centroid and the ball.

    Positive longitudinal distance = ball is ahead of the team
    in its attacking direction.

    Positive lateral distance = ball is towards positive Y
    relative to the team centroid.

    attack_directions:
        Dictionary mapping original team names to their X-axis
        attacking direction (+1 or -1).
    period_direction_multipliers:
        Optional mapping from period_id to +1/-1 to account for teams
        switching ends. Missing periods are rejected.

        Example:
            {
                "Barcelona": 1,
                "Atletic Club": -1
            }
    """
    df = features.copy()

    required_columns = ["video_frame", ball_x_col, ball_y_col]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required ball columns: {missing_columns}"
        )
    ball_x_m, ball_y_m = _ball_coordinates_in_metres(
        df, ball_x_col, ball_y_col
    )
    direction_multiplier = period_direction_multipliers(df, params)

    player_data, frame_cols = _build_player_data(df)

    teams = player_data["team"].dropna().unique()

    if attack_directions is None:
        raise ValueError(
            "Please provide attack_directions for both teams."
        )

    ball_positions = (
        df[frame_cols + [ball_x_col, ball_y_col]]
        .copy()
        .assign(
            **{
                ball_x_col: ball_x_m,
                ball_y_col: ball_y_m,
                "_period_direction_multiplier": direction_multiplier,
            }
        )
        .drop_duplicates(subset=frame_cols, keep="first")
    )

    for team in teams:
        if team not in attack_directions:
            raise ValueError(
                f"Missing attacking direction for team '{team}'."
            )

        direction = attack_directions[team]

        if direction not in (-1, 1):
            raise ValueError(
                f"Attack direction for '{team}' must be +1 or -1."
            )

        team_name = _clean_team_name(team)

        team_centroid = (
            player_data.loc[player_data["team"].eq(team)]
            .dropna(subset=["x", "y"])
            .groupby(frame_cols)[["x", "y"]]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "x": "centroid_x",
                    "y": "centroid_y",
                }
            )
        )

        team_positions = team_centroid.merge(
            ball_positions,
            on=frame_cols,
            how="left",
        )

        # Positive means the ball is ahead in the team's
        # attacking direction.
        team_positions[
            f"team_ball_longitudinal_distance_{team_name}"
        ] = (
            team_positions[ball_x_col] - team_positions["centroid_x"]
        ) * direction * team_positions["_period_direction_multiplier"]

        # Positive means the ball is towards positive Y.
        team_positions[
            f"team_ball_lateral_distance_{team_name}"
        ] = (
            team_positions[ball_y_col] - team_positions["centroid_y"]
        )

        result_columns = [
            f"team_ball_longitudinal_distance_{team_name}",
            f"team_ball_lateral_distance_{team_name}",
        ]

        df = df.merge(
            team_positions[frame_cols + result_columns],
            on=frame_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return df

"""Functions to build team-focused features from player tracking data."""
import re
import pandas as pd


def _clean_team_name(team) -> str:
	"""Lowercase and replace any non-alphanumeric run with an underscore."""
	return re.sub(r"[^0-9a-z]+", "_", str(team).strip().lower()).strip("_") or "unknown"


def _build_player_data(features_hbl: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	"""Build a long-form table of player coordinates and their teams."""
	frame_cols = ["period_id", "frame_id"] if "period_id" in features_hbl else ["frame_id"]
	if "frame_id" not in features_hbl:
		raise ValueError("Cannot calculate team features without a 'frame_id' column.")

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
				team=features_hbl[f"{player_id}_team"],
				x=pd.to_numeric(features_hbl[f"{player_id}_x_m"], errors="coerce"),
				y=pd.to_numeric(features_hbl[f"{player_id}_y_m"], errors="coerce"),
			)
			for player_id in player_ids
		],
		ignore_index=True,
	)
	return player_data, frame_cols


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
		with_team_centroids = df.merge(team_centroids, on=frame_cols, how="left")

	return with_team_centroids

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
		with_team_dispersion = df.merge(team_dispersion, on=frame_cols, how="left")
	return with_team_dispersion
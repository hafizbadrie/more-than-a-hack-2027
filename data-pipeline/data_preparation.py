"""Convert the tactical EPTS and Smart Tagging assets into CSVs."""

import numpy as np
import pandas as pd
from kedro.pipeline import node, pipeline
from kloppy import metrica, sportscode


def convert_atd_to_dataframe(paths, second_half_start):
    dataset = metrica.load_tracking_epts(
        meta_data=paths["atd_meta"], raw_data=paths["atd_raw"]
    )
    frame_df = dataset.to_df().rename(columns={"frame_id": "video_frame"})
    frame_df["video_frame"] = frame_df["video_frame"].astype(int) - 1
    frame_df.loc[frame_df["video_frame"] >= second_half_start, "period_id"] = 2

    ball_speed_by_frame = {}
    with open(paths["atd_raw"]) as raw:
        for line in raw:
            atd_frame, _players, ball = line.rstrip("\r\n").split(":")
            ball_speed_by_frame[int(atd_frame)] = float(ball.split(",")[2])
    frame_df["ball_speed"] = (frame_df["video_frame"] + 1).map(ball_speed_by_frame)

    length = dataset.metadata.pitch_dimensions.pitch_length
    width = dataset.metadata.pitch_dimensions.pitch_width
    ball_x_m = frame_df["ball_x"] * length
    ball_y_m = frame_df["ball_y"] * width
    extra = {}
    for player in dataset.metadata.teams[0].players + dataset.metadata.teams[1].players:
        pid = player.player_id
        x_m = frame_df[f"{pid}_x"] * length
        y_m = frame_df[f"{pid}_y"] * width
        extra[f"{pid}_x_m"] = x_m
        extra[f"{pid}_y_m"] = y_m
        extra[f"{pid}_ball_dist_m"] = np.hypot(x_m - ball_x_m, y_m - ball_y_m)
        extra[f"{pid}_team"] = player.team.name

    return pd.concat([frame_df, pd.DataFrame(extra, index=frame_df.index)], axis=1)


def convert_smart_tagging_to_dataframe(paths, fps):
    events = sportscode.load(paths["smart_tagging"]).to_df()
    events["start_frame"] = (events["timestamp"].dt.total_seconds() * fps).round().astype(int)
    events["end_frame"] = (
        events["end_timestamp"].dt.total_seconds() * fps
    ).round().astype(int)
    return events


def build_data_preparation_pipeline():
    return pipeline(
        [
            node(
                convert_atd_to_dataframe,
                ["params:atd_paths", "params:second_half_start"],
                "raw_tactical",
                name="atd_to_csv",
            ),
            node(
                convert_smart_tagging_to_dataframe,
                ["params:atd_paths", "params:fps"],
                "raw_smart_tag",
                name="smart_tagging_to_csv",
            ),
        ]
    )

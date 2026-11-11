"""Step 0: turn the provided data bundle into the two CSVs the pipeline reads.

    python prepare_data.py

Same two conversions as the data-prep notebook (mp_dataprep.ipynb, sections 1.1 and 2.1), as a script so that one
command can run the whole pipeline.  Run it from the smart-tags folder (next to games.py's parent).

  data/tactical/BAR-ATH_tactical_ds.csv        per-frame ATD tracking: ball, 46 player slots, metres, ball distance, team name;
                                              `video_frame` = ATD frame id - 1
  data/tactical/BAR-ATH_smart_tagging_ds.csv   Smart Tagging intervals with start_frame / end_frame (video frames, seconds x fps)

Needs kloppy, numpy and pandas, and the data bundle where games.py expects it.  No ground truth is read.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
from kloppy import metrica, sportscode   # noqa: E402

import games                # noqa: E402


def prepare_tracking():
    paths = games.asset_paths(source="tactical")
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    frame_df = dataset.to_df()
    frame_df = frame_df.rename(columns={"frame_id": "video_frame"})
    frame_df["video_frame"] = frame_df["video_frame"].astype(int) - 1

    ball_speed_by_frame = {}
    with open(paths["atd_raw"]) as f:
        for line in f:
            atd_frame, _players, ball = line.rstrip("\r\n").split(":")
            speed = ball.split(",")[2]
            ball_speed_by_frame[int(atd_frame)] = float(speed) if speed != "NaN" else np.nan
    frame_df["ball_speed"] = (frame_df["video_frame"].astype(int) + 1).map(ball_speed_by_frame)

    # per-player positions in metres, distance to the ball, team name
    length = dataset.metadata.pitch_dimensions.pitch_length
    width = dataset.metadata.pitch_dimensions.pitch_width
    ball_x_m = frame_df["ball_x"] * length
    ball_y_m = frame_df["ball_y"] * width
    extra = {}
    for player in dataset.metadata.teams[0].players + dataset.metadata.teams[1].players:
        pid = player.player_id
        x_m = frame_df["%s_x" % pid] * length
        y_m = frame_df["%s_y" % pid] * width
        extra["%s_x_m" % pid] = x_m
        extra["%s_y_m" % pid] = y_m
        extra["%s_ball_dist_m" % pid] = np.hypot(x_m - ball_x_m, y_m - ball_y_m)
        extra["%s_team" % pid] = player.team.name
    frame_df = pd.concat([frame_df, pd.DataFrame(extra, index=frame_df.index)], axis=1)

    out = os.path.join(games.DATA_DIR, "tactical", "BAR-ATH_tactical_ds.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frame_df.to_csv(out, index=False)
    print("tracking: wrote %d frames to %s" % (len(frame_df), out))


def prepare_smart_tagging():
    fps = games.GAMES[games.GAME]["fps"]
    events = sportscode.load(games.asset_paths()["smart_tagging"]).to_df()
    events["start_frame"] = (events["timestamp"].dt.total_seconds() * fps).round().astype(int)
    events["end_frame"] = (events["end_timestamp"].dt.total_seconds() * fps).round().astype(int)
    out = os.path.join(games.DATA_DIR, "tactical", "BAR-ATH_smart_tagging_ds.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    events.to_csv(out, index=False)
    print("smart tagging: wrote %d intervals to %s" % (len(events), out))


if __name__ == "__main__":
    prepare_tracking()
    prepare_smart_tagging()

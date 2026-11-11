"""Read the per-frame tracking CSV whichever way its frame column is named.

The data-prep notebook writes `video_frame` (0-based video frame, = ATD frame id - 1); older exports have `frame_id`
(ATD frame id, counts from 1).  Every script here works with `frame_id` = video frame + 1, so this reader converts.
"""
import numpy as np
import pandas as pd


def read_tracking(path, columns):
    """Read only `columns` (plus the frame column) and return them with a `frame_id` column (= video frame + 1)."""
    header = list(pd.read_csv(path, nrows=0).columns)
    if "video_frame" in header:                       # preferred: unambiguous 0-based video frame
        fcol, add = "video_frame", 1
    elif "frame_id" in header:
        fcol, add = "frame_id", 0
    else:
        raise ValueError("tracking csv has neither video_frame nor frame_id; it has %d columns, e.g. %s" % (len(header), header[:8]))
    want = [c for c in columns if c not in ("frame_id", "video_frame")]
    missing = [c for c in want if c not in header]
    if missing:
        raise ValueError("tracking csv lacks columns: %s" % missing[:6])
    df = pd.read_csv(path, usecols=[fcol] + want)
    frame = df.pop(fcol).astype(np.int64) + add
    df.insert(0, "frame_id", frame)
    return df

"""Stage 3b: tag features - join compact Smart Tagging features onto the per-frame table.

Node signature mirrors stages/features.py:

    node(build_tag_features,
         ["features", "raw_smart_tag", "params:features", "match_info"],
         "features_tagged", name="tag_features")

Inputs
  df         per-frame table (the "features" output). Needs a frame column, see `frame_col`.
  smart_tag  the Smart Tagging events CSV as loaded by CSVDataset (one row per tag interval) with
             at least `code`, `start_frame`, `end_frame`; `Team`, `Type`, `Half` and
             `Max Players in the box` are used when present, everything else is ignored.
             start_frame / end_frame are video frames (seconds * 25 from the first frame of the
             tactical video), end_frame being the first frame AFTER the interval.
  params     optional keys (all have defaults):
               frame_col        column holding the frame number. Default precedence:
                                video_frame, frame_id, then frame.
               frame_offset     added to that column to get the VIDEO frame. Default: -1 for
                                frame_id (ATD frame ids count from 1), 0 otherwise.
               fps              default 25
               pitch            (length, width) in metres, default (105.3, 68.0)
               ball_cols        default ("ball_x", "ball_y"); used only to confirm the restart
                                frame of "Kick Off (After Goal)" tags
               ball_normalised  True (default) if the ball columns are 0-1 pitch fractions
               kickoff_frames   default {"1st Half": 11185, "2nd Half": 113742} (DATA.md / FAQ.md)
               team_aliases     substring -> +1 (home) / -1 (away); default covers Barcelona and
                                Athletic ("Atletic Club" is how the export spells it)
  match_info the dict from main.py; home_team / away_team are also accepted as exact team names.

Columns added
  tag_att_signed        +1 / 0 / -1: attacking-type codes active for home / none / away. Overlaps are
                        summed and clipped, so contradicting tags cancel to 0.
  tag_att_trans_signed  Attacking Transition on its own, same sign convention (a hand-over marker).
  tag_def_signed        Defending (3 codes) + Defensive Transition. What `Team` means for these codes
                        is NOT confirmed (performer or possessor), so they are kept apart and the
                        model is expected to learn the sign.
  tag_n_active          number of tag intervals active on the frame; separates "no tag" from "tag
                        whose team is N/A".
  tag_box_players       max "Max Players in the box" over active tags (0-7; "7+" -> 7; 0 if none).
  sp_kind               set piece in progress: throw_in, corner_kick, goal_kick, free_kick,
                        kick_off_start, kick_off_after_goal; NaN otherwise.
  sp_team               +1 / -1 restarting team from the tag, 0 when the tag says N/A (all throw-ins).
  sp_since              seconds since the set-piece window started, NaN outside.
  evt_t_to_restart      seconds to the restart, ONLY for restarts that are confirmed, NaN elsewhere.
                        Negative = before the restart.

Nothing here is a label. The tags are automatic, their intervals are phases not possessions, and the
35 s set-piece windows are templates, not measured stoppages.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --- code groups (codes are matched case-insensitively, spaces -> underscores) ---------------
ATT = {"build_up", "progression", "creating_chances", "long_ball", "ball_in_final_third",
       "ball_in_the_box", "players_in_the_box", "goals"}
ATT_TRANS = "attacking_transition"
DEF = {"defending_in_attacking_third", "defending_in_middle_third",
       "defending_in_defensive_third", "defensive_transition"}
SET_PIECES = "set_pieces"

SP_KINDS = {"throw-in": "throw_in", "corner kick": "corner_kick", "goal kick": "goal_kick",
            "free kick": "free_kick", "kick off (start)": "kick_off_start",
            "kick off (after goal)": "kick_off_after_goal"}
SP_CATEGORIES = list(SP_KINDS.values())

TAG_COLUMNS = ["Team", "Type", "Half", "Max Players in the box"]
FRAME_COLUMNS = ["video_frame", "frame_id", "frame"]
DEFAULT_ALIASES = {"barcelona": 1, "barça": 1, "barca": 1, "athletic": -1, "atletic": -1}
DEFAULT_KICKOFFS = {"1st Half": 11185, "2nd Half": 113742}
CENTRE_RADIUS_M = 1.0      # ball counts as "on the centre spot" within this distance
OUT_COLUMNS = ["tag_att_signed", "tag_att_trans_signed", "tag_def_signed", "tag_n_active",
               "tag_box_players", "sp_kind", "sp_team", "sp_since", "evt_t_to_restart"]


# --- reading the events --------------------------------------------------------------------
def _text(value):
    """Cell -> stripped string; NaN (what pandas makes of "N/A" and empty cells) -> ''."""
    return "" if pd.isna(value) else str(value).strip()


def _load_events(smart_tag):
    """[{code, start_frame, end_frame, tags}] from the CSV rows."""
    frame = smart_tag.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in ("code", "start_frame", "end_frame") if c not in frame.columns]
    if missing:
        raise ValueError("smart tag table is missing column(s) %s; it has %s"
                         % (missing, list(frame.columns)))

    present = [c for c in TAG_COLUMNS if c in frame.columns]
    start = pd.to_numeric(frame["start_frame"], errors="coerce")
    end = pd.to_numeric(frame["end_frame"], errors="coerce")
    bad = int((start.isna() | end.isna()).sum())
    if bad:
        log.warning("tag_features: %d tag row(s) without a usable start_frame/end_frame skipped", bad)

    events = []
    for i, row in enumerate(frame.to_dict("records")):
        if pd.isna(start.iloc[i]) or pd.isna(end.iloc[i]):
            continue
        events.append({"code": _text(row["code"]),
                       "start_frame": int(start.iloc[i]),
                       "end_frame": int(end.iloc[i]),
                       "tags": {c: _text(row[c]) for c in present}})
    return events


def _code_key(code):
    return (code or "").strip().lower().replace(" ", "_")


def _tag(event, group):
    wanted = group.lower()
    for name, text in event["tags"].items():
        if name.strip().lower() == wanted:
            return text
    return ""


def _sp_kind_of(event):
    return SP_KINDS.get(_tag(event, "Type").lower())


def _players_in_box(text):
    """'7+' -> 7, '3' -> 3, '3.0' -> 3, '' -> 0."""
    try:
        return int(float(text.replace("+", "").strip()))
    except ValueError:
        return 0


def _make_signer(params, match_info):
    aliases = {k.lower(): v for k, v in params.get("team_aliases", DEFAULT_ALIASES).items()}
    exact = {}
    if match_info.get("home_team"):
        exact[str(match_info["home_team"]).lower()] = 1
    if match_info.get("away_team"):
        exact[str(match_info["away_team"]).lower()] = -1

    def sign(text):
        t = (text or "").strip().lower()
        if t in exact:
            return exact[t]
        for alias, value in aliases.items():
            if alias in t:
                return value
        return 0                                  # N/A, empty or unrecognised
    return sign


# --- time base and restarts ----------------------------------------------------------------
def _time_base_ok(events, kickoffs):
    """Do the tagged 'Kick Off (Start)' windows contain the documented kick-off frames?"""
    rows = [(e, kickoffs.get(_tag(e, "Half")))
            for e in events if _code_key(e["code"]) == SET_PIECES and _sp_kind_of(e) == "kick_off_start"]
    return len(rows) == 2 and all(f is not None and e["start_frame"] <= f <= e["end_frame"]
                                  for e, f in rows)


def _confirmed_restarts(events, kickoffs, ball_m, pitch):
    """[{event, frame, source}] for restarts that can be pinned to a frame.

    kick-off (start)       documented frame, accepted only if it lies inside the tagged window.
    kick-off (after goal)  last frame in the window with the ball within 1 m of the centre spot
                           (the moment it leaves the spot). Not confirmed if never seen there.
    everything else        not confirmed: the 35 s windows give no verified restart time.
    """
    out = []
    for index, event in enumerate(events):
        if _code_key(event["code"]) != SET_PIECES:
            continue
        kind = _sp_kind_of(event)
        if kind == "kick_off_start":
            frame = kickoffs.get(_tag(event, "Half"))
            if frame is not None and event["start_frame"] <= frame <= event["end_frame"]:
                out.append({"event": index, "frame": int(frame), "source": "documented"})
        elif kind == "kick_off_after_goal" and ball_m is not None:
            window = np.arange(max(event["start_frame"], 0), min(event["end_frame"], len(ball_m)))
            with np.errstate(invalid="ignore"):
                dist = np.hypot(ball_m[window, 0] - pitch[0] / 2, ball_m[window, 1] - pitch[1] / 2)
                near = window[dist <= CENTRE_RADIUS_M]
            if len(near):
                out.append({"event": index, "frame": int(near[-1]), "source": "ball"})
    return out


# --- the table -----------------------------------------------------------------------------
def _tag_table(events, n_frames, restarts, fps, sign_of):
    att = np.zeros(n_frames, np.float32)
    att_trans = np.zeros(n_frames, np.float32)
    deff = np.zeros(n_frames, np.float32)
    n_active = np.zeros(n_frames, np.int16)
    box = np.zeros(n_frames, np.float32)
    sp_kind = np.full(n_frames, -1, np.int8)
    sp_team = np.zeros(n_frames, np.int8)
    sp_since = np.full(n_frames, np.nan, np.float32)
    t_restart = np.full(n_frames, np.nan, np.float32)

    unknown_codes, unknown_types = set(), set()
    for event in sorted(events, key=lambda e: e["start_frame"]):      # later set pieces overwrite
        a, b = max(event["start_frame"], 0), min(event["end_frame"], n_frames)
        if b <= a:
            continue
        sl = slice(a, b)
        key = _code_key(event["code"])
        sign = sign_of(_tag(event, "Team"))
        n_active[sl] += 1

        if key in ATT:
            att[sl] += sign
        elif key == ATT_TRANS:
            att_trans[sl] += sign
        elif key in DEF:
            deff[sl] += sign
        elif key == SET_PIECES:
            kind = _sp_kind_of(event)
            if kind is None:
                unknown_types.add(_tag(event, "Type"))
            else:
                sp_kind[sl] = SP_CATEGORIES.index(kind)
            sp_team[sl] = sign
            sp_since[sl] = (np.arange(a, b) - event["start_frame"]) / fps
        else:
            unknown_codes.add(event["code"])

        players = _players_in_box(_tag(event, "Max Players in the box"))
        if players:
            box[sl] = np.maximum(box[sl], players)

    if unknown_codes:
        log.warning("tag_features: codes not in any group (counted in tag_n_active only): %s",
                    sorted(unknown_codes))
    if unknown_types:
        log.warning("tag_features: set-piece Type values not recognised (sp_kind left NaN): %s",
                    sorted(unknown_types))

    for restart in restarts:
        event = events[restart["event"]]
        a, b = max(event["start_frame"], 0), min(event["end_frame"], n_frames)
        if b > a:
            t_restart[a:b] = (np.arange(a, b) - restart["frame"]) / fps

    return pd.DataFrame({
        "tag_att_signed": np.clip(att, -1, 1),
        "tag_att_trans_signed": np.clip(att_trans, -1, 1),
        "tag_def_signed": np.clip(deff, -1, 1),
        "tag_n_active": n_active,
        "tag_box_players": box,
        "sp_kind": pd.Categorical.from_codes(sp_kind, categories=SP_CATEGORIES),
        "sp_team": sp_team,
        "sp_since": sp_since,
        "evt_t_to_restart": t_restart,
    })[OUT_COLUMNS]


def _video_frames(df, params):
    """(video frame of each row, frame column used). Fails loudly instead of guessing."""
    col = params.get("frame_col") or next((c for c in FRAME_COLUMNS if c in df.columns), None)
    if col is None or col not in df.columns:
        raise ValueError("no frame column: looked for %s in %s. Set params['frame_col'] "
                         "(and params['frame_offset'] if it is not the ATD video_frame)."
                         % (FRAME_COLUMNS, list(df.columns)))
    offset = params.get("frame_offset", -1 if col == "frame_id" else 0)
    frames = pd.to_numeric(df[col], errors="coerce")
    if frames.isna().any():
        raise ValueError("frame column %r has %d non-numeric value(s)" % (col, int(frames.isna().sum())))
    frames = frames.to_numpy().astype(np.int64) + int(offset)
    if (frames < 0).any():
        raise ValueError("frame column %r gives negative video frames with offset %d" % (col, offset))
    return frames, col


def build_tag_features(df: pd.DataFrame, smart_tag: pd.DataFrame, params: dict,
                       match_info: dict) -> pd.DataFrame:
    fps = params.get("fps", 25)
    pitch = tuple(params.get("pitch", (105.3, 68.0)))
    kickoffs = params.get("kickoff_frames", DEFAULT_KICKOFFS)

    frames, frame_col = _video_frames(df, params)
    n_frames = int(frames.max()) + 1
    events = _load_events(smart_tag)

    if not _time_base_ok(events, kickoffs):
        log.warning("tag_features: time base check FAILED - the tagged kick-off windows do not "
                    "contain the documented kick-off frames %s. Check that start_frame/end_frame "
                    "are video frames and not frames within a half.", kickoffs)

    ball_m = None
    bx, by = params.get("ball_cols", ("ball_x", "ball_y"))
    if bx in df.columns and by in df.columns:
        scale = pitch if params.get("ball_normalised", True) else (1.0, 1.0)
        ball_m = np.full((n_frames, 2), np.nan)
        ball_m[frames, 0] = pd.to_numeric(df[bx], errors="coerce").to_numpy(float) * scale[0]
        ball_m[frames, 1] = pd.to_numeric(df[by], errors="coerce").to_numpy(float) * scale[1]
    else:
        log.warning("tag_features: ball columns %s/%s not found; after-goal restarts stay unconfirmed",
                    bx, by)

    restarts = _confirmed_restarts(events, kickoffs, ball_m, pitch)
    table = _tag_table(events, n_frames, restarts, fps, _make_signer(params, match_info))
    log.info("tag_features: %d tag intervals, %d confirmed restarts, frame column %r",
             len(events), len(restarts), frame_col)

    out = df.copy()
    for col in OUT_COLUMNS:
        out[col] = table[col].iloc[frames].array          # positional: the index of df is irrelevant
    return out


def extract_tag_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Only the frame key and the 9 tag columns, for saving on their own.

    Takes the output of build_tag_features and keeps the same frame column it joined on, so the
    file can be merged back onto any per-frame table by that key.
    """
    _, frame_col = _video_frames(df, params)
    missing = [c for c in OUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("tag columns %s not found; run build_tag_features first" % missing)
    return df[[frame_col] + OUT_COLUMNS].reset_index(drop=True)


# --- four per-frame phase flags: home/away x attacking/defending -------------------------------
PHASE_COLUMNS = ["home_attacking", "home_defending", "away_attacking", "away_defending"]
PHASE_ATTACKING = {"build_up", "progression", "long_ball", "creating_chances", "ball_in_final_third",
                   "ball_in_the_box", "goals", "attacking_transition"}
PHASE_DEFENDING = {"defensive_transition", "defending_in_defensive_third",
                   "defending_in_middle_third", "defending_in_attacking_third"}


def build_phase_flags(df: pd.DataFrame, smart_tag: pd.DataFrame, params: dict,
                      match_info: dict) -> pd.DataFrame:
    """Add home_attacking, home_defending, away_attacking, away_defending (0/1, int8) to df.

    A flag is 1 on a frame when at least one tag of that phase and that team covers the frame
    (overlapping tags of the same team just count once). Frames outside every tag are 0.

    Same inputs as build_tag_features (frame column, smart_tag CSV, match_info for the team names).
    Optional params:
      attacking_codes / defending_codes   sets of code keys (lower case, underscores) replacing
                                          PHASE_ATTACKING / PHASE_DEFENDING. PLAYERS IN THE BOX is
                                          not in the default attacking set; add "players_in_the_box"
                                          to include it.
      defending_team_means                "defender" (default: Team on a Defending tag is the team
                                          that defends) or "possessor" (Team is the team with the
                                          ball, so the flag goes to the other team). Check this with
                                          smart_tags_hypothesis.py before trusting the defending flags.
    Tags without a recognisable team (N/A, empty) set no flag; the number skipped is logged.
    """
    attacking = {_code_key(c) for c in params.get("attacking_codes", PHASE_ATTACKING)}
    defending = {_code_key(c) for c in params.get("defending_codes", PHASE_DEFENDING)}
    means = params.get("defending_team_means", "defender")
    if means not in ("defender", "possessor"):
        raise ValueError("params['defending_team_means'] must be 'defender' or 'possessor', got %r" % means)

    frames, frame_col = _video_frames(df, params)
    n_frames = int(frames.max()) + 1
    sign_of = _make_signer(params, match_info)
    # difference arrays: +1 at start, -1 at end (end exclusive); positive running sum = flag on
    diff = {c: np.zeros(n_frames + 1, np.int32) for c in PHASE_COLUMNS}
    no_team = 0
    for ev in _load_events(smart_tag):
        key = _code_key(ev["code"])
        phase = "attacking" if key in attacking else "defending" if key in defending else None
        if phase is None:
            continue
        sign = sign_of(_tag(ev, "Team"))
        if sign == 0:
            no_team += 1
            continue
        if phase == "defending" and means == "possessor":
            sign = -sign
        a, b = max(ev["start_frame"], 0), min(ev["end_frame"], n_frames)
        if b <= a:
            continue
        col = ("home_" if sign > 0 else "away_") + phase
        diff[col][a] += 1
        diff[col][b] -= 1
    if no_team:
        log.warning("phase_flags: %d attacking/defending tag(s) without a recognisable team ignored", no_team)

    out = df.copy()
    for col in PHASE_COLUMNS:
        on = (np.cumsum(diff[col][:n_frames]) > 0).astype(np.int8)
        out[col] = on[frames]
    log.info("phase_flags: frame column %r, flagged frames: %s", frame_col,
             {c: int(out[c].sum()) for c in PHASE_COLUMNS})
    return out
# Frame Seeker

Single-page frame-seeking tool for the tactical video. Enter a frame ID to seek the tactical player to that frame and pause; the page also shows the current frame's smart-tagging labels and tracking data. A separate Broadcast player displays the match broadcast video below.

## Prerequisites

- **Node.js** (includes `npx`) — any recent LTS version.
  - [Download Node.js](https://nodejs.org/) (choose the LTS version)
  - To verify installation, run: `node --version` and `npx --version`
- A modern browser (Chrome recommended).
- Internet access: Omakase Player and hls.js are loaded from the jsDelivr CDN.
- The tactical video at `data/tactical/BAR-ATH_tactical.mp4` and broadcast video at
  `data/broadcast/BAR-ATH_broadcast.mp4` (relative to the repo root).
- The smart-tagging data at `data/tactical/BAR-ATH_tactical_smart_tagging.xml`.
- The tracking CSV at `data/tactical/BAR-ATH_tactical_ds.csv`. If it does not exist, run
  [`convert_atd_to_csv.ipynb`](../starter/convert_atd_to_csv.ipynb) first.

## Run

Serve the **repo root** (not `frame-seeker/`), because the page loads the video from `../data/`. The server must support HTTP range requests so the browser can fetch only the parts of the video needed when seeking to a frame, rather than downloading the whole file. Python's `http.server` does not.

```
npx http-server . -p 8000 -c-1
```

If the server is already running, skip the command. Then open
<http://localhost:8000/frame-seeker/>.

`-c-1` disables caching so edits show up on a normal reload.

## Usage

Enter a frame ID (1 to 199319) and click **Go**. The video pauses on that frame. An unknown ID shows "Frame ID does not exist!".

Frame IDs are 1-based and match the ATD data: `video_frame = frame_id - 1` at 25 fps.

## Notes

- Opening `index.html` directly (`file://`) will not work; the page needs a server.
- The "AudioContext was not allowed to start" console warning is harmless.

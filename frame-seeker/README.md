# Frame Seeker

Single-page video player (Omakase Player) that seeks to an exact frame ID and pauses there.

## Prerequisites

- **Node.js** (includes `npx`) — any recent LTS version.
  - [Download Node.js](https://nodejs.org/) (choose the LTS version)
  - To verify installation, run: `node --version` and `npx --version`
- A modern browser (Chrome recommended).
- Internet access: Omakase Player and hls.js are loaded from the jsDelivr CDN.
- The video at `data/tactical/BAR-ATH_tactical.mp4` (relative to the repo root).
- The CSV at `data/tactical/BAR-ATH_tactical_ds.csv`. If it does not exist, run
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

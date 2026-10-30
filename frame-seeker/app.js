import { OmakasePlayer } from '@omakase-player';

const VIDEO_URL = '../data/tactical/BAR-ATH_tactical.mp4';
const TAGS_URL = '../data/tactical/BAR-ATH_tactical_smart_tagging.xml';
const FPS = 25;
const TOTAL_FRAMES = 199319;
const DS_URL = '../data/tactical/BAR-ATH_tactical_ds.csv';
const trackingTable = document.getElementById('tracking-table');
// Raw CSV lines; index 0 is the header, so index === frame ID.
let dsLines = [];
let dsHeader = [];
let dsState = 'loading';
let lastTime = null;
const currentFrameOutput = document.getElementById('current-frame');
const frameTags = document.getElementById('frame-tags');
let tagInstances = [];
let tagLoadFailed = false;

function renderFrameTags(time) {
  frameTags.replaceChildren();

  if (tagLoadFailed) {
    frameTags.textContent = 'Tag data unavailable';
    return;
  }

  const matches = tagInstances.filter((instance) => time >= instance.start && time <= instance.end);
  if (matches.length === 0) {
    frameTags.textContent = 'No tag';
    return;
  }

  for (const instance of matches) {
    const entry = document.createElement('div');
    entry.className = 'tag-entry';

    const code = document.createElement('p');
    code.textContent = `Code: ${instance.code}`;
    entry.append(code);

    for (const label of instance.labels) {
      const labelLine = document.createElement('p');
      labelLine.textContent = `Label - ${label.group}: ${label.text} (ID: ${instance.id})`;
      entry.append(labelLine);
    }

    frameTags.append(entry);
  }
}

let objectSort = { column: 0, dir: 1 };

function compareValues(a, b) {
  const an = a === '' ? NaN : Number(a);
  const bn = b === '' ? NaN : Number(b);
  if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
  if (a === b) return 0;
  if (a === '') return 1;
  if (b === '') return -1;
  return String(a).localeCompare(String(b));
}

function makeTable(headers, rows, onSort) {
  const table = document.createElement('table');
  const head = table.createTHead().insertRow();
  headers.forEach((h, i) => {
    const th = Object.assign(document.createElement('th'), { textContent: h });
    if (onSort) {
      th.style.cursor = 'pointer';
      th.tabIndex = 0;
      th.title = `Sort by ${h}`;
      const active = objectSort.column === i;
      th.setAttribute('aria-sort', active ? (objectSort.dir === 1 ? 'ascending' : 'descending') : 'none');
      if (active) th.textContent = `${h} ${objectSort.dir === 1 ? '▲' : '▼'}`;
      const activate = () => {
        objectSort = { column: i, dir: active ? -objectSort.dir : 1 };
        onSort();
      };
      th.addEventListener('click', activate);
      th.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          activate();
        }
      });
    }
    head.append(th);
  });
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const v of row) tr.insertCell().textContent = v;
  }
  return table;
}

const fmt = (v) => (v === undefined || v === '' ? '' : String(Number.isNaN(Number(v)) ? v : Math.round(Number(v) * 1000) / 1000));

function renderTracking(frameId) {
  trackingTable.replaceChildren();
  if (dsState === 'error') {
    trackingTable.textContent = 'Tracking data unavailable';
    return;
  }
  const line = dsLines[frameId];
  if (!line) {
    trackingTable.textContent = dsState === 'loading' ? 'Loading tracking data...' : 'No tracking data';
    return;
  }

  const cells = line.split(',');
  const col = Object.fromEntries(dsHeader.map((name, i) => [name, cells[i]]));
  const info = ['period_id', 'timestamp', 'frame_id', 'ball_x', 'ball_y', 'ball_speed'];
  trackingTable.append(makeTable(info, [info.map((k) => fmt(col[k]))]));

  const rows = [];
  for (let id = 1; col[`${id}_x`] !== undefined; id++) {
    const values = ['x', 'y', 's', 'x_m', 'y_m', 'ball_dist_m', 'team']
      .map((k) => col[`${id}_${k}`]);
    if (values.some((v) => v !== '')) rows.push([id, ...values.map(fmt)]);
  }
  const sorted = rows.slice().sort((a, b) => objectSort.dir * compareValues(a[objectSort.column], b[objectSort.column]));
  trackingTable.append(makeTable(
    ['object', 'x', 'y', 's', 'x_m', 'y_m', 'ball_dist_m', 'team'],
    sorted,
    () => renderTracking(frameId),
  ));
}

function updateFrameDetails(time) {
  if (!Number.isFinite(time) || time < 0) return;
  lastTime = time;

  const frameId = Math.min(TOTAL_FRAMES, Math.floor(time * FPS) + 1);
  currentFrameOutput.value = String(frameId);
  currentFrameOutput.textContent = String(frameId);
  renderFrameTags(time);
  renderTracking(frameId);
}

async function loadTrackingData() {
  try {
    const response = await fetch(DS_URL);
    if (!response.ok) throw new Error(`Tracking request failed: ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let rest = '';
    let first = true;
    let sinceRender = 0;
    for (;;) {
      const { done, value } = await reader.read();
      rest += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      const parts = rest.split('\n');
      rest = done ? '' : parts.pop();
      if (done && parts[parts.length - 1] === '') parts.pop();
      for (const part of parts) {
        const line = part.replace(/\r$/, '');
        if (first) {
          dsHeader = line.split(',');
          first = false;
        }
        dsLines.push(line);
      }
      sinceRender += parts.length;
      if (done || sinceRender > 20000) {
        sinceRender = 0;
        if (done) dsState = 'ready';
        if (lastTime !== null) updateFrameDetails(lastTime);
      }
      if (done) break;
    }
  } catch (error) {
    dsState = 'error';
    console.error('Failed to load tracking data', error);
    if (lastTime !== null) updateFrameDetails(lastTime);
    else trackingTable.textContent = 'Tracking data unavailable';
  }
}

loadTrackingData();

fetch(TAGS_URL)
  .then((response) => {
    if (!response.ok) throw new Error(`Tag request failed: ${response.status}`);
    return response.text();
  })
  .then((xmlText) => {
    const xml = new DOMParser().parseFromString(xmlText, 'application/xml');
    if (xml.querySelector('parsererror')) throw new Error('Tag XML could not be parsed');

    tagInstances = [...xml.getElementsByTagName('instance')].map((instance) => ({
      start: Number(instance.querySelector('start')?.textContent),
      end: Number(instance.querySelector('end')?.textContent),
      id: instance.querySelector('ID')?.textContent.trim() ?? '',
      code: instance.querySelector('code')?.textContent.trim() ?? '',
      labels: [...instance.getElementsByTagName('label')].map((label) => ({
        group: label.querySelector('group')?.textContent.trim() ?? '',
        text: label.querySelector('text')?.textContent.trim() ?? '',
      })),
    })).filter((instance) => Number.isFinite(instance.start) && Number.isFinite(instance.end));

    const video = document.querySelector('#omakase-player video');
    if (video) updateFrameDetails(video.currentTime);
  })
  .catch((error) => {
    tagLoadFailed = true;
    console.error('Failed to load frame tags', error);
    frameTags.textContent = 'Tag data unavailable';
  });

// Both spellings are passed because the docs use each; the unknown one is ignored.
const player = new OmakasePlayer({
  playerHtmlElementId: 'omakase-player',
  playerHTMLElementId: 'omakase-player',
});
player.loadMainMedia(VIDEO_URL, { frameRate: FPS }).subscribe({
  error: (err) => console.error('Failed to load video', err),
});

function attachVideoTimeListeners() {
  const video = document.querySelector('#omakase-player video');
  if (!video) return false;

  const update = () => updateFrameDetails(video.currentTime);
  video.addEventListener('timeupdate', update);
  video.addEventListener('seeked', update);
  video.addEventListener('loadedmetadata', update);
  update();
  return true;
}

if (!attachVideoTimeListeners()) {
  const playerObserver = new MutationObserver(() => {
    if (attachVideoTimeListeners()) playerObserver.disconnect();
  });
  playerObserver.observe(document.getElementById('omakase-player'), { childList: true, subtree: true });
}

document.getElementById('seek-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const raw = document.getElementById('frame-id').value.trim();
  const frameId = Number(raw);

  if (!/^\d+$/.test(raw) || frameId < 1 || frameId > TOTAL_FRAMES) {
    alert('Frame ID does not exist!');
    return;
  }

  // Frame ID is 1-based; aim at the middle of the frame to avoid rounding onto a neighbour.
  const time = (frameId - 1 + 0.5) / FPS;
  player.player.pause();
  player.player.seekTo(time).subscribe();
});

"""Overlap of your labels with the ground truth on the public 5-minute slice (frames 11185-18684).

    python plot_overlap.py --pred ../evaluation/results/hybrid_results.csv --gt path/to/public_ground_truth.csv
    python plot_overlap.py --pred output/hybrid_results.csv --gt ... --out overlap.png --start 11185 --end 18684

Both files need one row per frame with a frame column (video_frame / frame / frame_id / ...) and one column whose values
are H / A / D / X; the columns are found automatically (override with --pred-col / --gt-col / --frame-col).  A file
without a frame column is read row by row (frame = row number, or --start when it has exactly the slice's frames).
If --gt is left out, a file with "truth", "ground", "gt" or "label" in its name under ../evaluation is used when there is
exactly one.

The figure has two panels (colour-blind-safe palette, every state also named in the legend):
  1. FRAMES   ground truth, prediction, and where they agree, over the slice.
              agreement colours: match, wrong team (H<->A), dead/alive mismatch (D vs H or A), prediction abstains (X)
  2. SPELLS   the possessions as bars: a spell is a run of one team with X carried forward and D ending it (the scorer's
              convention as I understand it).  Spells with the same team and IoU >= --iou are matched one-to-one
              (greedy by IoU); matched ones have a solid outline, the others a dashed one.  The IoU is written on the pair.
With --tags FILE (or data/tactical/BAR-ATH_smart_tagging_ds.csv in the repo) every Smart Tag code gets a row under the agreement
row (Set Pieces first; tags of one code that overlap in time are stacked).  Set pieces are coloured by kind (corner, free kick,
throw-in, goal kick, kick-off ...), other codes by Type.  In the page the tooltip lists every tag that is on at the frame and how
many seconds after its start (useful for judging the dead-ball cutoffs); each code has a checkbox to hide it.
With --html FILE it also writes an INTERACTIVE page (one self-contained file, no internet needed): hover for the labels and
spell details, wheel to zoom, drag to pan, click a spell to zoom to it, buttons to jump between mismatches, tables of all
spells.  --no-png skips the static picture.
It also prints the frame-level confusion matrix and the spell counts.  Cross-check against your scoreboard row: accuracy,
D recall / precision and spell_f1_at_headline should agree (if they do not, the file or the slice differs).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

FPS = 25
STATES = ("H", "A", "D", "X")
FRAME_NAMES = ("video_frame", "frame", "frame_id", "frame_idx", "frame_number")
LABEL_HINTS = ("label", "possession", "hybrid_possession", "pred", "prediction", "gt", "truth", "ground_truth")
COLOR = {"H": "#0072B2", "A": "#E69F00", "D": "#4D4D4D", "X": "#D9D9D9"}
AGREE = {"match": "#009E73", "wrong team": "#D55E00", "dead/alive": "#CC79A7", "abstain (X)": "#F0E442"}


def find_columns(path, frame_col=None, label_col=None):
    header = list(pd.read_csv(path, nrows=0).columns)
    frame = frame_col or next((c for c in FRAME_NAMES if c in header), None)
    if label_col:
        return frame, label_col
    sample = pd.read_csv(path, nrows=2000, dtype=str, keep_default_na=False)
    labelish = [c for c in header if c != frame
                and sample[c].str.strip().str.upper().isin(STATES).mean() > 0.95]
    if not labelish:
        sys.exit("%s: no column with H / A / D / X values (columns: %s)" % (path, header[:12]))
    pick = next((c for h in LABEL_HINTS for c in labelish if c.lower() == h), labelish[0])
    if len(labelish) > 1:
        print("%s: several label columns %s, using %r (override with --pred-col / --gt-col)" % (path, labelish, pick))
    return frame, pick


def read_labels(path, start, end, frame_col=None, label_col=None):
    """-> array of H/A/D/X for frames start..end (inclusive); frames absent from the file become 'X'."""
    frame, label = find_columns(path, frame_col, label_col)
    df = pd.read_csv(path, usecols=[frame, label] if frame else [label], dtype={label: str}, keep_default_na=False)
    lab = df[label].str.strip().str.upper()
    if frame:
        idx = pd.to_numeric(df[frame], errors="coerce")
    else:
        idx = pd.Series(np.arange(len(df)) + (start if len(df) == end - start + 1 else 0))
        if len(df) == end - start + 1:
            print("%s: no frame column, %d rows = the slice, assuming it starts at frame %d" % (path, len(df), start))
    series = pd.Series(lab.to_numpy(), index=idx.to_numpy()).dropna()
    series = series[~series.index.duplicated(keep="first")]
    frames = np.arange(start, end + 1)
    out = series.reindex(frames).fillna("X").to_numpy(dtype="<U1")
    missing = int(series.reindex(frames).isna().sum())
    if missing:
        print("%s: %d of %d slice frames are not in the file (shown as X)" % (path, missing, len(frames)))
    bad = set(out) - set(STATES)
    if bad:
        sys.exit("%s: unexpected label values %s" % (path, sorted(bad)))
    return out


def runs(labels):
    """[(start, end_exclusive, label)] of consecutive equal labels."""
    cut = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b), labels[a]) for a, b in zip(cut[:-1], cut[1:])]


def spells(labels):
    """Possessions: runs of one team; X is carried forward, D ends the spell.  -> [(start, end_exclusive, team)]"""
    out, team, start = [], None, 0
    for i, lab in enumerate(labels):
        if lab == "D":
            if team is not None:
                out.append((start, i, team))
            team = None
        elif lab in ("H", "A") and lab != team:
            if team is not None:
                out.append((start, i, team))
            team, start = lab, i
    if team is not None:
        out.append((start, len(labels), team))
    return out


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def match(gt, pred, threshold):
    """One-to-one matching of same-team spells by IoU >= threshold, greedy by IoU.  -> {gt index: (pred index, IoU)}"""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if g[2] == p[2] and iou(g, p) >= threshold), reverse=True)
    used_g, used_p, out = set(), set(), {}
    for score, i, j in pairs:
        if i not in used_g and j not in used_p:
            out[i] = (j, score)
            used_g.add(i); used_p.add(j)
    return out


def agreement(gt, pred):
    kind = np.full(len(gt), "match", dtype=object)
    wrong = gt != pred
    kind[wrong & np.isin(gt, ["H", "A"]) & np.isin(pred, ["H", "A"])] = "wrong team"
    kind[wrong & ((gt == "D") != (pred == "D")) & (pred != "X")] = "dead/alive"
    kind[wrong & (pred == "X")] = "abstain (X)"
    return kind


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prediction vs ground truth</title>
<style>
:root{--bg:#ffffff;--fg:#1d1d1f;--muted:#6b6b70;--grid:#e3e3e6;--panel:#f4f4f6;--line:#d6d6db;
--H:#0072B2;--A:#E69F00;--D:#4D4D4D;--X:#D9D9D9;--match:#009E73;--wrong:#D55E00;--dead:#CC79A7;--abst:#F0E442;
--bad:#B00020;--deadbg:#ececef;--tipbg:#ffffff;
--t-corner:#B9A7E8;--t-free:#E3B98F;--t-throw:#8ED1D3;--t-goalkick:#B8C4CE;--t-kickoff:#E8A09C;--t-penalty:#F2C6DE;--t-other:#D5D5D5}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf0;--muted:#9a9aa3;--grid:#34343a;--panel:#202026;--line:#3a3a42;
--D:#8d8d96;--X:#55555c;--bad:#ff6b81;--deadbg:#26262c;--tipbg:#26262c}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1{font-size:18px;margin:0 0 2px}
.sub{color:var(--muted);margin:0 0 12px;font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.card{background:var(--panel);border-radius:8px;padding:6px 12px;min-width:110px}
.card b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
.card span{color:var(--muted);font-size:12px}
.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:6px 0 8px}
.bar label{display:flex;gap:5px;align-items:center;cursor:pointer;user-select:none}
#tagboxes{display:contents}
button,select{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:3px 10px;cursor:pointer}
button:hover{border-color:var(--muted)}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--H);outline-offset:1px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:8px 0;font-size:12.5px;color:var(--muted)}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-2px}
#wrap{position:relative;border:1px solid var(--line);border-radius:10px;padding:6px 0 0;background:var(--bg)}
#chart{display:block;width:100%;cursor:grab;touch-action:none;user-select:none}
#chart.drag{cursor:grabbing}
#overview{display:block;width:100%;margin-top:4px;cursor:pointer;touch-action:none}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--muted);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}
.rowl{fill:var(--fg);font-size:12px}
.sH{fill:var(--H)}.sA{fill:var(--A)}.sD{fill:var(--D)}.sX{fill:var(--X)}
.match{fill:var(--match)}.wrong{fill:var(--wrong)}.dead{fill:var(--dead)}.abst{fill:var(--abst)}
.deadbg{fill:var(--deadbg)}
.tg{stroke:var(--fg);stroke-width:1;opacity:.92}
.tg-corner{fill:var(--t-corner)}.tg-free{fill:var(--t-free)}.tg-throw{fill:var(--t-throw)}.tg-goalkick{fill:var(--t-goalkick)}
.tg-kickoff{fill:var(--t-kickoff)}.tg-penalty{fill:var(--t-penalty)}.tg-other{fill:var(--t-other)}
.tgtext{fill:#111;font-size:11px;pointer-events:none}
.grptitle{fill:var(--muted);font-size:11px;font-weight:700;letter-spacing:.03em;text-transform:uppercase}
.grp{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;flex-basis:100%}
.grp b{font-weight:700}
.sp{stroke-linejoin:round}
.sp.H{fill:var(--H)}.sp.A{fill:var(--A)}
.sp.ok{stroke:var(--fg);stroke-width:1.5;opacity:.95}
.sp.no{stroke:var(--bad);stroke-width:1.2;stroke-dasharray:4 3;opacity:.45}
.sp.hot{stroke-width:3;opacity:1}
.iou{fill:var(--fg);font-size:10.5px}
.cursor{stroke:var(--fg);stroke-width:1;stroke-dasharray:3 3;opacity:.6}
.brush{fill:rgba(0,114,178,.18);stroke:var(--H);stroke-width:1.5}
#tip{position:fixed;pointer-events:none;background:var(--tipbg);border:1px solid var(--line);border-radius:8px;padding:7px 10px;
font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.18);max-width:290px;display:none;z-index:5}
#tip .k{color:var(--muted)}
.chip{display:inline-block;min-width:1.5em;text-align:center;border-radius:4px;padding:0 5px;color:#fff;font-weight:600}
.chip.H{background:var(--H)}.chip.A{background:var(--A);color:#222}.chip.D{background:var(--D)}.chip.X{background:var(--X);color:#222}
.tables{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-top:16px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
caption{text-align:left;font-weight:600;padding:0 0 4px}
th,td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--bg)}
tbody tr{cursor:pointer}
tbody tr:hover,tbody tr.hot{background:var(--panel)}
.scroll{max-height:300px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.no-match{color:var(--bad)}
.hint{color:var(--muted);font-size:12.5px;margin-top:8px}
</style>
</head>
<body>
<h1 id="title"></h1>
<p class="sub" id="sub"></p>
<div class="cards" id="cards"></div>
<div class="bar">
  <label><input type="checkbox" data-row="gt" checked> Ground truth</label>
  <label><input type="checkbox" data-row="pred" checked> Prediction</label>
  <label><input type="checkbox" data-row="agree" checked> Agreement</label>
  <span id="tagboxes"></span>
  <label id="sidebox" style="display:none"><input type="checkbox" id="sideH" checked> home tags</label>
  <label id="sidebox2" style="display:none"><input type="checkbox" id="sideA" checked> away tags</label>
  <label><input type="checkbox" data-row="spells" checked> Spells</label>
  <span style="flex:1"></span>
  <button id="zin" title="Zoom in (+)">+</button><button id="zout" title="Zoom out (-)">&minus;</button>
  <button id="reset" title="Show the whole slice (0)">Reset</button>
  <span class="k">Mismatches &ge;</span>
  <select id="minlen"><option value="0.2">0.2 s</option><option value="0.5">0.5 s</option><option value="1" selected>1 s</option><option value="2">2 s</option><option value="5">5 s</option></select>
  <button id="prev" title="Previous mismatch (p)">&#9664; prev</button>
  <button id="next" title="Next mismatch (n)">next &#9654;</button>
  <span id="mmstatus" class="sub" style="margin:0"></span>
</div>
<div id="wrap"><svg id="chart" role="img" aria-label="Timeline of ground truth, prediction and agreement"></svg><svg id="overview" height="46"></svg></div>
<div class="legend" id="legend"></div>
<div class="hint">Wheel: zoom at the cursor &middot; drag: pan &middot; click a spell: zoom to it &middot; double-click: reset &middot; keys: &larr; &rarr; pan, + &minus; zoom, 0 reset, n / p next / previous mismatch. Hover a spell to see its partner.</div>
<div class="tables">
  <div><table id="tg"><caption>Truth spells</caption><thead><tr><th>#</th><th>team</th><th>start</th><th>end</th><th>len</th><th>match</th><th>IoU</th></tr></thead><tbody></tbody></table></div>
  <div><table id="tp"><caption>Predicted spells</caption><thead><tr><th>#</th><th>team</th><th>start</th><th>end</th><th>len</th><th>match</th><th>IoU</th></tr></thead><tbody></tbody></table></div>
</div>
<div id="tip"></div>
<script>
(function(){
const D = /*__DATA__*/null;
const N = D.n, FPS = D.fps;
const $ = id => document.getElementById(id);
const chart = $('chart'), ov = $('overview'), tip = $('tip');
const KCLS = {'match':'match','wrong team':'wrong','dead/alive':'dead','abstain (X)':'abst'};
const NAMES = {H:'H (team A)', A:'A (team B)', D:'D dead ball', X:'X unknown'};
const RIGHT = 14, TOP = 8, AXIS = 30;
let v0 = 0, v1 = N, hover = null, cursorF = null, rowsInfo = [], mm = [], mmIndex = -1;
const show = {gt:true, pred:true, agree:true, spells:true};
const TROWS = D.tagrows || [], showTag = {}, showGroup = {}, GROUPS = D.groups || [], showSide = {H:true, A:true, '':true}, LANE = 22;
TROWS.forEach(r => { showTag[r.code] = true; showGroup[r.group] = true; });
const LEFT = Math.min(230, Math.max(112, 24 + 6.6 * Math.max(0, ...TROWS.map(r => r.code.length))));          // room for the longest tag code

function fmt(f, fine){
  const s = f / FPS, m = Math.floor(s / 60), r = s - 60 * m;
  return m + ':' + (fine ? r.toFixed(1).padStart(4, '0') : String(Math.floor(r)).padStart(2, '0'));
}
function dur(a, b){ return ((b - a) / FPS).toFixed(1) + ' s'; }
function idxAt(list, f){
  let lo = 0, hi = list.length - 1;
  while (lo <= hi){
    const mid = (lo + hi) >> 1, r = list[mid];
    if (f < r[0]) hi = mid - 1; else if (f >= r[1]) lo = mid + 1; else return mid;
  }
  return -1;
}
function clampView(a, b){
  let span = Math.min(Math.max(b - a, 50), N);
  a = Math.max(0, Math.min(a, N - span));
  v0 = a; v1 = a + span;
}
function layout(){
  const rows = []; let y = TOP;
  if (show.gt){ rows.push({k:'gt', label:'Ground truth', y, h:34}); y += 42; }
  if (show.pred){ rows.push({k:'pred', label:'Prediction', y, h:34}); y += 42; }
  if (show.agree){ rows.push({k:'agree', label:'Agreement', y, h:22}); y += 34; }
  let lastGroup = null;
  for (const tr of TROWS){
    if (!showTag[tr.code] || !showGroup[tr.group]) continue;
    if (tr.group !== lastGroup){
      lastGroup = tr.group;
      rows.push({k:'grp', label:(GROUPS.find(g => g[0] === tr.group) || [0, tr.group])[1], y: y + 4, h:14}); y += 24;
    }
    const h = tr.lanes * LANE - 2; rows.push({k:'tag', code:tr.code, label:tr.code, y, h, tr}); y += h + 12;
  }
  if (show.spells){
    y += 8; rows.push({k:'sg', label:'Truth spells', y, h:30});
    rows.push({k:'sp', label:'Predicted spells', y:y + 46, h:30}); y += 46 + 30 + 6;
  }
  return {rows, height: y + AXIS};
}
function partner(){
  if (!hover) return [-1, -1];
  return hover.side === 'g' ? [hover.i, D.gs[hover.i][3]] : [D.ps[hover.i][3], hover.i];
}

function render(){
  const W = chart.clientWidth || 900, pw = W - LEFT - RIGHT;
  const L = layout(); rowsInfo = L.rows;
  chart.setAttribute('height', L.height);
  const X = f => LEFT + (f - v0) / (v1 - v0) * pw;
  const cx = (a, b) => [Math.max(X(a), LEFT), Math.min(X(b), LEFT + pw)];
  const plotBottom = L.height - AXIS;
  let s = '';
  const span = (v1 - v0) / FPS, steps = [1, 2, 5, 10, 15, 30, 60, 120, 300];
  const step = steps.find(t => span / t <= 12) || 300;
  for (let t = Math.ceil(v0 / FPS / step) * step; t * FPS <= v1; t += step){
    const x = X(t * FPS);
    s += `<line class="grid" x1="${x}" x2="${x}" y1="${TOP}" y2="${plotBottom}"/><text class="tick" x="${x}" y="${plotBottom + 17}" text-anchor="middle">${fmt(t * FPS, false)}</text>`;
  }
  s += `<line class="axis" x1="${LEFT}" x2="${LEFT + pw}" y1="${plotBottom}" y2="${plotBottom}"/>`;
  const bars = (list, y, h, cls) => {
    let o = '';
    for (const r of list){
      if (r[1] <= v0 || r[0] >= v1) continue;
      const [a, b] = cx(r[0], r[1]);
      o += `<rect class="${cls(r)}" x="${a.toFixed(2)}" y="${y}" width="${Math.max(b - a, 0.6).toFixed(2)}" height="${h}"/>`;
    }
    return o;
  };
  const [hotG, hotP] = partner();
  for (const r of rowsInfo){
    if (r.k === 'grp'){
      s += `<line class="grid" x1="6" x2="${LEFT + pw}" y1="${r.y - 2}" y2="${r.y - 2}"/><text class="grptitle" x="8" y="${r.y + 11}">${r.label}</text>`;
      continue;
    }
    s += `<text class="rowl" x="${LEFT - 10}" y="${r.y + r.h / 2 + 4}" text-anchor="end">${r.label}</text>`;
    if (r.k === 'gt') s += bars(D.gt, r.y, r.h, q => 's' + q[2]);
    else if (r.k === 'pred') s += bars(D.pred, r.y, r.h, q => 's' + q[2]);
    else if (r.k === 'agree') s += bars(D.kind, r.y, r.h, q => KCLS[q[2]]);
    else if (r.k === 'tag'){
      for (const q of r.tr.items){
        if (q[1] <= v0 || q[0] >= v1 || !showSide[q[6]]) continue;
        const [a, b] = cx(q[0], q[1]), ty = r.y + q[5] * LANE, th = LANE - 2, w = Math.max(b - a, 1);
        s += `<rect class="tg" style="fill:${q[4]}" x="${a.toFixed(2)}" y="${ty}" width="${w.toFixed(2)}" height="${th}" rx="3"/>`;
        if (q[6]) s += `<rect class="tg" style="fill:var(--${q[6]})" x="${a.toFixed(2)}" y="${ty}" width="${Math.min(w, 22).toFixed(2)}" height="${th}" rx="3"/>`
                     + (w > 24 ? `<text x="${(a + 11).toFixed(2)}" y="${ty + th / 2 + 4}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:700;pointer-events:none">${q[6]}</text>` : '');
        const room = b - a - (q[6] ? 24 : 0);
        if (room > 6.2 * Math.min(q[2].length, 9)) s += `<text class="tgtext" x="${(a + b) / 2 + (q[6] ? 11 : 0)}" y="${ty + th / 2 + 4}" text-anchor="middle">${q[2]}</text>`;
      }
    }
    else {
      const list = r.k === 'sg' ? D.gs : D.ps, labels = r.k === 'sg' ? D.gt : D.pred, hotI = r.k === 'sg' ? hotG : hotP;
      s += bars(labels.filter(q => q[2] === 'D'), r.y - 3, r.h + 6, () => 'deadbg');
      list.forEach((q, i) => {
        if (q[1] <= v0 || q[0] >= v1) return;
        const [a, b] = cx(q[0], q[1]);
        s += `<rect class="sp ${q[2]} ${q[3] >= 0 ? 'ok' : 'no'}${i === hotI ? ' hot' : ''}" x="${a.toFixed(2)}" y="${r.y}" width="${Math.max(b - a, 0.6).toFixed(2)}" height="${r.h}" rx="2"/>`;
      });
    }
  }
  if (show.spells){
    const g = rowsInfo.find(r => r.k === 'sg');
    D.gs.forEach(q => {
      if (q[3] < 0) return;
      const p = D.ps[q[3]], a = Math.max(q[0], p[0]), b = Math.min(q[1], p[1]);
      if (b <= v0 || a >= v1) return;
      const [xa, xb] = cx(a, b);
      if (xb - xa < 30) return;
      s += `<text class="iou" x="${(xa + xb) / 2}" y="${g.y + g.h + 12}" text-anchor="middle">${q[4].toFixed(2)}</text>`;
    });
  }
  if (cursorF !== null && cursorF >= v0 && cursorF <= v1){
    const x = X(cursorF);
    s += `<line class="cursor" x1="${x}" x2="${x}" y1="${TOP}" y2="${plotBottom}"/>`;
  }
  chart.innerHTML = s;
  renderOverview();
  document.querySelectorAll('tbody tr').forEach(tr => {
    const side = tr.dataset.side, i = +tr.dataset.i;
    tr.classList.toggle('hot', (side === 'g' && i === hotG) || (side === 'p' && i === hotP));
  });
}
function renderOverview(){
  const W = ov.clientWidth || 900, pw = W - LEFT - RIGHT, X = f => LEFT + f / N * pw;
  let s = `<text class="rowl" x="${LEFT - 10}" y="27" text-anchor="end">Overview</text>`;
  for (const r of D.kind) s += `<rect class="${KCLS[r[2]]}" x="${X(r[0]).toFixed(2)}" y="12" width="${Math.max(X(r[1]) - X(r[0]), 0.5).toFixed(2)}" height="18"/>`;
  s += `<rect class="brush" x="${X(v0)}" y="6" width="${Math.max(X(v1) - X(v0), 3)}" height="30" rx="3"/>`;
  ov.innerHTML = s;
}

function setView(a, b){ clampView(a, b); render(); }
function zoomAt(f, factor){
  const span = (v1 - v0) * factor, t = (f - v0) / (v1 - v0);
  setView(f - t * span, f + (1 - t) * span);
}
function zoomTo(a, b){
  const pad = Math.max((b - a) * 0.25, 25);
  setView(a - pad, b + pad);
}

function frameAt(e){
  const rect = chart.getBoundingClientRect(), pw = rect.width - LEFT - RIGHT;
  return {f: v0 + (e.clientX - rect.left - LEFT) / pw * (v1 - v0), x: e.clientX - rect.left, y: e.clientY - rect.top, pw};
}
function chip(l){ return `<span class="chip ${l}">${l}</span>`; }
function spellText(side, i){
  const q = (side === 'g' ? D.gs : D.ps)[i];
  let t = `<div><span class="k">${side === 'g' ? 'Truth' : 'Predicted'} spell #${i + 1}:</span> ${chip(q[2])} ${fmt(q[0], true)} &ndash; ${fmt(q[1], true)} (${dur(q[0], q[1])})</div>`;
  if (q[3] >= 0){
    const o = (side === 'g' ? D.ps : D.gs)[q[3]];
    t += `<div><span class="k">matched with</span> #${q[3] + 1} (${fmt(o[0], true)} &ndash; ${fmt(o[1], true)}), IoU ${q[4].toFixed(2)}</div>`;
  } else t += `<div class="no-match">unmatched (no same-team spell with IoU &ge; ${D.iou})</div>`;
  return t;
}
function onMove(e){
  const m = frameAt(e);
  if (drag){
    const df = -(e.clientX - drag.x) / m.pw * (drag.v1 - drag.v0);
    drag.moved = drag.moved || Math.abs(e.clientX - drag.x) > 3;
    clampView(drag.v0 + df, drag.v1 + df); render(); return;
  }
  if (m.x < LEFT || m.x > LEFT + m.pw){ tip.style.display = 'none'; cursorF = null; hover = null; render(); return; }
  cursorF = m.f;
  const row = rowsInfo.find(r => m.y >= r.y - 3 && m.y <= r.y + r.h + 3);
  const f = Math.floor(m.f);
  const g = D.gt[idxAt(D.gt, f)], p = D.pred[idxAt(D.pred, f)], k = D.kind[idxAt(D.kind, f)];
  let html = `<div><b>${fmt(m.f, true)}</b> <span class="k">&middot; frame ${D.start + f}</span></div>`;
  html += `<div><span class="k">Truth</span> ${g ? chip(g[2]) + ' ' + NAMES[g[2]] : '-'}</div>`;
  html += `<div><span class="k">Prediction</span> ${p ? chip(p[2]) + ' ' + NAMES[p[2]] : '-'}</div>`;
  if (k) html += `<div><span class="k">Agreement</span> ${k[2]}</div>`;
  for (const tr of TROWS){
    for (const q of tr.items){
      if (f >= q[0] && f < q[1] && showSide[q[6]]) html += `<div><span class="k">${tr.code}</span> <b>${q[2]}</b>${q[3] ? ' (' + q[3] + (q[6] ? ', ' + (q[6] === 'H' ? 'home' : 'away') : '') + ')' : ''}, ${((f - q[0]) / FPS).toFixed(1)} s after its start (${dur(q[0], q[1])})</div>`;
    }
  }
  let newHover = null;
  if (row && (row.k === 'sg' || row.k === 'sp')){
    const side = row.k === 'sg' ? 'g' : 'p', i = idxAt(side === 'g' ? D.gs : D.ps, f);
    if (i >= 0){ newHover = {side, i}; html += '<hr style="border:0;border-top:1px solid var(--line);margin:6px 0">' + spellText(side, i); }
  }
  hover = newHover;
  tip.innerHTML = html; tip.style.display = 'block';
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - w - 8) + 'px';
  tip.style.top = Math.min(e.clientY + 14, window.innerHeight - h - 8) + 'px';
  render();
}
let drag = null;
chart.addEventListener('pointerdown', e => { chart.setPointerCapture(e.pointerId); drag = {x:e.clientX, v0, v1, moved:false}; chart.classList.add('drag'); });
chart.addEventListener('pointerup', e => {
  const wasClick = drag && !drag.moved; drag = null; chart.classList.remove('drag');
  if (wasClick && hover){
    const q = (hover.side === 'g' ? D.gs : D.ps)[hover.i]; zoomTo(q[0], q[1]);
  }
});
chart.addEventListener('pointermove', onMove);
chart.addEventListener('pointerleave', () => { if (!drag){ tip.style.display = 'none'; cursorF = null; hover = null; render(); } });
chart.addEventListener('wheel', e => { e.preventDefault(); zoomAt(frameAt(e).f, Math.exp(e.deltaY * 0.0015)); }, {passive:false});
chart.addEventListener('dblclick', () => setView(0, N));

let ovDrag = false;
function ovMove(e){
  const rect = ov.getBoundingClientRect(), pw = rect.width - LEFT - RIGHT;
  const c = (e.clientX - rect.left - LEFT) / pw * N, half = (v1 - v0) / 2;
  setView(c - half, c + half);
}
ov.addEventListener('pointerdown', e => { ov.setPointerCapture(e.pointerId); ovDrag = true; ovMove(e); });
ov.addEventListener('pointermove', e => { if (ovDrag) ovMove(e); });
ov.addEventListener('pointerup', () => { ovDrag = false; });

function buildMismatches(){
  const minLen = +$('minlen').value * FPS;
  mm = D.kind.filter(r => r[2] !== 'match' && r[1] - r[0] >= minLen);
  mmIndex = -1; $('mmstatus').textContent = mm.length + ' mismatch' + (mm.length === 1 ? '' : 'es');
}
function jump(dir){
  if (!mm.length) return;
  const c = (v0 + v1) / 2;
  let i;
  if (dir > 0){ i = mm.findIndex(r => (r[0] + r[1]) / 2 > c + 1); if (i < 0) i = 0; }
  else { i = -1; mm.forEach((r, k) => { if ((r[0] + r[1]) / 2 < c - 1) i = k; }); if (i < 0) i = mm.length - 1; }
  const r = mm[i], mid = (r[0] + r[1]) / 2, span = Math.max(300, (r[1] - r[0]) * 4);
  setView(mid - span / 2, mid + span / 2);
  $('mmstatus').textContent = (i + 1) + ' / ' + mm.length + ': ' + r[2] + ', ' + dur(r[0], r[1]) + ' at ' + fmt(r[0], false);
}
$('next').onclick = () => jump(1); $('prev').onclick = () => jump(-1);
$('minlen').onchange = buildMismatches;
$('zin').onclick = () => zoomAt((v0 + v1) / 2, 0.6); $('zout').onclick = () => zoomAt((v0 + v1) / 2, 1 / 0.6);
$('reset').onclick = () => setView(0, N);
document.querySelectorAll('input[data-row]').forEach(c => c.onchange = () => { show[c.dataset.row] = c.checked; render(); });
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  const span = v1 - v0;
  if (e.key === 'ArrowLeft') setView(v0 - span * 0.1, v1 - span * 0.1);
  else if (e.key === 'ArrowRight') setView(v0 + span * 0.1, v1 + span * 0.1);
  else if (e.key === '+' || e.key === '=') zoomAt((v0 + v1) / 2, 0.7);
  else if (e.key === '-') zoomAt((v0 + v1) / 2, 1 / 0.7);
  else if (e.key === '0') setView(0, N);
  else if (e.key === 'n') jump(1); else if (e.key === 'p') jump(-1);
});
window.addEventListener('resize', render);

function fillTable(id, list, side){
  const tb = document.querySelector('#' + id + ' tbody');
  tb.innerHTML = list.map((q, i) => `<tr data-side="${side}" data-i="${i}"><td>${i + 1}</td><td>${chip(q[2])}</td><td>${fmt(q[0], true)}</td><td>${fmt(q[1], true)}</td><td>${dur(q[0], q[1])}</td><td class="${q[3] >= 0 ? '' : 'no-match'}">${q[3] >= 0 ? '#' + (q[3] + 1) : 'none'}</td><td>${q[3] >= 0 ? q[4].toFixed(2) : ''}</td></tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr => {
    tr.onclick = () => { const q = list[+tr.dataset.i]; zoomTo(q[0], q[1]); window.scrollTo({top:0, behavior:'smooth'}); };
    tr.onmouseenter = () => { hover = {side, i:+tr.dataset.i}; render(); };
    tr.onmouseleave = () => { hover = null; render(); };
  });
}
$('title').textContent = D.title;
$('sub').textContent = 'frames ' + D.start + ' to ' + (D.start + N - 1) + ' (' + fmt(N, false) + ' at ' + FPS + ' fps)';
$('cards').innerHTML = D.stats.map(s => `<div class="card"><b>${s[1]}</b><span>${s[0]}</span></div>`).join('');
const sw = c => `<span><i style="background:var(--${c})"></i>`;
$('legend').innerHTML = [sw('H') + 'H (team A)</span>', sw('A') + 'A (team B)</span>', sw('D') + 'D dead ball</span>', sw('X') + 'X unknown</span>',
  sw('match') + 'match ' + D.share.match + '%</span>', sw('wrong') + 'wrong team ' + D.share.wrong + '%</span>',
  sw('dead') + 'dead/alive mismatch ' + D.share.dead + '%</span>', sw('abst') + 'prediction abstains ' + D.share.abst + '%</span>'].join('');
for (const [gid, gname] of GROUPS){
  const rows = TROWS.filter(r => r.group === gid);
  if (!rows.length) continue;
  const box = document.createElement('span'); box.className = 'grp';
  const gl = document.createElement('label');
  gl.innerHTML = '<input type="checkbox" checked> <b>' + gname + '</b>';
  box.appendChild(gl);
  const inputs = [];
  for (const tr of rows){
    const lab = document.createElement('label');
    lab.innerHTML = '<input type="checkbox" checked> ' + tr.code;
    lab.firstChild.onchange = e => { showTag[tr.code] = e.target.checked; render(); };
    inputs.push([lab.firstChild, tr]);
    box.appendChild(lab);
  }
  gl.firstChild.onchange = e => {
    showGroup[gid] = e.target.checked;
    render();
  };
  $('tagboxes').appendChild(box);
  $('legend').innerHTML += `<span style="flex-basis:100%"><b>${gname}</b></span>`;
  for (const tr of rows){
    const types = [...new Map(tr.items.map(q => [q[2], q[4]])).entries()];
    $('legend').innerHTML += `<span style="flex-basis:100%;padding-left:12px"><b>${tr.code}:</b> ` + types.map(([n, c]) => `<span style="margin-right:12px"><i style="background:${c}"></i>${n}</span>`).join('') + '</span>';
  }
}
const hasSide = TROWS.some(r => r.items.some(q => q[6]));
if (hasSide){
  $('sidebox').style.display = $('sidebox2').style.display = 'flex';
  $('sideH').onchange = e => { showSide.H = e.target.checked; render(); };
  $('sideA').onchange = e => { showSide.A = e.target.checked; render(); };
  $('legend').innerHTML += `<span style="flex-basis:100%"><b>Tag team:</b> <span style="margin-right:12px"><i style="background:var(--H)"></i>H = home (left stripe)</span><span><i style="background:var(--A)"></i>A = away</span></span>`;
}
fillTable('tg', D.gs, 'g'); fillTable('tp', D.ps, 'p');
buildMismatches(); render();
})();
</script>
</body>
</html>
"""


def clock(seconds, _=None):
    return "%d:%02d" % divmod(int(round(seconds)), 60)


def summarize(gt, pred, matched, g_sp, p_sp, threshold):
    print("\nFRAME LEVEL (%d frames = %.1f minutes)" % (len(gt), len(gt) / FPS / 60))
    print(pd.crosstab(pd.Series(gt, name="truth"), pd.Series(pred, name="prediction")).to_string())
    print("accuracy %.6f" % np.mean(gt == pred))
    for s in ("H", "A", "D"):
        tp = np.sum((gt == s) & (pred == s))
        rec = tp / max(np.sum(gt == s), 1); prec = tp / max(np.sum(pred == s), 1)
        print("  %s  recall %.4f  precision %.4f  (truth %d frames, predicted %d)" % (s, rec, prec, np.sum(gt == s), np.sum(pred == s)))
    tp = len(matched)
    f1 = 2 * tp / (len(g_sp) + len(p_sp)) if (g_sp or p_sp) else float("nan")
    print("\nSPELLS: truth %d, predicted %d, matched at IoU >= %.2f: %d  -> F1 %.4f  (compare spell_f1_at_headline)"
          % (len(g_sp), len(p_sp), threshold, tp, f1))
    if g_sp:
        print("mean length: truth %.1f s, predicted %.1f s"
              % (np.mean([(e - s) for s, e, _ in g_sp]) / FPS, np.mean([(e - s) for s, e, _ in p_sp]) / FPS if p_sp else np.nan))


def draw(gt, pred, g_sp, p_sp, matched, threshold, out, start, title, tags=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    kind = agreement(gt, pred)
    seconds = lambda frames: frames / FPS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6.6 + 0.45 * sum(tr["lanes"] for tr in tags) + 0.3 * len({tr["group"] for tr in tags})), gridspec_kw={"height_ratios": [3 + 0.45 * sum(tr["lanes"] for tr in tags) + 0.3 * len({tr["group"] for tr in tags}), 2.4]}, sharex=True)

    rows = [("Ground truth", gt, COLOR), ("Prediction", pred, COLOR)]
    for y, (name, lab, palette) in zip((2, 1), rows):
        for a, b, s in runs(lab):
            ax1.broken_barh([(seconds(a), seconds(b - a))], (y - 0.4, 0.8), facecolors=palette[s], edgecolor="none")
    for a, b, k in runs(kind.astype("<U12")):
        ax1.broken_barh([(seconds(a), seconds(b - a))], (-0.4, 0.8), facecolors=AGREE[k], edgecolor="none")
    ticks, names = [2, 1, 0], ["Ground truth", "Prediction", "Agreement"]
    low = -0.7
    titles, prev_group = dict(GROUPS), None
    for tr in tags:
        if tr["group"] != prev_group:                   # group header: a thin line and the group's name
            prev_group = tr["group"]
            ax1.plot([0, seconds(len(gt))], [low - 0.05] * 2, color="#BBBBBB", linewidth=0.7, clip_on=False)
            ax1.text(0.2, low - 0.3, titles[tr["group"]], fontsize=8.5, fontweight="bold", color="#555555", va="center")
            low -= 0.55
        top = low - 0.1
        for a, b, kind_name, team, colour, lane, side in tr["items"]:
            y0 = top - 0.7 * (lane + 1)
            ax1.broken_barh([(seconds(a), seconds(b - a))], (y0, 0.62), facecolors=colour, edgecolor="#333333", linewidth=0.8)
            if side:                                    # team stripe on the left edge, in the H / A colour
                ax1.broken_barh([(seconds(a), min(seconds(b - a), 1.2))], (y0, 0.62), facecolors=COLOR[side], edgecolor="#333333", linewidth=0.8)
            label = ("%s  %s" % (side, kind_name)) if side else kind_name
            if seconds(b - a) > 1.3 * len(label):
                ax1.text(seconds((a + b) / 2) + (0.5 if side else 0), y0 + 0.31, label, ha="center", va="center", fontsize=7.5, clip_on=True)
        ticks.append(top - 0.35 * tr["lanes"] - 0.05); names.append(tr["code"])
        low = top - 0.7 * tr["lanes"] - 0.2
    ax1.set_yticks(ticks); ax1.set_yticklabels(names)
    ax1.set_ylim(low - 0.1 if tags else -0.6, 2.6)
    share = {k: 100 * np.mean(kind == k) for k in AGREE}
    ax1.set_title("%s   |   frame agreement %.1f%%" % (title, share["match"]), loc="left", fontsize=12)
    handles = [Patch(color=COLOR[s], label={"H": "H (team A)", "A": "A (team B)", "D": "D dead ball", "X": "X unknown"}[s]) for s in STATES]
    handles += [Patch(color=AGREE[k], label="%s %.1f%%" % (k, share[k])) for k in AGREE]
    seen_types = dict.fromkeys((it[2], it[4]) for tr in tags for it in tr["items"])
    handles += [Patch(facecolor=c, edgecolor="#333333", label=n) for n, c in seen_types]
    if any(it[6] for tr in tags for it in tr["items"]):
        handles += [Patch(facecolor=COLOR[s], edgecolor="#333333", label="tag by %s (left stripe)" % n) for s, n in (("H", "home"), ("A", "away"))]
    ax1.legend(handles=handles, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

    for y, (name, sp, lab) in zip((1, 0), (("Ground truth", g_sp, gt), ("Prediction", p_sp, pred))):
        for a, b in [(a, b) for a, b, s in runs(lab) if s == "D"]:
            ax2.broken_barh([(seconds(a), seconds(b - a))], (y - 0.45, 0.9), facecolors="#EDEDED", edgecolor="none", zorder=0)
    g_hit = {i: m for i, m in matched.items()}
    p_hit = {m[0]: (i, m[1]) for i, m in matched.items()}
    for y, sp, hit in ((1, g_sp, g_hit), (0, p_sp, p_hit)):
        for k, (a, b, team) in enumerate(sp):
            ok = k in hit
            ax2.broken_barh([(seconds(a), seconds(b - a))], (y - 0.3, 0.6), facecolors=COLOR[team], alpha=0.9 if ok else 0.45,
                            edgecolor="black" if ok else "#B00020", linewidth=1.4 if ok else 1.1, linestyle="-" if ok else (0, (3, 2)), zorder=2)
    for i, (j, score) in matched.items():
        mid = seconds((max(g_sp[i][0], p_sp[j][0]) + min(g_sp[i][1], p_sp[j][1])) / 2)
        ax2.annotate("%.2f" % score, (mid, 0.5), ha="center", va="center", fontsize=7.5,
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=3)
    ax2.set_yticks([1, 0]); ax2.set_yticklabels(["Truth spells", "Predicted spells"]); ax2.set_ylim(-0.6, 1.6)
    ax2.set_title("Possession spells: solid outline = matched (same team, IoU >= %.2f), dashed = unmatched; grey = dead ball" % threshold,
                  loc="left", fontsize=10)
    ax2.xaxis.set_major_locator(MultipleLocator(30)); ax2.xaxis.set_minor_locator(MultipleLocator(10))
    ax2.xaxis.set_major_formatter(FuncFormatter(clock))
    ax2.set_xlabel("match time (mm:ss from the slice start, frame %d)" % start)
    ax2.set_xlim(0, len(gt) / FPS)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("\nwrote", out)


TAG_KINDS = (("corner", "corner"), ("free kick", "free"), ("free-kick", "free"), ("throw", "throw"), ("goal kick", "goalkick"),
             ("kick off", "kickoff"), ("kick-off", "kickoff"), ("kickoff", "kickoff"), ("penalty", "penalty"))
SET_PIECE_COLORS = {"corner": "#B9A7E8", "free": "#E3B98F", "throw": "#8ED1D3", "goalkick": "#B8C4CE", "kickoff": "#E8A09C",
                    "penalty": "#F2C6DE", "other": "#D5D5D5"}
OTHER_PALETTE = ["#9DB4D6", "#E5C07B", "#A8D5A2", "#D9A5C8", "#8FD0C9", "#E8B4A0", "#C3B6E0", "#BFC9A0", "#F0D58C", "#A9C7E8"]


def tag_kind(text):
    low = str(text).lower().replace("_", " ")
    return next((k for key, k in TAG_KINDS if key in low), "other")


# Smart Tag codes in three groups (edit the lists to move a code).  Keys are the code lower-cased, spaces -> "_".
GROUPS = [("attack", "Attacking actions"), ("defence", "Defensive actions"), ("neutral", "Neutral actions / information")]
GROUP_ORDER = {
    "attack": ["long_ball", "build_up", "progression", "ball_in_final_third", "ball_in_the_box", "creating_chances", "goals",
               "players_in_the_box", "attacking_transition"],
    "defence": ["defending_in_attacking_third", "defending_in_middle_third", "defending_in_defensive_third",
                "defensive_transition"],
    "neutral": ["set_pieces"],
}


def group_of(code):
    """'attack' / 'defence' / 'neutral' for a Smart Tag code; anything not listed in GROUP_ORDER is judged by its name
    (attack..., defen...) and otherwise counts as neutral information (set pieces, transitions without a side, ...)."""
    key = code.strip().lower().replace(" ", "_")
    for name, keys in GROUP_ORDER.items():
        if key in keys:
            return name
    if "defen" in key:
        return "defence"
    if "attack" in key:
        return "attack"
    return "neutral"


def group_rank(code):
    key = code.strip().lower().replace(" ", "_")
    name = group_of(code)
    order = GROUP_ORDER[name]
    return ([g for g, _ in GROUPS].index(name), order.index(key) if key in order else len(order), key)


HOME_ALIASES = ["home", "h", "bar", "barcelona", "fc barcelona"]
AWAY_ALIASES = ["away", "a", "ath", "athletic", "athletic club", "athletic bilbao"]


def side_of(team, home, away):
    """'H' / 'A' for a tag's Team value (case-insensitive, exact or full-name match against the aliases), '' if neither."""
    t = team.strip().lower()
    if t in home:
        return "H"
    if t in away:
        return "A"
    return ""


def aliases(extra, default):
    names = [n.strip().lower() for part in (extra or []) for n in part.split(",") if n.strip()]
    return names + default


def read_tags(path, start, end, home=None, away=None):
    """All Smart Tag rows inside the slice, one display row per `code`:
    [{"code", "lanes", "items": [(start, end_exclusive, Type, Team, colour, lane)]}], frames relative to `start`.
    Tags of one code that overlap in time are stacked in lanes.  Set pieces are coloured by kind (corner, free kick, ...),
    every other code by Type."""
    home, away = home or HOME_ALIASES, away or AWAY_ALIASES
    unknown = set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    col = {c.lower().strip(): c for c in df.columns}
    if any(n not in col for n in ("start_frame", "end_frame", "type")):
        print("%s: needs columns start_frame, end_frame and Type (found %s); no tag rows drawn" % (path, list(df.columns)[:10]))
        return []
    codes = df[col["code"]].str.strip() if "code" in col else pd.Series("Tags", index=df.index)
    palette, rows = {}, []
    for code in sorted(set(codes), key=group_rank):
        items = []
        for i in df.index[codes == code]:
            r = df.loc[i]
            try:
                s0, e0 = int(float(r[col["start_frame"]])), int(float(r[col["end_frame"]])) + 1
            except ValueError:
                continue
            if e0 <= start or s0 > end:
                continue
            kind_name = r[col["type"]].strip() or "(no type)"
            if "set piece" in code.lower():
                colour = SET_PIECE_COLORS[tag_kind(kind_name)]
            else:
                colour = palette.setdefault(kind_name, OTHER_PALETTE[len(palette) % len(OTHER_PALETTE)])
            team = r[col["team"]].strip() if "team" in col else ""
            side = side_of(team, home, away)
            if team and not side and "set piece" not in code.lower():
                unknown.add(team)
            items.append([max(s0, start) - start, min(e0, end + 1) - start, kind_name, team, colour, side])
        if not items:
            continue
        items.sort()
        lane_end = []
        for it in items:
            lane = next((k for k, e in enumerate(lane_end) if e <= it[0]), None)
            if lane is None:
                lane_end.append(it[1]); lane = len(lane_end) - 1
            else:
                lane_end[lane] = it[1]
            it.insert(5, lane)                         # -> [start, end, Type, Team, colour, lane, side]
        rows.append({"code": code, "group": group_of(code), "lanes": len(lane_end), "items": items})
    if unknown:
        print("Tag Team values that are neither home nor away (drawn without a side; use --home-team / --away-team): %s"
              % ", ".join(sorted(unknown)))
    return rows


def stats_rows(gt, pred, g_sp, p_sp, matched):
    rec = lambda s: np.sum((gt == s) & (pred == s)) / max(np.sum(gt == s), 1)
    prec = lambda s: np.sum((gt == s) & (pred == s)) / max(np.sum(pred == s), 1)
    tp = len(matched)
    f1 = 2 * tp / (len(g_sp) + len(p_sp)) if (g_sp or p_sp) else float("nan")
    return [["frame accuracy", "%.1f%%" % (100 * np.mean(gt == pred))],
            ["D recall", "%.3f" % rec("D")], ["D precision", "%.3f" % prec("D")],
            ["truth spells", str(len(g_sp))], ["predicted spells", str(len(p_sp))],
            ["matched spells", str(tp)], ["spell F1 at the IoU", "%.3f" % f1]]


def write_html(gt, pred, g_sp, p_sp, matched, threshold, start, title, out, tags=()):
    kind = agreement(gt, pred)
    p_hit = {m[0]: (i, m[1]) for i, m in matched.items()}
    data = {
        "n": int(len(gt)), "fps": FPS, "start": int(start), "title": title, "iou": threshold,
        "gt": [[a, b, str(l)] for a, b, l in runs(gt)],
        "pred": [[a, b, str(l)] for a, b, l in runs(pred)],
        "kind": [[a, b, str(l)] for a, b, l in runs(kind.astype("<U12"))],
        "gs": [[a, b, t, matched[i][0] if i in matched else -1, round(matched[i][1], 4) if i in matched else 0.0]
               for i, (a, b, t) in enumerate(g_sp)],
        "ps": [[a, b, t, p_hit[j][0] if j in p_hit else -1, round(p_hit[j][1], 4) if j in p_hit else 0.0]
               for j, (a, b, t) in enumerate(p_sp)],
        "share": {key: round(100 * float(np.mean(kind == name)), 1)
                  for key, name in (("match", "match"), ("wrong", "wrong team"), ("dead", "dead/alive"), ("abst", "abstain (X)"))},
        "stats": stats_rows(gt, pred, g_sp, p_sp, matched),
        "tagrows": list(tags), "groups": [list(g) for g in GROUPS],
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(HTML_TEMPLATE.replace("/*__DATA__*/null", payload))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--gt")
    ap.add_argument("--out", default="overlap.png"); ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--html", help="also write an interactive, self-contained HTML page")
    ap.add_argument("--tags", help="Smart Tag csv (code, Type, Team, start_frame, end_frame): adds one row per code; "
                                   "default: data/tactical/BAR-ATH_smart_tagging_ds.csv when it exists")
    ap.add_argument("--home-team", action="append", help="name(s) the tag file uses for the home team (comma separated; "
                                                        "default home, h, BAR, Barcelona)")
    ap.add_argument("--away-team", action="append", help="name(s) the tag file uses for the away team (default away, a, ATH, Athletic)")
    ap.add_argument("--start", type=int, default=11185); ap.add_argument("--end", type=int, default=18684)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--frame-col"); ap.add_argument("--pred-col"); ap.add_argument("--gt-col")
    ap.add_argument("--title", default="Public slice: prediction vs ground truth")
    a = ap.parse_args()
    if not a.gt:
        found = [p for pat in ("truth", "ground", "gt", "label") for p in glob.glob("../evaluation/**/*%s*.csv" % pat, recursive=True)]
        found = sorted(set(found))
        if len(found) != 1:
            sys.exit("give --gt; candidates under ../evaluation: %s" % (found or "none"))
        a.gt = found[0]; print("ground truth:", a.gt)
    gt = read_labels(a.gt, a.start, a.end, a.frame_col, a.gt_col)
    pred = read_labels(a.pred, a.start, a.end, a.frame_col, a.pred_col)
    tags = []
    tag_path = a.tags or next((p for p in ("data/tactical/BAR-ATH_smart_tagging_ds.csv", "../data/tactical/BAR-ATH_smart_tagging_ds.csv")
                               if os.path.exists(p)), None)
    if tag_path:
        tags = read_tags(tag_path, a.start, a.end, aliases(a.home_team, HOME_ALIASES), aliases(a.away_team, AWAY_ALIASES))
        print("Smart tags from %s inside the slice: %s" % (tag_path, ", ".join("%s (%d)" % (tr["code"], len(tr["items"])) for tr in tags) or "none"))
    g_sp, p_sp = spells(gt), spells(pred)
    matched = match(g_sp, p_sp, a.iou)
    summarize(gt, pred, matched, g_sp, p_sp, a.iou)
    if not a.no_png:
        draw(gt, pred, g_sp, p_sp, matched, a.iou, a.out, a.start, a.title, tags)
    if a.html:
        write_html(gt, pred, g_sp, p_sp, matched, a.iou, a.start, a.title, a.html, tags)


if __name__ == "__main__":
    main()
# Merah Putih: More than a Hack 2027 (BAR-ATH possession)

Rule-based, label-free per-frame possession labels (H / A / D / X) for the BAR-ATH match.
No model is trained and no ground truth is read to produce the labels.

## Reproduce the submission

The match data is **not** in this repository. Place it where the challenge repo expects it, then run:

```bash
cd smart-tags                      # this folder, inside the challenge repo
pip install -r requirements-smart-tags.txt
bash run_all.sh                    # writes "Merah Putih.csv" and validates it
```

`run_all.sh` does three things:

1. `prepare_data.py` builds `../data/tactical/BAR-ATH_tactical_ds.csv` (tracking) and `BAR-ATH_smart_tagging_ds.csv` (Smart Tagging) from the raw files, mirroring the data-prep notebook (kloppy `metrica.load_tracking_epts`, `sportscode.load`).
2. `combined_submission.py` produces the labels with fixed parameters (all listed in `run_all.sh`).
3. `../evaluation/validate_submission.py` checks the file format.

Set another name with `TEAM="Other Name" bash run_all.sh`. Pinned versions: kloppy 3.18.0, numpy 2.5.4, pandas 3.1.0.

## Layout

| File | Role |
|---|---|
| `run_all.sh` | one-command regeneration |
| `prepare_data.py` | raw files to the two input CSVs |
| `tracking_io.py` | tracking reader (`video_frame` or `frame_id`) |
| `smart_tags.py` | per-frame tag features, home/away attacking/defending flags |
| `tag_only_submission.py` | tag-only labelling rule and writer |
| `combined_submission.py` | final pipeline: geometry + tags + dead-ball + smoothing |
| `dead_ball_diagnostic.py`, `restart_offset.py`, `speed_threshold_calibration.py` | how the dead-ball thresholds were set (label-free) |
| `pseudo_label_agreement.py` | diagnostic: tags vs. tracking-based pseudo-labels |
| `smart_tags_*.py` | exploratory diagnostics (not used by the final labels) |
| `test_*.py` | unit tests on synthetic data |
| `TECHNICAL.md` | method, parameter provenance, results, limitations |

## Tests

```bash
python -m pytest test_*.py
```

They use synthetic data only.

## Data

Nothing derived from the match data is committed (see `.gitignore`).

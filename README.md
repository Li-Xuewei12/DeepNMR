# DeepNMR

Stage δ predicts calculated chemical shifts; Stage J predicts total coupling and its four components; the overlay predicts experimental chemical shifts. Run the commands below from this directory.

## Environment and downloads

Use Python 3.10.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
```

**Weights and data (Zenodo):** <!-- Add the Zenodo URL here after uploading. -->

After downloading, copy the archive contents as follows:

| Archive directory | Destination in this repository |
| --- | --- |
| `weights/` | `weights/` |
| `data/` | `data/` |
| `Our_data/` | `data/evaluation/` |

The archive contains processed training/validation caches, evaluated cohorts, and seven checkpoints (five prediction models and two topology initializers). Only load trusted PyTorch/pickle files.

## Direct use and evaluation

```sh
python scripts/predict.py --stage delta --nucleus 1H --input data/evaluation/Stage_delta/graphs_DFT8K.pt --output runs/delta_H.parquet
python scripts/predict.py --stage overlay --nucleus 1H --input data/evaluation/Stage2B_assigned/graphs_1H.pt --output runs/overlay_H.parquet
python scripts/predict.py --stage j --input data/evaluation/Stage_J/graphs_internal.pkl --output runs/J.parquet
python scripts/evaluate_assigned.py runs/delta_H.parquet
python scripts/evaluate_assigned.py runs/overlay_H.parquet --stage overlay
python scripts/evaluate_assigned.py runs/J.parquet --stage j
```

Use `--nucleus 13C` for carbon and the corresponding carbon input/output files. Prediction runs on CPU by default; add `--device cuda:0` for a GPU or `--limit 20` for a quick check. Inputs must follow the supplied graph schema and atom order. Outputs include atom/pair identifiers, predictions, and references when present; existing output files are never overwritten.

Dataset3 uses placeholder labels in its inference graphs. Supply the separate references when evaluating it:

```sh
python scripts/predict.py --stage delta --nucleus 1H --input data/evaluation/Dataset3/prepared/graphs.pt --reference data/evaluation/Dataset3/reference_labels.csv.gz --output runs/dataset3_H.parquet
```

The external CHAMPS holdout also has placeholder graph targets. Use `--reference data/evaluation/Stage_J/DeepNMR__External_CHAMPS_holdout__pair_errors.parquet` when predicting `graphs_holdout.pkl`; references are joined by molecule, atom pair, and coupling type.

Recheck predictions, peak metrics, training caches, or basic regression tests:

```sh
python scripts/audit_release.py --limit 20
python scripts/audit_peaks.py
python scripts/audit_training_data.py
python -m unittest discover -s tests -v
```

Use `--limit 0` for full prediction replay. `analysis/analyze_frozen.py` recomputes frozen-cohort statistics without generating figures.

## Retraining

```sh
python -m stage_delta.train --config configs/stage_delta_1H.yaml
python -m stage_delta.train --config configs/stage_delta_13C.yaml
python scripts/train_j.py --config configs/stage_j.yaml
python -m deepnmr.overlay --nucleus 1H --data-root data/assigned_overlay --output runs/overlay_1H
python -m deepnmr.overlay --nucleus 13C --data-root data/assigned_overlay --output runs/overlay_13C
```

Training defaults to CUDA. For CPU, set `runtime.device: cpu` in the YAML configurations, or pass `--device cpu` to overlay training. Training uses the supplied train/validation splits; test data do not select checkpoints. Choose a new output directory for each run. These commands were checked with small one-epoch runs; full training has not been repeated for this code cleanup.

To use a retrained checkpoint, copy its `best_validation.pt` to a separate weights directory under the corresponding name listed in `weights/README.md`, then pass `--weights-dir` to prediction. For an overlay, include the matching Stage δ checkpoint used for its training.

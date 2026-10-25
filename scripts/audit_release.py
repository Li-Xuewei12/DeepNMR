"""Replay checkpoints against frozen predictions; verify hashes and invariance."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from deepnmr.checkpoints import ROOT, load_stage_delta, load_overlay, load_j
from stage_delta.data import collate_graphs
from stage_j.constants import COMPONENT_NAMES
from stage_j.data import collate_couplings
from predict import predict_records, attach_references
from train_j import invariance_audit


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["files"]
    for item in entries:
        path = root / item["file"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise AssertionError(f"Manifest mismatch: {path}")
    return len(entries)


def compare_predictions(actual, reference, keys, columns, tolerance, full):
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():
        raise AssertionError(f"Duplicate observation keys: {keys}")
    selected = reference.merge(actual[keys], on=keys, how="inner", validate="one_to_one")
    if len(selected) != len(actual) or (full and len(actual) != len(reference)):
        raise AssertionError("Replay observation keys do not cover the expected frozen cohort")
    a = actual.set_index(keys).sort_index()
    b = selected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    differences = {}
    for column in columns:
        x, y = a[column].to_numpy(float), b[column].to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError(f"Non-finite replay values: {column}")
        difference = float(np.max(np.abs(x - y)))
        if difference > tolerance:
            raise AssertionError(f"{column}: {difference} exceeds {tolerance}")
        differences[column] = difference
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/evaluation")
    parser.add_argument("--archive-root", type=Path, help="Also verify the Zenodo manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/release_audit.json")
    parser.add_argument("--limit", type=int, default=20, help="Graphs per cohort; 0 means all")
    parser.add_argument("--j-limit", type=int, help="Override graph limit for the larger J cohorts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=["all", "delta", "overlay", "j"], default="all")
    args = parser.parse_args()
    if args.limit < 0 or (args.j_limit is not None and args.j_limit < 0):
        parser.error("Limits must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    report = {"python": sys.version, "torch": torch.__version__, "device": args.device, "checks": []}
    if args.archive_root:
        report["archive_files_verified"] = verify_manifest(args.archive_root)
    if (ROOT / "manifest.json").exists():
        report["code_files_verified"] = verify_manifest(ROOT)
    cases = []
    for nucleus in ["1H", "13C"]:
        cases.extend([
            ("delta", nucleus, "DFT8K", "Stage_delta/graphs_DFT8K.pt",
             f"Stage_delta/DeepNMR-Stage1__Internal_DFT8K_test__{nucleus}__atom_errors.parquet"),
            ("delta", nucleus, "Dataset3", "Dataset3/prepared/graphs.pt",
             f"Stage_delta/DeepNMR__External_Dataset3_test__{nucleus}__atom_errors.parquet"),
            ("overlay", nucleus, "assigned", f"Stage2B_assigned/graphs_{nucleus}.pt",
             f"Stage2B_assigned/DeepNMR-overlay__Internal_NMRShiftDB2_test__{nucleus}__atom_errors.parquet"),
        ])
    for cohort, label in [("internal", "Internal_CHAMPS_test"), ("holdout", "External_CHAMPS_holdout")]:
        cases.append(("j", None, cohort, f"Stage_J/graphs_{cohort}.pkl", f"Stage_J/DeepNMR__{label}__pair_errors.parquet"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for stage, nucleus, cohort, graphs, frozen in cases:
        if args.stage != "all" and stage != args.stage:
            continue
        path = args.data_root / graphs
        if stage == "j":
            with path.open("rb") as handle:
                records = pickle.load(handle)
            model = load_j(args.weights_dir)
            keys = ["molecule_id", "atom_i", "atom_j", "coupling_type"]
            columns = [f"prediction_{name}_hz" for name in COMPONENT_NAMES]
            columns += [f"target_{name}_hz" for name in COMPONENT_NAMES]
        else:
            records = torch.load(path, map_location="cpu", weights_only=False)
            code = 0 if nucleus == "1H" else 1
            records = [r for r in records if r["target_nucleus"].eq(code).any()]
            model = (load_overlay(nucleus, args.weights_dir) if stage == "overlay"
                     else load_stage_delta(args.weights_dir / f"stage_delta_{nucleus}.pt", nucleus))
            keys = ["molecule_uid", "atom_index_conformer"]
            columns = ["prediction_ppm", "target_ppm"]
        total_graphs = len(records)
        limit = args.j_limit if stage == "j" and args.j_limit is not None else args.limit
        if limit:
            records = records[:limit]
        print(f"Replaying {stage} {nucleus or ''} {cohort}: {len(records)}/{total_graphs} graphs", flush=True)
        actual = predict_records(model, records, stage, nucleus, device=args.device)
        if cohort == "Dataset3":
            actual = attach_references(actual, args.data_root / "Dataset3/reference_labels.csv.gz", stage)
        if stage == "j" and cohort == "holdout":
            actual = attach_references(actual, args.data_root / frozen, stage)
        reference = pd.read_parquet(args.data_root / frozen).rename(columns={"predicted_j_hz": "prediction_total_hz"})
        difference = compare_predictions(actual, reference, keys, columns, 1e-3, len(records) == total_graphs)
        # Verify spatial invariance using freshly recomputed radius edges.
        model = model.cpu().eval()
        if stage == "j":
            invariance = invariance_audit(model, collate_couplings(records[:2], cutoff=model.cutoff, require_targets=False), torch.device("cpu"))
        else:
            q, _ = torch.linalg.qr(torch.randn(3, 3))
            changed = [dict(r, coordinates=r["coordinates"] @ q + torch.tensor([2., -3., 1.])) for r in records[:2]]
            with torch.inference_mode():
                cutoff = model.shift.cutoff if stage == "overlay" else model.cutoff
                a = model(collate_graphs(records[:2], cutoff=cutoff, require_targets=False))
                b = model(collate_graphs(changed, cutoff=cutoff, require_targets=False))
                if stage == "delta":
                    a, b = a["prediction_ppm"], b["prediction_ppm"]
            invariance = {"rotation_translation_max_abs_difference": float((a - b).abs().max())}
        if max(invariance.values()) > 1e-3:
            raise AssertionError(invariance)
        row = dict(stage=stage, nucleus=nucleus, cohort=cohort, graphs=len(records), total_graphs=total_graphs,
                   observations=len(actual), max_absolute_difference=difference, invariance=invariance, result="PASS")
        report["checks"].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"PASS: {len(actual)} observations", flush=True)
    print(f"Audit complete: {args.output}")


if __name__ == "__main__":
    main()

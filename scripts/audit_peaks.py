"""Recompute peak-set errors from archived peak predictions and source references."""
import argparse
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from deepnmr.evaluation import peak_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("runs/peak_audit"))
    parser.add_argument("--limit", type=int, default=0, help="Experiments per cohort/nucleus; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    for cohort in ["NPS", "NMRexp"]:
        for nucleus in ["1H", "13C"]:
            root = args.data_root / cohort
            refs = {}
            with gzip.open(root / f"processed_records_{nucleus}.jsonl.gz", "rt") as handle:
                for line in handle:
                    record = json.loads(line)
                    uid = record["deepnmr_experiment_uid"]
                    if uid in refs:
                        raise ValueError(f"Duplicate reference experiment {uid}")
                    peaks = record["h_nmr_peaks" if nucleus == "1H" else "c_nmr_peaks"]
                    refs[uid] = [float(p["centroid" if nucleus == "1H" else "delta (ppm)"]) for p in peaks]
                    if args.limit and len(refs) >= args.limit:
                        break
            for model in ["DeepNMR", "CSP5", "NMRNet"]:
                predictions = pd.read_parquet(root / f"{model}_{nucleus}_peaks.parquet").set_index("experiment_uid")
                archived = pd.read_parquet(root / f"{model}_{nucleus}_per_experiment.parquet").set_index("experiment_uid")
                if predictions.index.has_duplicates or archived.index.has_duplicates:
                    raise ValueError("Duplicate prediction/evaluation experiment IDs")
                if not args.limit and (set(refs) != set(predictions.index) or set(refs) != set(archived.index)):
                    raise ValueError(f"Experiment coverage differs: {cohort} {nucleus} {model}")
                rows, errors = [], []
                for uid, y in refs.items():
                    peaks = predictions.loc[uid, "peaks_json"]
                    peaks = json.loads(peaks) if isinstance(peaks, str) else peaks
                    p = [float(x.get("center_ppm", x.get("delta_center_ppm"))) for x in peaks]
                    result = peak_errors(y, p, 1 if nucleus == "1H" else 10)
                    old = archived.loc[uid]
                    errors.append(abs(result["set_error"] - old.capped_set_mae_ppm))
                    if result["matched_pairs"]:
                        errors.append(abs(result["absolute_error_sum"] / result["matched_pairs"] - old.matched_shift_mae_ppm))
                    rows.append(dict(experiment_uid=uid, **result))
                maximum = float(max(errors))
                if not np.isfinite(errors).all() or maximum > 1e-8:
                    raise AssertionError((cohort, nucleus, model, maximum))
                pd.DataFrame(rows).to_parquet(args.output / f"{cohort}_{nucleus}_{model}.parquet", index=False)
                checks.append(dict(cohort=cohort, nucleus=nucleus, model=model,
                                   experiments=len(refs), max_discrepancy=maximum, result="PASS"))
                print(cohort, nucleus, model, len(refs), "PASS", flush=True)
                (args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()

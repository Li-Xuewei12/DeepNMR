"""Predict processed graphs while preserving atom/pair identities and references."""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import torch
from stage_delta.data import collate_graphs, move_batch
from stage_j.constants import COMPONENT_NAMES, TYPE_NAMES
from stage_j.data import collate_couplings
from deepnmr.checkpoints import ROOT, load_stage_delta, load_overlay, load_j


def attach_references(frame, path, stage):
    reference = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    keys = (["molecule_id", "atom_i", "atom_j", "coupling_type"] if stage == "j"
            else ["molecule_uid", "atom_index_conformer", "nucleus"])
    targets = [f"target_{n}_hz" for n in COMPONENT_NAMES] if stage == "j" else ["target_ppm"]
    columns = keys + targets
    joined = frame.drop(columns=targets).merge(
        reference[columns], on=keys, how="left", validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all() or not torch.isfinite(torch.tensor(joined[targets].to_numpy())).all():
        raise ValueError("Missing or invalid references for predicted observation keys")
    return joined.drop(columns="_merge")


def predict_records(model, records, stage, nucleus=None, batch_size=16, device="cpu"):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = model.to(device).eval()
    cutoff = model.shift.cutoff if stage == "overlay" else model.cutoff
    rows = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            items = records[start:start + batch_size]
            if stage == "j":
                batch = collate_couplings(items, cutoff=cutoff, require_targets=False)
            else:
                # Inference selects atoms by nucleus, even when no reference shift exists.
                batch = collate_graphs(items, cutoff=cutoff, require_targets=False)
            batch = move_batch(batch, torch.device(device))
            output = model(batch)
            if stage == "j":
                predictions = output["prediction_hz"].cpu()
                if not torch.isfinite(predictions).all():
                    raise FloatingPointError("Non-finite coupling predictions")
                for i, values in enumerate(predictions.tolist()):
                    row = dict(molecule_id=batch["pair_molecule_ids"][i],
                               atom_i=batch["pair_atom_i"][i], atom_j=batch["pair_atom_j"][i],
                               coupling_type=TYPE_NAMES[int(batch["coupling_type"][i])])
                    for k, name in enumerate(COMPONENT_NAMES):
                        row[f"prediction_{name}_hz"] = values[k]
                        row[f"target_{name}_hz"] = float(batch["targets_hz"][i, k])
                    rows.append(row)
            else:
                code = 0 if nucleus == "1H" else 1
                mask = batch["target_nucleus"].eq(code)
                predictions = output if stage == "overlay" else output["prediction_ppm"]
                if not torch.isfinite(predictions).all():
                    raise FloatingPointError("Non-finite chemical-shift predictions")
                for gid, atom, target, pred in zip(
                    batch["target_graph_index"][mask].tolist(),
                    batch["target_atom_index_conformer"][mask].tolist(),
                    batch["target_shift"][mask].tolist(), predictions.tolist(),
                ):
                    rows.append(dict(molecule_uid=batch["molecule_uids"][gid],
                                     conformer_uid=batch["conformer_uids"][gid],
                                     atom_index_conformer=atom, nucleus=nucleus,
                                     target_ppm=target, prediction_ppm=pred))
    if not rows:
        raise ValueError("No requested atoms or coupling pairs were found")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["delta", "overlay", "j"], required=True)
    parser.add_argument("--nucleus", choices=["1H", "13C"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--reference", type=Path, help="Optional reference CSV/parquet, joined by atom/pair identity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("--limit must be nonnegative and --batch-size positive")
    if args.stage != "j" and args.nucleus is None:
        parser.error("--nucleus is required for delta or overlay")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    if args.stage == "j":
        with args.input.open("rb") as handle:
            records = pickle.load(handle)
        model = load_j(args.weights_dir)
    else:
        records = torch.load(args.input, map_location="cpu", weights_only=False)
        code = 0 if args.nucleus == "1H" else 1
        records = [r for r in records if r["target_nucleus"].eq(code).any()]
        model = (load_overlay(args.nucleus, args.weights_dir) if args.stage == "overlay"
                 else load_stage_delta(args.weights_dir / f"stage_delta_{args.nucleus}.pt", args.nucleus))
    if args.limit:
        records = records[:args.limit]
    frame = predict_records(model, records, args.stage, args.nucleus, args.batch_size, args.device)
    if args.reference is not None:
        frame = attach_references(frame, args.reference, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(len(frame), "predictions ->", args.output)


if __name__ == "__main__":
    main()

"""Audit released training caches: identities, dimensions, labels, and statistics."""
import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(records, coupling=False, nucleus=None):
    ids, values, types = [], [], []
    atoms = observations = 0
    for record in records:
        uid = record["molecule_id" if coupling else "molecule_uid"]
        ids.append(uid)
        z = np.asarray(record["atomic_number"])
        xyz = np.asarray(record["coordinates"])
        n = len(z)
        if xyz.shape != (n, 3) or not np.isfinite(xyz).all() or n == 0:
            raise ValueError(f"Invalid coordinates: {uid}")
        for name in ["formal_charge", "chirality", "aromatic"]:
            if np.asarray(record[name]).shape != (n,):
                raise ValueError(f"Invalid atom features: {uid}")
        bonds = np.asarray(record["bond_index"]).reshape(-1, 2)
        if len(bonds) and (bonds.min() < 0 or bonds.max() >= n):
            raise ValueError(f"Invalid bond index: {uid}")
        if len(record["bond_type"]) != len(bonds):
            raise ValueError(f"Invalid bond types: {uid}")
        if coupling:
            y = np.asarray(record["targets_hz"], dtype=float)
            pairs = np.asarray(record["pair_index"])
            t = np.asarray(record["coupling_type"])
            if y.shape != (len(pairs), 5) or not np.isfinite(y).all():
                raise ValueError(f"Invalid coupling labels: {uid}")
            if pairs.shape != (len(y), 2) or pairs.min() < 0 or pairs.max() >= n:
                raise ValueError(f"Invalid pair index: {uid}")
            if t.shape != (len(y),) or not np.isin(t, np.arange(8)).all():
                raise ValueError(f"Invalid coupling classes: {uid}")
            paths = np.asarray(record["path_nodes"])
            if paths.shape[0] != len(y) or np.any(paths < -1) or np.any(paths >= n):
                raise ValueError(f"Invalid path index: {uid}")
            values.append(y); types.append(t)
        else:
            y = np.asarray(record["target_shift"], dtype=float)
            codes = np.asarray(record["target_nucleus"])
            active = codes >= 0
            if y.shape != (n,) or codes.shape != (n,) or not np.isfinite(y[active]).all():
                raise ValueError(f"Invalid shift labels: {uid}")
            if not np.isin(codes, [-1, 0, 1]).all():
                raise ValueError(f"Invalid nucleus codes: {uid}")
            if not np.all(z[codes == 0] == 1) or not np.all(z[codes == 1] == 6):
                raise ValueError(f"Label/element mismatch: {uid}")
            if nucleus is not None and np.any(codes[active] != nucleus):
                raise ValueError(f"Unexpected nucleus: {uid}")
            values.append(y[active]);types.append(codes[active])
        atoms += n;observations += len(values[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate molecule identifiers within a split")
    y, t = np.concatenate(values), np.concatenate(types)
    statistics = {str(k): dict(mean=np.mean(y[t == k], axis=0).tolist(),
                               std=np.std(y[t == k], axis=0).tolist(), count=int((t == k).sum()))
                  for k in sorted(set(t.tolist()))}
    return set(ids), dict(molecules=len(ids), atoms=atoms, observations=observations, statistics=statistics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/training_data_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {}
    cases = [("dft8k", False, None), ("champs", True, None),
             ("assigned_overlay/1H", False, 0), ("assigned_overlay/13C", False, 1)]
    for folder, coupling, nucleus in cases:
        splits = ["train", "validation"] if nucleus is not None else ["train", "validation", "test"]
        identities, rows = {}, {}
        for split in splits:
            path = args.data_root / folder / (split + (".pkl" if coupling else ".pt"))
            if coupling:
                with path.open("rb") as handle:
                    records = pickle.load(handle)
            else:
                records = torch.load(path, map_location="cpu", weights_only=False)
            identities[split], rows[split] = inspect(records, coupling, nucleus)
            rows[split]["sha256"] = sha256(path)
            del records
        overlaps = {f"{a}/{b}": len(identities[a] & identities[b])
                    for a, b in itertools.combinations(splits, 2)}
        if any(overlaps.values()):
            raise AssertionError(f"Molecule-ID overlap in {folder}: {overlaps}")
        if coupling:
            stored = json.loads((args.data_root / folder / "target_statistics.json").read_text())
            for k in range(8):
                for field in ["mean", "std"]:
                    np.testing.assert_allclose(rows["train"]["statistics"][str(k)][field], stored[field][k], atol=1e-5, rtol=1e-5)
        report[folder] = dict(splits=rows, molecule_id_overlap=overlaps, result="PASS")
        print(folder, "PASS", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

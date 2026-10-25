from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


class CouplingRecordDataset(Dataset):
    def __init__(self, path: str | Path, limit: int = 0):
        with Path(path).open("rb") as handle:
            self.records = pickle.load(handle)
        if limit:
            self.records = self.records[: int(limit)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


def _radius_edges(pos: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, ...]:
    n = len(pos)
    distances = torch.cdist(pos, pos)
    not_self = ~torch.eye(n, dtype=torch.bool)
    selected = (distances < float(cutoff)) & (distances > 1.0e-8) & not_self
    edges = selected.nonzero(as_tuple=False)
    source, target = edges[:, 0], edges[:, 1]
    displacement = pos[target] - pos[source]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    unit = displacement / distance.clamp_min(1.0e-8).unsqueeze(-1)
    return source, target, distance, unit


def collate_couplings(
    records: Iterable[dict], cutoff: float = 5.0, delta_mode: str = "none", require_targets: bool = True
) -> dict:
    if delta_mode not in {"none", "zero", "predicted", "shuffled", "oracle"}:
        raise ValueError(f"unsupported delta_mode={delta_mode!r}")
    records = list(records)
    if not records or cutoff <= 0:
        raise ValueError("A nonempty coupling batch and positive cutoff are required")
    atom_offset = 0
    atomic_number, formal_charge, chirality, aromatic, coordinates = [], [], [], [], []
    graph_index = []
    bond_edges, bond_types = [], []
    radius_source, radius_target, radius_distance, radius_unit = [], [], [], []
    pair_indices, pair_types, targets = [], [], []
    path_nodes, path_mask, path_geometry = [], [], []
    pair_atom_i, pair_atom_j, pair_molecule_ids = [], [], []
    molecule_ids, source_molecule_ids = [], []
    delta_node_features = []

    for graph_id, record in enumerate(records):
        z = torch.as_tensor(record["atomic_number"], dtype=torch.long)
        pos = torch.as_tensor(record["coordinates"], dtype=torch.float32)
        n = len(z)
        atomic_number.append(z)
        formal_charge.append(torch.as_tensor(record["formal_charge"], dtype=torch.long))
        chirality.append(torch.as_tensor(record["chirality"], dtype=torch.long))
        aromatic.append(torch.as_tensor(record["aromatic"], dtype=torch.long))
        coordinates.append(pos)
        graph_index.append(torch.full((n,), graph_id, dtype=torch.long))
        if delta_mode != "none":
            if delta_mode == "zero":
                delta_value = torch.zeros(n, dtype=torch.float32)
                delta_mask = torch.zeros(n, dtype=torch.float32)
            else:
                value_key = {
                    "predicted": "predicted_delta_standardized",
                    "shuffled": "shuffled_delta_standardized",
                    "oracle": "oracle_delta_standardized",
                }[delta_mode]
                mask_key = (
                    "oracle_delta_mask" if delta_mode == "oracle"
                    else "predicted_delta_mask"
                )
                if value_key not in record or mask_key not in record:
                    raise KeyError(
                        f"record {record.get('molecule_id')} lacks cached {value_key} "
                        f"or {mask_key}"
                    )
                delta_value = torch.as_tensor(record[value_key], dtype=torch.float32)
                delta_mask = torch.as_tensor(
                    record[mask_key], dtype=torch.float32
                )
                if delta_value.shape != (n,) or delta_mask.shape != (n,):
                    raise ValueError("cached delta feature/atom length mismatch")
            delta_node_features.append(torch.stack([delta_value, delta_mask], dim=-1))

        bonds = torch.as_tensor(record["bond_index"], dtype=torch.long).reshape(-1, 2)
        types = torch.as_tensor(record["bond_type"], dtype=torch.long)
        if len(bonds):
            bond_edges.extend([(bonds + atom_offset).t(), (bonds[:, [1, 0]] + atom_offset).t()])
            bond_types.extend([types, types])

        src, dst, dist, unit = _radius_edges(pos, cutoff)
        radius_source.append(src + atom_offset)
        radius_target.append(dst + atom_offset)
        radius_distance.append(dist)
        radius_unit.append(unit)

        pairs = torch.as_tensor(record["pair_index"], dtype=torch.long)
        pair_indices.append(pairs + atom_offset)
        pair_types.append(torch.as_tensor(record["coupling_type"], dtype=torch.long))
        values = torch.as_tensor(record.get("targets_hz", torch.full((len(pairs), 5), float("nan"))), dtype=torch.float32)
        if record.get("dataset") == "CHAMPS_original_hidden" or record.get("target_kind") == "zero_inference_placeholder":
            values = torch.full((len(pairs), 5), float("nan"))
        if require_targets and not torch.isfinite(values).all():
            raise ValueError("Training requires real finite coupling targets, not inference placeholders")
        targets.append(values)

        local_path = torch.as_tensor(record["path_nodes"], dtype=torch.long)
        valid_path = local_path.ge(0)
        path_nodes.append(torch.where(valid_path, local_path + atom_offset, local_path))
        path_mask.append(valid_path)
        path_geometry.append(torch.as_tensor(record["path_geometry"], dtype=torch.float32))

        pair_atom_i.extend(pairs[:, 0].tolist())
        pair_atom_j.extend(pairs[:, 1].tolist())
        pair_molecule_ids.extend([record["molecule_id"]] * len(pairs))
        molecule_ids.append(record["molecule_id"])
        source_molecule_ids.append(record["source_molecule_id"])
        atom_offset += n

    empty_edges = torch.empty((2, 0), dtype=torch.long)
    batch = {
        "atomic_number": torch.cat(atomic_number),
        "formal_charge": torch.cat(formal_charge),
        "chirality": torch.cat(chirality),
        "aromatic": torch.cat(aromatic),
        "coordinates": torch.cat(coordinates),
        "graph_index": torch.cat(graph_index),
        "edge_index": torch.cat(bond_edges, dim=1) if bond_edges else empty_edges,
        "edge_type": torch.cat(bond_types) if bond_types else torch.empty(0, dtype=torch.long),
        "radius_edge_index": torch.stack(
            [torch.cat(radius_source), torch.cat(radius_target)], dim=0
        ),
        "radius_distance": torch.cat(radius_distance),
        "radius_unit_vector": torch.cat(radius_unit),
        "pair_index": torch.cat(pair_indices),
        "coupling_type": torch.cat(pair_types),
        "targets_hz": torch.cat(targets),
        "path_nodes": torch.cat(path_nodes),
        "path_mask": torch.cat(path_mask),
        "path_geometry": torch.cat(path_geometry),
        "molecule_ids": molecule_ids,
        "source_molecule_ids": source_molecule_ids,
        "pair_molecule_ids": pair_molecule_ids,
        "pair_atom_i": pair_atom_i,
        "pair_atom_j": pair_atom_j,
    }
    if delta_mode != "none":
        batch["delta_node_feature"] = torch.cat(delta_node_features, dim=0)
    return batch


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

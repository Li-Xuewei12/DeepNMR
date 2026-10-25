from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, path: Path, limit: int = 0):
        self.graphs = torch.load(path, map_location="cpu", weights_only=False)
        if limit:
            self.graphs = self.graphs[:int(limit)]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def collate_graphs(items: Iterable[dict], cutoff: float = 5.0, require_targets: bool = True) -> dict:
    items = list(items)
    if not items or cutoff <= 0:
        raise ValueError("A nonempty graph batch and positive cutoff are required")
    z, charge, chirality, aromatic, coordinates = [], [], [], [], []
    covalent_edges, covalent_types, radius_edges, radius_vectors, radius_distances = [], [], [], [], []
    graph_index, target_indices, target_local_indices, targets, target_nuclei, target_graph = [], [], [], [], [], []
    molecule_uids, conformer_uids, sample_weights, heavy_counts = [], [], [], []
    offset = 0
    for gid, item in enumerate(items):
        n = len(item["atomic_number"])
        z.append(item["atomic_number"]); charge.append(item["formal_charge"])
        chirality.append(item["chirality"]); aromatic.append(item["aromatic"])
        coordinates.append(item["coordinates"])
        bonds = item["bond_index"]
        if len(bonds):
            covalent_edges.extend([(bonds + offset).t(), (bonds[:, [1, 0]] + offset).t()])
            covalent_types.extend([item["bond_type"], item["bond_type"]])
        delta = item["coordinates"][:, None, :] - item["coordinates"][None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        pair = ((distance < cutoff) & (distance > 1e-6)).nonzero(as_tuple=False)
        # pair columns are target i, source j; vector points source -> target.
        radius_edges.append(torch.stack([pair[:, 1] + offset, pair[:, 0] + offset]))
        vector = delta[pair[:, 0], pair[:, 1]]
        radius_vectors.append(vector / distance[pair[:, 0], pair[:, 1]].unsqueeze(-1))
        radius_distances.append(distance[pair[:, 0], pair[:, 1]])
        shifts = item.get("target_shift", torch.full((n,), float("nan")))
        if item.get("target_kind") == "zero_inference_placeholder":
            if require_targets:
                raise ValueError("Inference placeholders cannot be used as training targets")
            shifts = torch.full((n,), float("nan"))
        mask = item["target_nucleus"].ge(0)
        if require_targets:
            mask = mask & torch.isfinite(shifts)
        local = mask.nonzero(as_tuple=False).flatten()
        target_indices.append(local + offset); target_local_indices.append(local)
        targets.append(shifts[local])
        target_nuclei.append(item["target_nucleus"][local])
        target_graph.append(torch.full((len(local),), gid, dtype=torch.long))
        graph_index.append(torch.full((n,), gid, dtype=torch.long))
        molecule_uids.append(item["molecule_uid"]); conformer_uids.append(item["conformer_uid"])
        sample_weights.append(item["sample_weight"])
        heavy_counts.append(item["heavy_atom_count"])
        offset += n
    return {
        "atomic_number": torch.cat(z), "formal_charge": torch.cat(charge),
        "chirality": torch.cat(chirality), "aromatic": torch.cat(aromatic),
        "coordinates": torch.cat(coordinates),
        "edge_index": torch.cat(covalent_edges, 1) if covalent_edges else torch.empty((2, 0), dtype=torch.long),
        "edge_type": torch.cat(covalent_types) if covalent_types else torch.empty(0, dtype=torch.long),
        "radius_edge_index": torch.cat(radius_edges, 1),
        "radius_unit_vector": torch.cat(radius_vectors), "radius_distance": torch.cat(radius_distances),
        "graph_index": torch.cat(graph_index), "target_atom_index": torch.cat(target_indices),
        "target_atom_index_conformer": torch.cat(target_local_indices),
        "target_shift": torch.cat(targets), "target_nucleus": torch.cat(target_nuclei),
        "target_graph_index": torch.cat(target_graph),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
        "heavy_atom_count": torch.tensor(heavy_counts, dtype=torch.long),
        "molecule_uids": molecule_uids, "conformer_uids": conformer_uids,
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

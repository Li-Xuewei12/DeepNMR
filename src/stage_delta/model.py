from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from stage_delta.topology import TopologyEncoder


class DistanceRBF(nn.Module):
    """Smooth radial expansion for invariant spatial edges."""

    def __init__(self, count: int, cutoff: float, gamma: float = 10.0):
        super().__init__()
        self.count = int(count)
        self.cutoff = float(cutoff)
        self.gamma = float(gamma)
        self.register_buffer("centers", torch.linspace(0.0, self.cutoff, self.count))

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-self.gamma * (distance[:, None] - self.centers[None, :]).square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        return rbf * envelope.clamp_min(0.0)[:, None]


class DynamicSpatialEdgeBlock(nn.Module):
    """CASCADE-inspired joint edge/node update on a 3D radius graph.

    Edge states are updated using source atom, target atom, and the preceding
    edge state. Updated edges then gate source-atom messages before summation at
    the target atom. Only invariant scalar quantities enter this block.
    """

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.message_value = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.message_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.node_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        edge_index: torch.Tensor,
        update_edges: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        if update_edges:
            edge_delta = self.edge_update(
                torch.cat([node_state[source], node_state[target], edge_state], dim=-1)
            )
            edge_state = self.edge_norm(edge_state + edge_delta)
        messages = self.message_value(node_state[source]) * self.message_gate(edge_state)
        # CUDA autocast makes the MLP message fp16 while the embedding state
        # can remain fp32. index_add_ requires identical source/destination
        # dtypes, so allocate the accumulation buffer from the message dtype.
        aggregated = torch.zeros(
            node_state.shape, dtype=messages.dtype, device=node_state.device
        )
        aggregated.index_add_(0, target, messages)
        node_delta = self.node_update(torch.cat([node_state, aggregated], dim=-1))
        node_state = self.node_norm(node_state + node_delta)
        return node_state, edge_state


class NucleusShiftModel(nn.Module):
    """3D-dominant, nucleus-specific atom-level chemical-shift predictor."""

    NUCLEUS_TO_CODE = {"1H": 0, "13C": 1}

    def __init__(
        self,
        nucleus: str,
        hidden_dim: int = 256,
        topology_layers: int = 2,
        spatial_layers: int = 4,
        rbf_count: int = 128,
        rbf_gamma: float = 10.0,
        cutoff: float = 5.0,
        dropout: float = 0.05,
        use_spatial_3d: bool = True,
        dynamic_edge_updates: bool = True,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        if nucleus not in self.NUCLEUS_TO_CODE:
            raise ValueError(f"unsupported nucleus {nucleus!r}")
        self.nucleus = nucleus
        self.nucleus_code = self.NUCLEUS_TO_CODE[nucleus]
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)
        self.use_spatial_3d = bool(use_spatial_3d)
        self.dynamic_edge_updates = bool(dynamic_edge_updates)
        self.topology = TopologyEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=int(topology_layers),
            dropout=float(dropout),
            num_atom_classes=2,
            num_bond_classes=8,
        )
        # pretraining heads are checkpoint-compatibility scaffolding only.
        for parameter in list(self.topology.atom_head.parameters()) + list(self.topology.bond_head.parameters()):
            parameter.requires_grad_(False)
        self.rbf = DistanceRBF(rbf_count, cutoff, rbf_gamma)
        self.edge_projection = nn.Sequential(
            nn.Linear(int(rbf_count), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.spatial_blocks = nn.ModuleList(
            DynamicSpatialEdgeBlock(self.hidden_dim, float(dropout))
            for _ in range(int(spatial_layers))
        )
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(float(target_std), dtype=torch.float32))

    def set_target_statistics(self, mean: float, std: float) -> None:
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"invalid target statistics mean={mean}, std={std}")
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(float(std))

    def load_topology_initialization(self, checkpoint_path: str | Path) -> dict:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = payload["model"]
        current = self.topology.state_dict()
        loaded = []
        skipped = []
        for name, value in source.items():
            if name.startswith("atom_head.") or name.startswith("bond_head."):
                skipped.append(name)
                continue
            if name in current and current[name].shape == value.shape:
                current[name] = value
                loaded.append(name)
            else:
                skipped.append(name)
        self.topology.load_state_dict(current, strict=True)
        if not loaded:
            raise RuntimeError("no compatible pretraining encoder parameters were loaded")
        return {
            "checkpoint": str(checkpoint_path),
            "loaded_tensor_count": len(loaded),
            "skipped_tensor_count": len(skipped),
            "loaded": loaded,
            "skipped": skipped,
        }

    def target_mask(self, batch: dict) -> torch.Tensor:
        return batch["target_nucleus"].eq(self.nucleus_code)

    def forward(self, batch: dict) -> dict:
        node_state = self.topology.encode_nodes(batch)
        edge_state = self.edge_projection(self.rbf(batch["radius_distance"]))
        if self.use_spatial_3d:
            for block in self.spatial_blocks:
                node_state, edge_state = block(
                    node_state,
                    edge_state,
                    batch["radius_edge_index"],
                    update_edges=self.dynamic_edge_updates,
                )
        mask = self.target_mask(batch)
        target_node_index = batch["target_atom_index"][mask]
        standardized = self.readout(node_state[target_node_index]).squeeze(-1)
        prediction_ppm = self.target_mean + self.target_std * standardized
        return {
            "prediction_ppm": prediction_ppm,
            "standardized_prediction": standardized,
            "target_mask": mask,
            "node_state": node_state,
            "edge_state": edge_state,
        }

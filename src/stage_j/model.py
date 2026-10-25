from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class TopologyMessagePassingLayer(nn.Module):
    """Checkpoint-compatible copy of the pretraining topology layer.

    It lives locally so Stage J training does not import pretraining's pyarrow-based
    data loader into the GPU environment.
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, state: torch.Tensor, edge_index: torch.Tensor, edge_state: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(state)
        else:
            source, target = edge_index
            message = self.message(state[source] + edge_state)
            aggregated = torch.zeros(
                state.shape, dtype=message.dtype, device=state.device
            )
            aggregated.index_add_(0, target, message)
        return self.norm(
            state + self.dropout(self.update(state + aggregated))
        )


class TopologyEncoder(nn.Module):
    """Same parameter names/shapes as Stage delta's TopologyEncoder."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.atom_embedding = nn.Embedding(120, hidden_dim)
        self.charge_embedding = nn.Embedding(13, hidden_dim)
        self.chirality_embedding = nn.Embedding(6, hidden_dim)
        self.aromatic_embedding = nn.Embedding(4, hidden_dim)
        self.bond_embedding = nn.Embedding(9, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            TopologyMessagePassingLayer(hidden_dim, dropout)
            for _ in range(int(num_layers))
        )
        # These frozen heads keep state_dict compatibility with Stage delta.
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.bond_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 8),
        )

    def encode_nodes(self, batch: dict) -> torch.Tensor:
        state = (
            self.atom_embedding(batch["atomic_number"])
            + self.charge_embedding(batch["formal_charge"])
            + self.chirality_embedding(batch["chirality"])
            + self.aromatic_embedding(batch["aromatic"])
        )
        state = self.input_norm(state)
        edge_state = self.bond_embedding(batch["edge_type"])
        for layer in self.layers:
            state = layer(state, batch["edge_index"], edge_state)
        return state


class GaussianRBF(nn.Module):
    def __init__(self, count: int, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, int(count))
        width = max(float(centers[1] - centers[0]) if len(centers) > 1 else 1.0, 1.0e-3)
        self.gamma = 1.0 / (width * width)
        self.register_buffer("centers", centers)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        rbf = torch.exp(-self.gamma * (distance[:, None] - self.centers[None, :]).square())
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        return rbf * envelope.clamp_min(0.0).unsqueeze(-1)


class EquivariantInteraction(nn.Module):
    """One PaiNN-style scalar/vector interaction and on-site mixing block."""

    def __init__(self, hidden_dim: int, rbf_dim: int, dropout: float):
        super().__init__()
        hidden = int(hidden_dim)
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden * 3),
        )
        self.context_net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden * 3),
        )
        self.vector_v = nn.Linear(hidden, hidden, bias=False)
        self.vector_w = nn.Linear(hidden, hidden, bias=False)
        self.mix_net = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden * 2, hidden * 3),
        )
        self.scalar_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        unit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        message = self.filter_net(rbf) * self.context_net(scalar[source])
        delta_scalar, radial_vector, carried_vector = message.chunk(3, dim=-1)
        scalar_sum = torch.zeros(
            scalar.shape, dtype=delta_scalar.dtype, device=scalar.device
        )
        scalar_sum.index_add_(0, target, delta_scalar)
        vector_message = (
            radial_vector[:, None, :] * unit[:, :, None]
            + carried_vector[:, None, :] * vector[source]
        )
        vector_sum = torch.zeros(
            vector.shape, dtype=vector_message.dtype, device=vector.device
        )
        vector_sum.index_add_(0, target, vector_message)
        scalar = scalar + scalar_sum
        vector = vector + vector_sum

        vector_v = self.vector_v(vector)
        vector_w = self.vector_w(vector)
        vector_norm = torch.sqrt(vector_v.square().sum(dim=1) + 1.0e-8)
        vector_dot = (vector_v * vector_w).sum(dim=1)
        delta_scalar, vector_gate, scalar_vector_gate = self.mix_net(
            torch.cat([scalar, vector_norm], dim=-1)
        ).chunk(3, dim=-1)
        scalar = self.scalar_norm(scalar + delta_scalar + scalar_vector_gate * vector_dot)
        vector = vector + vector_gate[:, None, :] * vector_w
        return scalar, vector


class ShiftInformedPathEquivariantJNetwork(nn.Module):
    """Unified topology + equivariant geometry + explicit path J network."""

    def __init__(
        self,
        hidden_dim: int = 256,
        topology_layers: int = 2,
        equivariant_layers: int = 4,
        rbf_count: int = 64,
        pair_rbf_count: int = 32,
        cutoff: float = 5.0,
        dropout: float = 0.05,
        delta_conditioning: bool = False,
        use_path_features: bool = True,
        use_equivariant_geometry: bool = True,
        predict_components: bool = True,
        target_mean=None,
        target_std=None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)
        self.delta_conditioning = bool(delta_conditioning)
        self.use_path_features = bool(use_path_features)
        self.use_equivariant_geometry = bool(use_equivariant_geometry)
        self.predict_components = bool(predict_components)
        self.topology = TopologyEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=int(topology_layers),
            dropout=float(dropout),
        )
        for parameter in list(self.topology.atom_head.parameters()) + list(
            self.topology.bond_head.parameters()
        ):
            parameter.requires_grad_(False)

        # The two inputs are a standardized Stage delta prediction and an
        # availability mask.  This is deliberately optional so legacy Stage J
        # checkpoints remain architecture-compatible.  Zero-feature controls
        # use the exact same additional parameter count as predicted-delta runs.
        self.delta_projection = (
            nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            )
            if self.delta_conditioning
            else None
        )

        self.rbf = GaussianRBF(int(rbf_count), self.cutoff)
        self.equivariant_blocks = nn.ModuleList(
            EquivariantInteraction(self.hidden_dim, int(rbf_count), float(dropout))
            for _ in range(int(equivariant_layers))
        )
        self.pair_rbf = GaussianRBF(int(pair_rbf_count), self.cutoff)
        h = self.hidden_dim
        self.endpoint_projection = nn.Sequential(nn.Linear(h * 3, h), nn.SiLU())
        self.vector_projection = nn.Sequential(nn.Linear(h * 4, h), nn.SiLU())
        self.path_projection = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU())
        self.geometry_projection = nn.Sequential(
            nn.Linear(int(pair_rbf_count) + 6, h), nn.SiLU(), nn.Linear(h, h),
        )
        self.type_embedding = nn.Embedding(8, h)
        self.shared_pair = nn.Sequential(
            nn.Linear(h * 5, h * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h * 2, h),
            nn.SiLU(),
        )
        self.order_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(h, h // 2),
                nn.SiLU(),
                nn.Linear(h // 2, 5 if self.predict_components else 1),
            )
            for _ in range(3)
        )
        default_shape = (8, 5) if self.predict_components else (8,)
        mean = torch.zeros(default_shape, dtype=torch.float32) if target_mean is None else torch.as_tensor(target_mean, dtype=torch.float32)
        std = torch.ones(default_shape, dtype=torch.float32) if target_std is None else torch.as_tensor(target_std, dtype=torch.float32)
        if not self.predict_components and tuple(mean.shape) == (8, 5):
            mean = mean[:, 0]
        if not self.predict_components and tuple(std.shape) == (8, 5):
            std = std[:, 0]
        if tuple(mean.shape) != default_shape or tuple(std.shape) != default_shape:
            raise ValueError(f"target mean/std must have shape {default_shape}")
        self.register_buffer("target_mean", mean.clone())
        self.register_buffer("target_std", std.clone())

    def load_topology_initialization(self, checkpoint_path: str | Path) -> dict:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = payload["model"]
        current = self.topology.state_dict()
        loaded, skipped = [], []
        for source_name, value in source.items():
            if not source_name.startswith("topology."):
                continue
            name = source_name[len("topology.") :]
            if name.startswith("atom_head.") or name.startswith("bond_head."):
                skipped.append(source_name)
            elif name in current and current[name].shape == value.shape:
                current[name] = value
                loaded.append(source_name)
            else:
                skipped.append(source_name)
        self.topology.load_state_dict(current, strict=True)
        if not loaded:
            raise RuntimeError("no compatible Stage delta topology tensors were loaded")
        return {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "loaded_tensor_count": len(loaded),
            "skipped_tensor_count": len(skipped),
            "loaded": loaded,
            "skipped": skipped,
        }

    @staticmethod
    def _masked_path_mean(
        scalar: torch.Tensor, nodes: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        gathered = scalar[nodes.clamp_min(0)]
        weight = mask.unsqueeze(-1).to(gathered.dtype)
        per_path = (gathered * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        valid_path = mask.any(dim=2).unsqueeze(-1).to(gathered.dtype)
        return (per_path * valid_path).sum(dim=1) / valid_path.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: dict) -> dict:
        scalar = self.topology.encode_nodes(batch)
        if self.delta_conditioning:
            if "delta_node_feature" not in batch:
                raise KeyError(
                    "delta_conditioning=True requires batch['delta_node_feature']"
                )
            delta_feature = batch["delta_node_feature"].to(scalar.dtype)
            if delta_feature.shape != (len(scalar), 2):
                raise ValueError("delta_node_feature must have shape [atoms, 2]")
            scalar = scalar + self.delta_projection(delta_feature)
        vector = scalar.new_zeros((len(scalar), 3, self.hidden_dim))
        radial = self.rbf(batch["radius_distance"])
        if self.use_equivariant_geometry:
            for block in self.equivariant_blocks:
                scalar, vector = block(
                    scalar,
                    vector,
                    batch["radius_edge_index"],
                    radial,
                    batch["radius_unit_vector"],
                )

        pair = batch["pair_index"]
        left, right = pair[:, 0], pair[:, 1]
        scalar_left, scalar_right = scalar[left], scalar[right]
        endpoint = self.endpoint_projection(
            torch.cat(
                [
                    scalar_left + scalar_right,
                    torch.abs(scalar_left - scalar_right),
                    scalar_left * scalar_right,
                ],
                dim=-1,
            )
        )

        displacement = batch["coordinates"][right] - batch["coordinates"][left]
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        direction = displacement / distance.clamp_min(1.0e-8).unsqueeze(-1)
        vector_left, vector_right = vector[left], vector[right]
        vector_dot = (vector_left * vector_right).sum(dim=1)
        projection_left = (vector_left * direction[:, :, None]).sum(dim=1)
        projection_right = (vector_right * direction[:, :, None]).sum(dim=1)
        if self.use_equivariant_geometry:
            vector_feature = self.vector_projection(
                torch.cat(
                    [
                        vector_dot,
                        torch.abs(projection_left + projection_right),
                        torch.abs(projection_left - projection_right),
                        projection_left * projection_right,
                    ],
                    dim=-1,
                )
            )
        else:
            vector_feature = scalar_left.new_zeros((len(left), self.hidden_dim))

        path_mean = self._masked_path_mean(
            scalar, batch["path_nodes"], batch["path_mask"]
        )
        internal_mask = batch["path_mask"].clone()
        internal_mask[:, :, 0] = False
        path_length = batch["path_mask"].sum(dim=2)
        last = (path_length - 1).clamp_min(0)
        internal_mask.scatter_(2, last[:, :, None], False)
        internal_mean = self._masked_path_mean(
            scalar, batch["path_nodes"], internal_mask
        )
        if self.use_path_features:
            path_feature = self.path_projection(
                torch.cat([path_mean, internal_mean], dim=-1)
            )
        else:
            path_feature = scalar_left.new_zeros((len(left), self.hidden_dim))

        pair_radial = self.pair_rbf(distance)[:, None, :].expand(
            -1, batch["path_geometry"].shape[1], -1
        )
        per_path_geometry = self.geometry_projection(
            torch.cat([pair_radial, batch["path_geometry"]], dim=-1)
        )
        valid_path = batch["path_mask"].any(dim=2).unsqueeze(-1).to(
            per_path_geometry.dtype
        )
        if self.use_equivariant_geometry:
            geometry = (per_path_geometry * valid_path).sum(dim=1) / valid_path.sum(
                dim=1
            ).clamp_min(1.0)
        else:
            geometry = scalar_left.new_zeros((len(left), self.hidden_dim))
        coupling_type = batch["coupling_type"]
        pair_state = self.shared_pair(
            torch.cat(
                [
                    endpoint,
                    vector_feature,
                    path_feature,
                    geometry,
                    self.type_embedding(coupling_type),
                ],
                dim=-1,
            )
        )
        standardized = pair_state.new_empty(
            (len(pair_state), 5) if self.predict_components else (len(pair_state),)
        )
        order = coupling_type.new_tensor([1, 1, 2, 2, 2, 3, 3, 3])[coupling_type]
        for order_index, expert in enumerate(self.order_experts, start=1):
            selected = order.eq(order_index)
            if selected.any():
                expert_output = expert(pair_state[selected])
                standardized[selected] = (
                    expert_output if self.predict_components else expert_output.squeeze(-1)
                )
        prediction_hz = standardized * self.target_std[coupling_type] + self.target_mean[coupling_type]
        return {
            "standardized_prediction": standardized,
            "prediction_hz": prediction_hz,
            "scalar_state": scalar,
            "vector_state": vector,
        }

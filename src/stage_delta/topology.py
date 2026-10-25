"""Topology primitives owned by the production Stage delta package.

These classes preserve the parameter names and tensor shapes of the historical
encoder so that frozen Stage delta checkpoints remain strictly loadable.
No separate pretraining package is required.
"""
from __future__ import annotations

import torch
from torch import nn


BOND_MASK = 8
CHARGE_UNKNOWN = 11
CHIRAL_UNKNOWN = 4
BOND_UNKNOWN = 7

CHIRAL_TO_ID = {
    "CHI_UNSPECIFIED": 0,
    "CHI_TETRAHEDRAL_CW": 1,
    "CHI_TETRAHEDRAL_CCW": 2,
    "CHI_OTHER": 3,
}
BOND_TO_ID = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3,
    "QUADRUPLE": 4,
    "DATIVE": 5,
    "IONIC": 6,
}


def charge_id(value: int) -> int:
    value = int(value)
    return value + 5 if -5 <= value <= 5 else CHARGE_UNKNOWN


class MessagePassingLayer(nn.Module):
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
        self, h: torch.Tensor, edge_index: torch.Tensor, edge_h: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(h)
        else:
            source, target = edge_index
            messages = self.message(h[source] + edge_h)
            aggregated = torch.zeros(
                h.shape, dtype=messages.dtype, device=h.device
            )
            aggregated.index_add_(0, target, messages)
        return self.norm(h + self.dropout(self.update(h + aggregated)))


class TopologyEncoder(nn.Module):
    """Stage delta topology encoder with checkpoint-compatible auxiliary heads."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_atom_classes: int = 119,
        num_bond_classes: int = 8,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_atom_classes = int(num_atom_classes)
        self.num_bond_classes = int(num_bond_classes)
        self.atom_embedding = nn.Embedding(120, hidden_dim)
        self.charge_embedding = nn.Embedding(13, hidden_dim)
        self.chirality_embedding = nn.Embedding(6, hidden_dim)
        self.aromatic_embedding = nn.Embedding(4, hidden_dim)
        self.bond_embedding = nn.Embedding(BOND_MASK + 1, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim, dropout) for _ in range(num_layers)
        )
        # Kept only so historical Stage delta checkpoints load strictly. They are
        # frozen and never called by the production shift forward path.
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atom_classes),
        )
        self.bond_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bond_classes),
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



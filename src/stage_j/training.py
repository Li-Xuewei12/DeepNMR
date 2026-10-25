from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def balanced_component_loss(
    standardized_prediction: torch.Tensor,
    standardized_target: torch.Tensor,
    prediction_hz: torch.Tensor,
    coupling_type: torch.Tensor,
    target_std: torch.Tensor,
    component_weights: torch.Tensor,
    consistency_weight: float,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    per_type = []
    for type_id in range(8):
        selected = coupling_type.eq(type_id)
        if not selected.any():
            continue
        elementwise = F.smooth_l1_loss(
            standardized_prediction[selected],
            standardized_target[selected],
            beta=float(beta),
            reduction="none",
        )
        per_component = elementwise.mean(dim=0)
        per_type.append(
            (per_component * component_weights).sum()
            / component_weights.sum().clamp_min(1.0e-8)
        )
    supervised = torch.stack(per_type).mean()
    consistency = (
        prediction_hz[:, 0] - prediction_hz[:, 1:].sum(dim=1)
    ) / target_std[coupling_type, 0]
    consistency_loss = F.smooth_l1_loss(
        consistency,
        torch.zeros_like(consistency),
        beta=float(beta),
    )
    total = supervised + float(consistency_weight) * consistency_loss
    return total, {
        "supervised_normalized_smooth_l1": float(supervised.detach()),
        "consistency_normalized_smooth_l1": float(consistency_loss.detach()),
        "total_loss": float(total.detach()),
    }


"""Strict loaders for the released checkpoints (trusted local artifacts only)."""
from pathlib import Path
import torch
from stage_delta.model import NucleusShiftModel

ROOT = Path(__file__).resolve().parents[2]


def load_stage_delta(path, nucleus):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["config"]["nucleus"] != nucleus:
        raise ValueError(f"Checkpoint nucleus does not match {nucleus}")
    stats = payload["target_statistics"]
    model = NucleusShiftModel(
        nucleus=nucleus, target_mean=float(stats["mean"]),
        target_std=float(stats["std"]), **payload["config"]["model"],
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def load_overlay(nucleus, weights_dir=ROOT / "weights"):
    from deepnmr.overlay import SimplifiedExperimentalShift
    weights_dir = Path(weights_dir)
    payload = torch.load(weights_dir / f"overlay_{nucleus}.pt", map_location="cpu", weights_only=False)
    model = SimplifiedExperimentalShift(nucleus, weights_dir / f"stage_delta_{nucleus}.pt", "full")
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def load_j(weights_dir=ROOT / "weights"):
    from stage_j.model import ShiftInformedPathEquivariantJNetwork
    payload = torch.load(Path(weights_dir) / "stage_j.pt", map_location="cpu", weights_only=False)
    model = ShiftInformedPathEquivariantJNetwork(
        target_mean=payload["target_mean"], target_std=payload["target_std"],
        **payload["config"]["model"],
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()

#!/usr/bin/env python3
"""Experimental chemical-shift fine-tuning on training/validation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
from stage_delta.data import GraphDataset, collate_graphs, move_batch
from deepnmr.checkpoints import load_stage_delta
DEFAULT_DATA = ROOT / "data/assigned_overlay"
DEFAULT_CKPT = {n: ROOT / "weights" / ("stage_delta_" + n + ".pt") for n in ["1H", "13C"]}


class ZeroResidualAdapter(nn.Module):
    def __init__(self, hidden: int, target_mean: float, target_std: float):
        super().__init__()
        if not math.isfinite(float(target_std)) or float(target_std) <= 0:
            raise ValueError(f"invalid target_std={target_std}")
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))
        self.network = nn.Sequential(
            nn.Linear(int(hidden) + 1, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), int(hidden) // 2), nn.SiLU(),
            nn.Linear(int(hidden) // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, node_state: torch.Tensor, raw_ppm: torch.Tensor) -> torch.Tensor:
        # A fixed /100 scale nearly erases proton-shift information while leaving
        # carbon on a different numerical scale.  Reuse the nucleus-specific
        # Stage delta target statistics so this feature has the same semantics in both
        # heads and remains deterministic at inference.
        standardized_raw = (raw_ppm - self.target_mean) / self.target_std
        feature = torch.cat([node_state, standardized_raw[:, None]], dim=-1)
        return raw_ppm + self.network(feature).squeeze(-1)


class SimplifiedExperimentalShift(nn.Module):
    def __init__(self, nucleus: str, checkpoint: Path, unfreeze: str = "full"):
        super().__init__()
        self.nucleus = nucleus
        self.shift = load_stage_delta(checkpoint, nucleus)
        self.unfreeze = str(unfreeze)
        self.adapter = ZeroResidualAdapter(
            self.shift.hidden_dim,
            float(self.shift.target_mean),
            float(self.shift.target_std),
        )
        for parameter in self.shift.parameters():
            parameter.requires_grad_(False)
        if self.unfreeze not in {"adapter_only", "readout", "tail1", "tail2", "full"}:
            raise ValueError(f"unsupported unfreeze policy {self.unfreeze!r}")
        if self.unfreeze in {"readout", "tail1", "tail2", "full"}:
            for parameter in self.shift.readout.parameters():
                parameter.requires_grad_(True)
        if self.unfreeze in {"tail1", "tail2"}:
            count = 1 if self.unfreeze == "tail1" else 2
            for block in self.shift.spatial_blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        if self.unfreeze == "full":
            for parameter in self.shift.parameters():
                parameter.requires_grad_(True)
            # pretraining auxiliary heads are compatibility scaffolding and are not
            # used by this forward path, so do not optimize them accidentally.
            for parameter in list(self.shift.topology.atom_head.parameters()) + list(
                self.shift.topology.bond_head.parameters()
            ):
                parameter.requires_grad_(False)

    def forward(self, batch: dict) -> torch.Tensor:
        output = self.shift(batch)
        mask = output["target_mask"]
        node_index = batch["target_atom_index"][mask]
        return self.adapter(output["node_state"][node_index], output["prediction_ppm"])

    def parameter_groups(self, backbone_lr: float, adapter_lr: float) -> list[dict]:
        adapter = list(self.adapter.parameters())
        adapter_ids = {id(value) for value in adapter}
        backbone = [
            value for value in self.parameters()
            if value.requires_grad and id(value) not in adapter_ids
        ]
        return [
            {"params": backbone, "lr": float(backbone_lr), "name": "stage_delta_tail"},
            {"params": adapter, "lr": float(adapter_lr), "name": "zero_residual"},
        ]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def metrics(frame: pd.DataFrame) -> dict:
    error = frame.prediction_ppm - frame.target_ppm
    absolute = error.abs()
    squared = error ** 2
    denominator = float(((frame.target_ppm - frame.target_ppm.mean()) ** 2).sum())
    molecule = frame.assign(abs_error=absolute).groupby("molecule_uid").abs_error.mean()
    return {
        "count": int(len(frame)),
        "molecules": int(frame.molecule_uid.nunique()),
        "micro_mae_ppm": float(absolute.mean()),
        "micro_rmse_ppm": float(np.sqrt(squared.mean())),
        "r2": float(1.0 - squared.sum() / denominator) if denominator > 0 else None,
        "molecule_balanced_mae_ppm": float(molecule.mean()),
        "mean_signed_error_ppm": float(error.mean()),
        "absolute_error_p95_ppm": float(absolute.quantile(0.95)),
    }


def supervised_loss(prediction, target, graph_ids, name: str, huber_beta: float):
    if name == "l1":
        return torch.nn.functional.l1_loss(prediction, target)
    if name == "huber":
        return torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=float(huber_beta)
        )
    if name == "molecule_l1":
        per_atom = (prediction - target).abs()
        unique = torch.unique(graph_ids)
        return torch.stack([per_atom[graph_ids.eq(index)].mean() for index in unique]).mean()
    raise ValueError(name)


def run_epoch(
    model, loader, device, optimizer=None, amp: bool = True,
    loss_name: str = "l1", huber_beta: float = 0.1,
    accumulation_steps: int = 1, scaler=None,
):
    training = optimizer is not None
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train(training)
    amp = bool(amp) and torch.device(device).type == "cuda"
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=training and amp)
    rows = []
    total_loss = 0.0
    total_count = 0
    gradient_norms = []
    optimizer_steps = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=amp):
            prediction = model(batch).float()
            mask = batch["target_nucleus"].eq(model.shift.nucleus_code)
            target = batch["target_shift"][mask].float()
            loss = supervised_loss(
                prediction, target, batch["target_graph_index"][mask], loss_name, huber_beta
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite overlay loss")
        if training:
            window_start = (step // accumulation_steps) * accumulation_steps
            window_size = min(accumulation_steps, len(loader) - window_start)
            scaler.scale(loss / window_size).backward()
            update = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
            if update:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [value for value in model.parameters() if value.requires_grad], 1.0
                )
                gradient_norms.append(float(gradient_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        count = int(target.numel())
        total_loss += float(loss.detach()) * count
        total_count += count
        graph_ids = batch["target_graph_index"][mask]
        atom_ids = batch["target_atom_index_conformer"][mask]
        for observed, predicted, graph_id, atom_id in zip(
            target.detach().cpu(), prediction.detach().cpu(), graph_ids.cpu(), atom_ids.cpu()
        ):
            rows.append({
                "molecule_uid": batch["molecule_uids"][int(graph_id)],
                "atom_index_conformer": int(atom_id),
                "target_ppm": float(observed),
                "prediction_ppm": float(predicted),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Overlay split contains no requested targets")
    result = metrics(frame)
    result["loss"] = total_loss / max(total_count, 1)
    result["optimizer_steps"] = int(optimizer_steps)
    result["gradient_norm_mean"] = float(np.mean(gradient_norms)) if gradient_norms else None
    result["gradient_norm_max"] = float(np.max(gradient_norms)) if gradient_norms else None
    return result, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nucleus", choices=["1H", "13C"], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--validation-path", type=Path)
    parser.add_argument("--stage-delta-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--unfreeze", choices=["adapter_only", "readout", "tail1", "tail2", "full"],
        default="full",
    )
    parser.add_argument("--loss", choices=["l1", "huber", "molecule_l1"], default="l1")
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-train-predictions", action="store_true")
    args = parser.parse_args()
    train_path = args.train_path or (args.data_root / args.nucleus / "train.pt")
    validation_path = args.validation_path or (
        args.data_root / args.nucleus / "validation.pt"
    )
    if any("test" in str(path).lower() for path in (args.data_root, train_path, validation_path)):
        raise ValueError("strict-test data path is forbidden")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    resolved_args = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    atomic_json(args.output / "config" / "resolved_args.json", resolved_args)
    seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = args.stage_delta_checkpoint or DEFAULT_CKPT[args.nucleus]
    nucleus_code = 0 if args.nucleus == "1H" else 1
    train = GraphDataset(train_path)
    validation = GraphDataset(validation_path)
    train.graphs = [value for value in train.graphs if value["target_nucleus"].eq(nucleus_code).any()]
    validation.graphs = [value for value in validation.graphs if value["target_nucleus"].eq(nucleus_code).any()]
    train_limit = 128 if args.smoke and not args.train_limit else args.train_limit
    validation_limit = 64 if args.smoke and not args.validation_limit else args.validation_limit
    if train_limit:
        train.graphs = train.graphs[:train_limit]
    if validation_limit:
        validation.graphs = validation.graphs[:validation_limit]
    model = SimplifiedExperimentalShift(args.nucleus, checkpoint, args.unfreeze).to(device)
    collate = partial(collate_graphs, cutoff=model.shift.cutoff)
    options = {
        "batch_size": args.batch_size, "num_workers": args.workers,
        "pin_memory": True, "collate_fn": collate,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **options)
    validation_loader = DataLoader(validation, shuffle=False, **options)
    atomic_json(
        args.output / "provenance" / "run_environment.json",
        {
            "schema": "deepnmr_stage2b_shift_reproducibility_provenance_v1",
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "seed": int(args.seed),
            "deterministic_algorithms": bool(args.deterministic),
            "stage_delta_checkpoint": str(checkpoint.resolve()),
            "stage_delta_checkpoint_sha256": sha256(checkpoint),
            "train_data": str(train_path.resolve()),
            "train_data_sha256": sha256(train_path),
            "validation_data": str(validation_path.resolve()),
            "validation_data_sha256": sha256(validation_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
        },
    )
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.backbone_lr, args.adapter_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=[1e-7, 1e-6]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    max_epochs = 2 if args.smoke else args.max_epochs
    min_delta = args.min_delta
    if min_delta is None:
        min_delta = 0.001 if args.nucleus == "1H" else 0.01
    # `best_any` selects the checkpoint. `best_significant` alone drives
    # patience.  The old implementation used min_delta for both, silently
    # discarding a numerically better checkpoint whenever the gain was smaller
    # than min_delta.
    best_any = float("inf")
    best_significant = float("inf")
    bad = 0
    history = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics, _ = run_epoch(
            model, train_loader, device, optimizer,
            loss_name=args.loss, huber_beta=args.huber_beta,
            accumulation_steps=args.accumulation_steps, scaler=scaler,
        )
        with torch.no_grad():
            validation_metrics, _ = run_epoch(
                model, validation_loader, device,
                loss_name=args.loss, huber_beta=args.huber_beta,
            )
        monitor = float(validation_metrics["micro_mae_ppm"])
        if not math.isfinite(monitor):
            raise FloatingPointError("Non-finite validation MAE")
        checkpoint_improved = monitor < best_any
        significant_improvement = monitor < best_significant - float(min_delta)
        if checkpoint_improved:
            best_any = monitor
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "monitor": monitor,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "resolved_args": resolved_args},
                args.output / "best_validation.pt",
            )
        if significant_improvement:
            best_significant = monitor
            bad = 0
        else:
            bad += 1
        scheduler.step(monitor)
        row = {
            "epoch": epoch, "seconds": time.time() - epoch_started,
            "train": train_metrics, "validation": validation_metrics,
            "checkpoint_improved": checkpoint_improved,
            "significant_improvement": significant_improvement,
            "bad_epochs": bad,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else 0.0
            ),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        append_jsonl(args.output / "logs" / "epochs.jsonl", row)
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "monitor": monitor,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "bad_epochs": bad, "resolved_args": resolved_args},
            args.output / "last_training_state.pt",
        )
        print(json.dumps(row), flush=True)
        if bad >= args.patience:
            break
    payload = torch.load(args.output / "best_validation.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    with torch.no_grad():
        final_validation, predictions = run_epoch(
            model, validation_loader, device,
            loss_name=args.loss, huber_beta=args.huber_beta,
        )
    predictions.to_parquet(args.output / "validation_predictions.parquet", index=False)
    with torch.no_grad():
        selected_train, train_predictions = run_epoch(
            model, train_loader, device,
            loss_name=args.loss, huber_beta=args.huber_beta,
        )
    if args.save_train_predictions:
        train_predictions.to_parquet(
            args.output / "train_predictions.parquet", index=False
        )
    summary = {
        "status": "completed", "model": "DeepNMR_simplified_real_shift",
        "nucleus": args.nucleus, "seed": int(args.seed), "strict_test_locked": True,
        "test_evaluation_count": 0, "train_molecules": len(train),
        "validation_molecules": len(validation), "best_epoch": int(payload["epoch"]),
        "epochs_trained": len(history), "validation": final_validation,
        "selected_train": selected_train,
        "wall_seconds": time.time() - started,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "stage_delta_checkpoint": str(checkpoint.resolve()),
        "unfreeze": args.unfreeze,
        "loss_name": args.loss,
        "micro_batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "effective_batch_size": int(args.batch_size * args.accumulation_steps),
        "huber_beta": float(args.huber_beta),
        "min_delta_patience_only": float(min_delta),
        "checkpoint_monitor": "absolute minimum validation micro_mae_ppm",
        "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "deterministic_algorithms": bool(args.deterministic),
        "train_predictions_saved": bool(args.save_train_predictions),
    }
    atomic_json(args.output / "history.json", history)
    atomic_json(args.output / "summary.json", summary)
    (args.output / "training_complete.txt").write_text("completed\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

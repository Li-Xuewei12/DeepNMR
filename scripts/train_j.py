#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGEJ_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(STAGEJ_ROOT))

from stage_j.constants import COMPONENT_NAMES, TYPE_NAMES  # noqa: E402
from stage_j.data import (  # noqa: E402
    CouplingRecordDataset,
    collate_couplings,
    move_batch,
)
from stage_j.metrics import compute_metrics  # noqa: E402
from stage_j.model import ShiftInformedPathEquivariantJNetwork  # noqa: E402
from stage_j.training import (  # noqa: E402
    append_jsonl,
    atomic_torch_save,
    balanced_component_loss,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_statistics(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (8, 5) or std.shape != (8, 5) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid target statistics")
    return mean, std, payload


def make_loader(dataset, config: dict, shuffle: bool, seed: int | None = None):
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        generator=generator,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
        collate_fn=partial(
            collate_couplings,
            cutoff=float(config["cutoff"]),
            delta_mode=str(config.get("delta_mode", "none")),
        ),
    )


def model_loss(model, output, batch, component_weights, consistency_weight, beta):
    standardized_target = (
        batch["targets_hz"] - model.target_mean[batch["coupling_type"]]
    ) / model.target_std[batch["coupling_type"]]
    return balanced_component_loss(
        output["standardized_prediction"],
        standardized_target,
        output["prediction_hz"],
        batch["coupling_type"],
        model.target_std,
        component_weights,
        consistency_weight,
        beta,
    )


@torch.no_grad()
def evaluate(model, loader, device, component_weights, consistency_weight, beta):
    model.eval()
    targets, predictions, coupling_types = [], [], []
    molecule_ids, atom_i, atom_j = [], [], []
    loss_sum, pair_count = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        loss, _ = model_loss(
            model,
            output,
            batch,
            component_weights,
            consistency_weight,
            beta,
        )
        n = len(batch["coupling_type"])
        loss_sum += float(loss) * n
        pair_count += n
        targets.append(
            batch["targets_hz"]
            .cpu()
            .numpy()
        )
        predictions.append(output["prediction_hz"].float().cpu().numpy())
        coupling_types.append(batch["coupling_type"].cpu().numpy())
        molecule_ids.extend(batch["pair_molecule_ids"])
        atom_i.extend(batch["pair_atom_i"])
        atom_j.extend(batch["pair_atom_j"])
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    coupling_type = np.concatenate(coupling_types)
    return {
        "loss": loss_sum / max(pair_count, 1),
        "metrics": compute_metrics(target, prediction, coupling_type),
        "target": target,
        "prediction": prediction,
        "coupling_type": coupling_type,
        "molecule_id": molecule_ids,
        "atom_i": atom_i,
        "atom_j": atom_j,
    }


def write_predictions(path: Path, result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["molecule_id", "atom_i", "atom_j", "coupling_type"]
        for name in COMPONENT_NAMES:
            header.extend([f"target_{name}_hz", f"prediction_{name}_hz", f"error_{name}_hz"])
        writer.writerow(header)
        for row_index in range(len(result["coupling_type"])):
            row = [
                result["molecule_id"][row_index],
                result["atom_i"][row_index],
                result["atom_j"][row_index],
                TYPE_NAMES[int(result["coupling_type"][row_index])],
            ]
            for component_id in range(5):
                target = float(result["target"][row_index, component_id])
                prediction = float(result["prediction"][row_index, component_id])
                row.extend([target, prediction, prediction - target])
            writer.writerow(row)


@torch.no_grad()
def invariance_audit(model, batch, device) -> dict:
    model.eval()
    batch = move_batch(batch, device)
    reference = model(batch)["prediction_hz"]
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = dict(batch)
    rotated["coordinates"] = batch["coordinates"] @ q + torch.tensor(
        [2.1, -1.7, 0.6], device=device
    )
    rotated["radius_unit_vector"] = batch["radius_unit_vector"] @ q
    rotation_difference = float(
        (reference - model(rotated)["prediction_hz"]).abs().max()
    )

    reflection = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0], device=device)
    )
    reflected = dict(batch)
    reflected["coordinates"] = batch["coordinates"] @ reflection
    reflected["radius_unit_vector"] = batch["radius_unit_vector"] @ reflection
    reflection_difference = float(
        (reference - model(reflected)["prediction_hz"]).abs().max()
    )

    swapped = dict(batch)
    swapped["pair_index"] = batch["pair_index"][:, [1, 0]]
    reversed_nodes = batch["path_nodes"].clone()
    for row_index, path_lengths in enumerate(
        batch["path_mask"].sum(dim=2).tolist()
    ):
        for path_index, length in enumerate(path_lengths):
            reversed_nodes[row_index, path_index, :length] = torch.flip(
                batch["path_nodes"][row_index, path_index, :length], dims=[0]
            )
    swapped["path_nodes"] = reversed_nodes
    pair_swap_difference = float(
        (reference - model(swapped)["prediction_hz"]).abs().max()
    )
    return {
        "rotation_translation_max_abs_difference_hz": rotation_difference,
        "reflection_max_abs_difference_hz": reflection_difference,
        "pair_swap_max_abs_difference_hz": pair_swap_difference,
    }


def main() -> None:
    config_path = parse_args().config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device_name = str(config["runtime"]["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    seed_everything(int(config["seed"]))

    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for child in ["config", "logs", "checkpoints", "evaluation", "reports", "provenance"]:
        (output / child).mkdir(parents=True, exist_ok=False)
    (output / "config" / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    cache = Path(config["data"]["cache_dir"]).resolve()
    train_set = CouplingRecordDataset(
        cache / "train.pkl", int(config["data"].get("train_limit", 0))
    )
    validation_set = CouplingRecordDataset(
        cache / "validation.pkl", int(config["data"].get("validation_limit", 0))
    )
    run_final_test = bool(config["training"].get("run_final_test", True))
    test_set = (
        CouplingRecordDataset(
            cache / "test.pkl", int(config["data"].get("test_limit", 0))
        )
        if run_final_test
        else None
    )
    statistics_path = Path(
        config["data"].get("statistics_path", cache / "target_statistics.json")
    ).resolve()
    mean, std, statistics = load_statistics(statistics_path)
    model_config = dict(config["model"])
    model = ShiftInformedPathEquivariantJNetwork(
        target_mean=mean, target_std=std, **model_config
    )
    initialization = {"kind": str(config["initialization"]["kind"])}
    if initialization["kind"] == "topology":
        initialization.update(model.load_topology_initialization(config["initialization"]["checkpoint"]))
    elif initialization["kind"] != "random":
        raise ValueError(f"unsupported initialization: {initialization['kind']}")
    model = model.to(device)

    training = config["training"]
    loader_config = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "cutoff": model_config["cutoff"],
        "delta_mode": config["data"].get("delta_mode", "none"),
    }
    train_loader = make_loader(
        train_set, loader_config, True, seed=int(config["seed"])
    )
    validation_loader = make_loader(validation_set, loader_config, False)
    test_loader = make_loader(test_set, loader_config, False) if test_set else None
    component_weights = torch.as_tensor(
        training["component_weights"], dtype=torch.float32, device=device
    )

    topology_parameters = list(model.topology.parameters())
    topology_ids = {id(parameter) for parameter in topology_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in topology_ids
    ]
    trainable_topology = [parameter for parameter in topology_parameters if parameter.requires_grad]
    peak_lr = float(training["peak_lr"])
    topology_scale = (
        float(training["topology_lr_scale"])
        if initialization["kind"] == "topology"
        else 1.0
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_topology, "lr": peak_lr * topology_scale},
            {"params": other_parameters, "lr": peak_lr},
        ],
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    total_steps = max_epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * float(training["warmup_ratio"])))
    minimum_ratio = float(training["minimum_lr"]) / peak_lr

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(
            1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        )
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(training["mixed_precision"]) and device.type == "cuda"
    )
    provenance = {
        "schema": "stage_j_experiment_provenance_v1",
        "model": "Shift-informed Path-Equivariant J Network",
        "cache": str(cache),
        "cache_audit": str((cache / "data_audit.json").resolve()),
        "train_molecules": len(train_set),
        "validation_molecules": len(validation_set),
        "test_molecules": len(test_set) if test_set else 0,
        "delta_conditioning": bool(model_config.get("delta_conditioning", False)),
        "delta_is_prediction_target": False,
        "j_output_targets": list(COMPONENT_NAMES),
        "delta_mode": str(config["data"].get("delta_mode", "none")),
        "initialization": initialization,
        "target_statistics": statistics,
        "selection_metric": "validation total_macro.mean_log_mae",
        "test_policy": (
            "once_after_reloading_best_validation_checkpoint"
            if run_final_test
            else "disabled_during_ablation_model_selection"
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    (output / "provenance" / "data_model_and_initialization.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    sample = next(iter(validation_loader))
    invariance = invariance_audit(model, sample, device)
    tolerance = float(config["acceptance"]["invariance_tolerance_hz"])
    if max(invariance.values()) > tolerance:
        raise RuntimeError(f"invariance audit failed: {invariance}")
    (output / "reports" / "initial_invariance.json").write_text(
        json.dumps(invariance, indent=2), encoding="utf-8"
    )

    beta = float(training["smooth_l1_beta"])
    consistency_weight = float(training["consistency_weight"])
    best, best_epoch, bad_epochs, global_step = float("inf"), 0, 0, 0
    best_significant = float("inf")
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.time()
        model.train()
        running_loss, running_pairs = 0.0, 0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=bool(training["mixed_precision"]) and device.type == "cuda"
            ):
                output_batch = model(batch)
                loss, loss_parts = model_loss(
                    model,
                    output_batch,
                    batch,
                    component_weights,
                    consistency_weight,
                    beta,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch}, batch={batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            pair_count = len(batch["coupling_type"])
            running_loss += float(loss.detach()) * pair_count
            running_pairs += pair_count
            if global_step == 1 or global_step % int(training["log_interval_steps"]) == 0:
                append_jsonl(
                    output / "logs" / "train_steps.jsonl",
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "batch": batch_index,
                        "pairs": pair_count,
                        "gradient_norm": float(gradient_norm),
                        "topology_lr": optimizer.param_groups[0]["lr"],
                        "other_lr": optimizer.param_groups[1]["lr"],
                        **loss_parts,
                    },
                )

        validation = evaluate(
            model,
            validation_loader,
            device,
            component_weights,
            consistency_weight,
            beta,
        )
        monitor = float(validation["metrics"]["total_macro"]["mean_log_mae"])
        if not math.isfinite(monitor):
            raise FloatingPointError("Non-finite validation score")
        improved = monitor < best
        if improved:
            best, best_epoch = monitor, epoch
        if monitor < best_significant - float(training["early_stopping_min_delta"]):
            best_significant, bad_epochs = monitor, 0
        else:
            bad_epochs += 1
        payload = {
            "schema": "shift_informed_path_equivariant_j_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "best_validation_mean_log_mae": best,
            "target_mean": mean,
            "target_std": std,
            "config": config,
            "initialization": initialization,
        }
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
        if improved:
            atomic_torch_save(payload, output / "checkpoints" / "best_validation.pt")
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_pairs, 1),
            "validation_loss": validation["loss"],
            "validation": validation["metrics"],
            "selection_metric": monitor,
            "best_selection_metric": best,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "topology_lr": optimizer.param_groups[0]["lr"],
            "other_lr": optimizer.param_groups[1]["lr"],
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
            "gpu_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        append_jsonl(output / "logs" / "epochs.jsonl", epoch_row)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": epoch_row["train_loss"],
                    "validation_loss": epoch_row["validation_loss"],
                    "validation_total_micro_mae_hz": validation["metrics"]["total_micro"]["mae"],
                    "validation_total_macro_mae_hz": validation["metrics"]["total_macro"]["mae"],
                    "validation_mean_log_mae": monitor,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "epoch_seconds": epoch_row["epoch_seconds"],
                    "gpu_peak_memory_bytes": epoch_row["gpu_peak_memory_bytes"],
                }
            ),
            flush=True,
        )
        if bad_epochs >= int(training["early_stopping_patience_epochs"]):
            print(
                json.dumps(
                    {
                        "early_stop": True,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_validation_mean_log_mae": best,
                    }
                ),
                flush=True,
            )
            break

    best_payload = torch.load(
        output / "checkpoints" / "best_validation.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_payload["model"])
    selected_validation = evaluate(
        model,
        validation_loader,
        device,
        component_weights,
        consistency_weight,
        beta,
    )
    # Keep row-level validation outputs so future evaluator revisions can be
    # applied without retraining or re-opening the strict test set.
    write_predictions(
        output / "evaluation" / "validation_predictions.csv",
        selected_validation,
    )
    test_result = None
    if run_final_test:
        test_result = evaluate(
            model,
            test_loader,
            device,
            component_weights,
            consistency_weight,
            beta,
        )
        write_predictions(output / "evaluation" / "test_predictions.csv", test_result)
    selected_invariance = invariance_audit(model, sample, device)
    summary = {
        "status": "completed",
        "schema": "shift_informed_path_equivariant_j_v1",
        "initialization": initialization,
        "best_epoch": int(best_payload["epoch"]),
        "epochs_trained": epoch,
        "early_stopped": epoch < max_epochs,
        "optimizer_steps": global_step,
        "selected_validation": {
            "loss": selected_validation["loss"],
            "metrics": selected_validation["metrics"],
        },
        "test": (
            {"loss": test_result["loss"], "metrics": test_result["metrics"]}
            if test_result is not None
            else None
        ),
        "test_evaluation_count": int(test_result is not None),
        "selected_invariance": selected_invariance,
        "wall_seconds": time.time() - started,
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    (output / "reports" / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output),
                "best_epoch": summary["best_epoch"],
                "epochs_trained": summary["epochs_trained"],
                "test_total_micro": (
                    summary["test"]["metrics"]["total_micro"]
                    if summary["test"] is not None
                    else None
                ),
                "test_total_macro": (
                    summary["test"]["metrics"]["total_macro"]
                    if summary["test"] is not None
                    else None
                ),
                "test_evaluation_count": summary["test_evaluation_count"],
                "wall_seconds": summary["wall_seconds"],
                "gpu_peak_memory_bytes": summary["gpu_peak_memory_bytes"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from stage_delta.data import GraphDataset, collate_graphs, move_batch
from .model import NucleusShiftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filtered_dataset(path: Path, nucleus_code: int, limit: int = 0) -> GraphDataset:
    dataset = GraphDataset(path)
    dataset.graphs = [
        graph for graph in dataset.graphs
        if graph["target_nucleus"].eq(nucleus_code).any()
    ]
    if limit:
        dataset.graphs = dataset.graphs[: int(limit)]
    return dataset


def target_statistics(dataset: GraphDataset, nucleus_code: int) -> dict:
    if not dataset.graphs:
        raise ValueError("Training split contains no requested nucleus")
    values = torch.cat([
        graph["target_shift"][graph["target_nucleus"].eq(nucleus_code)]
        for graph in dataset.graphs
    ]).float()
    if not torch.isfinite(values).all() or float(values.std(unbiased=False)) <= 0:
        raise ValueError("Training targets must be finite with nonzero variance")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "count": int(values.numel()),
    }


def metric_block(frame: pd.DataFrame) -> dict:
    error = frame.prediction - frame.target
    absolute = error.abs()
    squared = error ** 2
    denominator = float(((frame.target - frame.target.mean()) ** 2).sum())
    molecule = frame.assign(abs_error=absolute, squared_error=squared).groupby("molecule_uid")
    return {
        "count": int(len(frame)),
        "molecules": int(frame.molecule_uid.nunique()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(squared.mean())),
        "r2": float(1.0 - squared.sum() / denominator) if denominator > 0 else None,
        "pearson_r": float(frame.target.corr(frame.prediction)),
        "mean_signed_error": float(error.mean()),
        "molecule_balanced_mae": float(molecule.abs_error.mean().mean()),
        "molecule_balanced_rmse": float(np.sqrt(molecule.squared_error.mean()).mean()),
        "absolute_error_p50": float(absolute.quantile(0.50)),
        "absolute_error_p90": float(absolute.quantile(0.90)),
        "absolute_error_p95": float(absolute.quantile(0.95)),
        "absolute_error_p99": float(absolute.quantile(0.99)),
        "absolute_error_max": float(absolute.max()),
    }


def shift_loss(
    prediction: torch.Tensor, target: torch.Tensor, training: dict
) -> torch.Tensor:
    kind = str(training.get("loss", "mae")).lower()
    if kind == "mae":
        return torch.nn.functional.l1_loss(prediction, target)
    if kind == "huber":
        return torch.nn.functional.huber_loss(
            prediction,
            target,
            delta=float(training.get("huber_delta_ppm", 1.0)),
        )
    raise ValueError(f"unsupported Stage delta loss {kind!r}")


@torch.no_grad()
def evaluate(
    model: NucleusShiftModel,
    loader: DataLoader,
    device: torch.device,
    prediction_path: Path | None = None,
) -> dict:
    model.eval()
    rows = []
    started = time.time()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        mask = output["target_mask"]
        target = batch["target_shift"][mask]
        prediction = output["prediction_ppm"]
        graph_ids = batch["target_graph_index"][mask]
        atom_indices = batch["target_atom_index_conformer"][mask]
        for y, pred, graph_id, atom_index in zip(
            target.cpu(), prediction.cpu(), graph_ids.cpu(), atom_indices.cpu()
        ):
            gid = int(graph_id)
            rows.append(
                {
                    "molecule_uid": batch["molecule_uids"][gid],
                    "nucleus": model.nucleus,
                    "conformer_uid": batch["conformer_uids"][gid],
                    "atom_index_conformer": int(atom_index),
                    "target": float(y),
                    "prediction": float(pred),
                    "heavy_atom_count": int(batch["heavy_atom_count"][gid]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"evaluation produced no {model.nucleus} rows")
    if frame.duplicated(["molecule_uid", "atom_index_conformer", "nucleus"]).any():
        raise RuntimeError("evaluation produced duplicate atom keys")
    if prediction_path is not None:
        frame.to_parquet(prediction_path, index=False)
    result = metric_block(frame)
    result["molecules_per_second"] = frame.molecule_uid.nunique() / max(time.time() - started, 1e-9)
    return result


@torch.no_grad()
def invariance_check(model: NucleusShiftModel, batch: dict, device: torch.device) -> float:
    model.eval()
    batch = move_batch(batch, device)
    reference = model(batch)["prediction_ppm"]
    transformed = dict(batch)
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    transformed["coordinates"] = batch["coordinates"] @ q + torch.tensor(
        [2.0, -3.0, 1.0], device=device
    )
    transformed["radius_unit_vector"] = batch["radius_unit_vector"] @ q
    prediction = model(transformed)["prediction_ppm"]
    return float((reference - prediction).abs().max())


def main() -> None:
    config = yaml.safe_load(parse_args().config.read_text(encoding="utf-8"))
    device = torch.device(config.get("runtime", {}).get("device", "cuda:0"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; set runtime.device to cpu")
        torch.cuda.set_device(device)
    seed = int(config["seed"])
    seed_everything(seed)

    nucleus = str(config["nucleus"])
    nucleus_code = NucleusShiftModel.NUCLEUS_TO_CODE[nucleus]
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for child in ["config", "logs", "checkpoints", "evaluation", "reports", "provenance"]:
        (output / child).mkdir(parents=True, exist_ok=False)
    (output / "config/resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    cache = Path(config["data"]["cache_dir"])
    train = filtered_dataset(
        cache / "train.pt", nucleus_code, int(config["data"].get("train_limit", 0))
    )
    validation = filtered_dataset(
        cache / "validation.pt", nucleus_code, int(config["data"].get("validation_limit", 0))
    )
    test = None
    run_final_test = bool(
        config["training"].get("run_final_test", config["mode"] == "full")
    )
    if run_final_test:
        test = filtered_dataset(cache / "test.pt", nucleus_code)
    statistics = target_statistics(train, nucleus_code)

    model_config = dict(config["model"])
    model = NucleusShiftModel(
        nucleus=nucleus,
        target_mean=statistics["mean"],
        target_std=statistics["std"],
        **model_config,
    )
    initialization = {"kind": config["initialization"]["kind"]}
    if initialization["kind"] == "topology":
        initialization.update(model.load_topology_initialization(config["initialization"]["checkpoint"]))
    elif initialization["kind"] != "random":
        raise ValueError(f"unsupported initialization {initialization['kind']}")
    model = model.to(device)

    collate = partial(collate_graphs, cutoff=float(model_config["cutoff"]))
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate,
        "pin_memory": True,
    }
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, shuffle=True, generator=train_generator, **loader_options
    )
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    test_loader = DataLoader(test, shuffle=False, **loader_options) if test is not None else None

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    provenance = {
        "schema": "stage_delta_dynamic_edge_nucleus_specific_v1",
        "nucleus": nucleus,
        "cache": str(cache.resolve()),
        "train_molecules": len(train),
        "validation_molecules": len(validation),
        "test_molecules": len(test) if test is not None else 0,
        "target_statistics": statistics,
        "initialization": initialization,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "loss": f"atom_level_raw_ppm_{str(config['training'].get('loss', 'mae')).lower()}",
        "sample_weighting": "none",
        "checkpoint_selection": f"validation_{nucleus}_mae",
        "test_evaluation_policy": "once_after_loading_best_validation_checkpoint" if run_final_test else "disabled",
    }
    (output / "provenance/data_model_and_initialization.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    sample = next(iter(validation_loader))
    invariance = invariance_check(model, sample, device)
    if invariance > float(config["acceptance"]["invariance_tolerance"]):
        raise RuntimeError(f"invariance check failed: {invariance}")
    (output / "reports/invariance.json").write_text(
        json.dumps({"max_abs_shift_difference_ppm": invariance}, indent=2), encoding="utf-8"
    )

    training = config["training"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["peak_lr"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    total_steps = max_epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * float(training["warmup_ratio"])))
    minimum_ratio = float(training["minimum_lr"]) / float(training["peak_lr"])

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(training["mixed_precision"]) and device.type == "cuda")

    initial_validation = evaluate(model, validation_loader, device)
    append_jsonl(
        output / "logs/validation_epochs.jsonl",
        {"epoch": 0, "phase": "initial", "selection_metric": "mae", **initial_validation},
    )
    best = float("inf")
    best_significant = float("inf")
    best_epoch = 0
    bad_epochs = 0
    global_step = 0
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        absolute_error_sum = 0.0
        atom_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(training["mixed_precision"]) and device.type == "cuda"):
                output_batch = model(batch)
                target = batch["target_shift"][output_batch["target_mask"]]
                loss = shift_loss(output_batch["prediction_ppm"], target, training)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            count = int(target.numel())
            absolute_error_sum += float(loss.detach()) * count
            atom_count += count
            if global_step == 1 or global_step % int(training["log_interval_steps"]) == 0:
                append_jsonl(
                    output / "logs/train_steps.jsonl",
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train_mae_ppm": float(loss.detach()),
                        "atoms": count,
                        "lr": optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(gradient_norm),
                    },
                )

        train_mae = absolute_error_sum / max(atom_count, 1)
        validation_metrics = evaluate(model, validation_loader, device)
        selection = float(validation_metrics["mae"])
        if not math.isfinite(selection):
            raise FloatingPointError("Non-finite validation MAE")
        improved = selection < best
        if improved:
            best, best_epoch = selection, epoch
        if selection < best_significant - float(training["early_stopping_min_delta"]):
            best_significant = selection
            bad_epochs = 0
        else:
            bad_epochs += 1

        payload = {
            "schema": "stage_delta_dynamic_edge_nucleus_specific_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "best_validation_mae": best,
            "target_statistics": statistics,
            "config": config,
        }
        torch.save(payload, output / "checkpoints/last.pt")
        if improved:
            torch.save(payload, output / "checkpoints/best_validation.pt")
        append_jsonl(
            output / "logs/train_epochs.jsonl",
            {
                "epoch": epoch,
                "train_mae_ppm": train_mae,
                "atoms": atom_count,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - started,
            },
        )
        append_jsonl(
            output / "logs/validation_epochs.jsonl",
            {
                "epoch": epoch,
                "phase": "trained",
                "selection_metric": "mae",
                "best_validation_mae": best,
                "bad_epochs": bad_epochs,
                "lr": optimizer.param_groups[0]["lr"],
                **validation_metrics,
            },
        )
        print(
            json.dumps(
                {
                    "nucleus": nucleus,
                    "epoch": epoch,
                    "train_mae": train_mae,
                    "validation_mae": selection,
                    "validation_rmse": validation_metrics["rmse"],
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            ),
            flush=True,
        )
        if bad_epochs >= int(training["early_stopping_patience_epochs"]):
            break

    best_payload = torch.load(output / "checkpoints/best_validation.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    selected_validation = evaluate(
        model, validation_loader, device, output / "evaluation/validation_predictions.parquet"
    )
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            model, test_loader, device, output / "evaluation/test_predictions.parquet"
        )
    summary = {
        "status": "completed",
        "schema": "stage_delta_dynamic_edge_nucleus_specific_v1",
        "mode": config["mode"],
        "nucleus": nucleus,
        "loss": f"atom_level_raw_ppm_{str(training.get('loss', 'mae')).lower()}",
        "initialization": initialization,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "optimizer_steps": global_step,
        "early_stopped": epoch < max_epochs,
        "target_statistics": statistics,
        "initial_validation": initial_validation,
        "selected_validation": selected_validation,
        "test": test_metrics,
        "test_evaluation_count": 1 if test_metrics is not None else 0,
        "invariance_max_abs_difference_ppm": invariance,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "wall_seconds": time.time() - started,
    }
    (output / "reports/training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from .constants import COMPONENT_NAMES, TYPE_NAMES


def _metric_block(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
        "mean_signed_error": float(np.mean(error)),
        "absolute_error_p50": float(np.quantile(np.abs(error), 0.50)),
        "absolute_error_p90": float(np.quantile(np.abs(error), 0.90)),
        "absolute_error_p95": float(np.quantile(np.abs(error), 0.95)),
        "absolute_error_p99": float(np.quantile(np.abs(error), 0.99)),
        "absolute_error_max": float(np.max(np.abs(error))),
    }


def compute_metrics(
    target: np.ndarray, prediction: np.ndarray, coupling_type: np.ndarray
) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    coupling_type = np.asarray(coupling_type, dtype=np.int64)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 5:
        raise ValueError("target and prediction must both have shape [N, 5]")
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("finite, nonempty targets and predictions required")
    if coupling_type.shape != (len(target),) or not np.isin(coupling_type, range(8)).all():
        raise ValueError("coupling types must align and be integers from 0 to 7")
    result = {"components": {}, "per_type": {}}
    for component_id, component_name in enumerate(COMPONENT_NAMES):
        result["components"][component_name] = _metric_block(
            target[:, component_id], prediction[:, component_id]
        )
    for type_id, type_name in enumerate(TYPE_NAMES):
        selected = coupling_type == type_id
        if not selected.any():
            raise ValueError(f"evaluation has no {type_name} rows")
        result["per_type"][type_name] = {}
        for component_id, component_name in enumerate(COMPONENT_NAMES):
            block = _metric_block(
                target[selected, component_id],
                prediction[selected, component_id],
            )
            if component_name == "total":
                block["log_mae"] = float(np.log(max(block["mae"], 1.0e-12)))
            result["per_type"][type_name][component_name] = block
    total_types = [result["per_type"][name]["total"] for name in TYPE_NAMES]
    result["total_micro"] = result["components"]["total"]
    result["total_macro"] = {
        "mae": float(np.mean([value["mae"] for value in total_types])),
        "rmse": float(np.mean([value["rmse"] for value in total_types])),
        "r2": (float(np.mean([v["r2"] for v in total_types]))
               if all(v["r2"] is not None for v in total_types) else None),
        "mean_log_mae": float(np.mean([value["log_mae"] for value in total_types])),
    }
    predicted_residual = prediction[:, 0] - prediction[:, 1:].sum(axis=1)
    target_residual = target[:, 0] - target[:, 1:].sum(axis=1)
    result["physical_consistency"] = {
        "prediction_component_sum_mean_abs_residual_hz": float(
            np.mean(np.abs(predicted_residual))
        ),
        "prediction_component_sum_max_abs_residual_hz": float(
            np.max(np.abs(predicted_residual))
        ),
        "target_component_sum_mean_abs_residual_hz": float(
            np.mean(np.abs(target_residual))
        ),
        "target_component_sum_max_abs_residual_hz": float(
            np.max(np.abs(target_residual))
        ),
    }
    return result


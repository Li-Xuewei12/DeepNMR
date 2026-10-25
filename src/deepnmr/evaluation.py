"""Assigned, peak-set, and Gaussian matching metrics; no target fitting."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("A finite one-dimensional array is required")
    return values


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def assigned_metrics(reference, prediction):
    y, p = _vector(reference), _vector(prediction)
    if y.shape != p.shape or not y.size:
        raise ValueError("Equal, nonempty reference/prediction arrays required")
    error = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    return dict(n=int(y.size), MAE=float(np.abs(error).mean()),
                RMSE=float(np.sqrt(np.mean(error ** 2))),
                R2=float(1 - np.sum(error ** 2) / denominator) if denominator else None,
                P95=float(np.quantile(np.abs(error), .95)))


def cluster_positions(values, tolerance):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    groups = []
    for value in sorted(_vector(values)):
        if groups and abs(value - np.mean(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return np.array([np.mean(group) for group in groups])


def peak_errors(reference, prediction, penalty):
    """Match by raw absolute error, then cap errors and penalize unmatched peaks."""
    _positive(penalty, "penalty")
    y, p = _vector(reference), _vector(prediction)
    cost = np.abs(y[:, None] - p[None, :])
    i, j = linear_sum_assignment(cost)
    errors = cost[i, j]
    return dict(matched_pairs=len(errors), absolute_error_sum=float(errors.sum()),
                squared_error_sum=float(np.sum(errors ** 2)),
                set_error=float((np.minimum(errors, penalty).sum()
                                 + penalty * abs(len(y) - len(p))) / max(len(y), len(p), 1)))


def gaussian_score(reference, prediction, sigma):
    _positive(sigma, "sigma")
    y, p = _vector(reference), _vector(prediction)
    if not len(y) or not len(p):
        return float(len(y) == len(p))
    similarity = np.exp(-(y[:, None] - p[None, :]) ** 2 / (2 * sigma ** 2))
    i, j = linear_sum_assignment(-similarity)
    return float(similarity[i, j].sum() / np.sqrt(len(y) * len(p)))

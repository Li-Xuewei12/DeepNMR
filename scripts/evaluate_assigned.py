"""Evaluate shift predictions or all five coupling components in a parquet file."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from deepnmr.evaluation import assigned_metrics
from stage_j.constants import COMPONENT_NAMES, TYPE_TO_ID
from stage_j.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--stage", choices=["delta", "overlay", "j"], default="delta")
    parser.add_argument("--reference", default="target_ppm")
    parser.add_argument("--prediction", default="prediction_ppm")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    if args.stage == "j":
        kinds = frame.coupling_type.map(TYPE_TO_ID)
        if kinds.isna().any():
            raise ValueError("Unknown coupling type")
        result = compute_metrics(
            frame[[f"target_{n}_hz" for n in COMPONENT_NAMES]].to_numpy(),
            frame[[f"prediction_{n}_hz" for n in COMPONENT_NAMES]].to_numpy(), kinds.to_numpy(),
        )
    else:
        result = assigned_metrics(frame[args.reference], frame[args.prediction])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

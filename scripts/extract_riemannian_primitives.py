from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motion_primitives.paths import PROJECT_ROOT

from riemannian import extract_reach_grasp_primitives
from motion_primitives.randg import RandG_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Riemannian primitives from ReachGrasp repetitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data selection and output behavior.
    parser.add_argument("--subject", default="sub-01", help="ReachGrasp subject identifier.")
    parser.add_argument("--task", default="FroRea", help="ReachGrasp action identifier.")
    parser.add_argument("--dataset-path", default=RandG_dir, help="ReachGrasp dataset root.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "results/ReachGrasp/riemannian/extract_riemannian_primitives",
                        help="Root directory for extracted repetition artifacts.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute repetitions that already have completed artifacts.")

    # Smoothing and segmentation.
    parser.add_argument("--window-length", type=int, default=31,
                        help="Odd Savitzky-Golay window length in samples.")
    parser.add_argument("--polyorder", type=int, default=2, help="Savitzky-Golay polynomial order.")
    parser.add_argument("--delta-theta", type=float, default=float(np.pi / 3),
                        help="Radian change in velocity direction required for a boundary.")
    parser.add_argument("--angle-confirm-samples", type=int, default=5,
                        help="Consecutive samples required to confirm an angular boundary.")
    parser.add_argument("--min-segment-samples", type=int, default=10,
                        help="Minimum samples retained in a primitive segment.")
    parser.add_argument("--speed-floor-fraction", type=float, default=0.05,
                        help="Fraction of peak speed below which direction estimates are ignored.")
    parser.add_argument("--min-tiny-segment-duration", type=float, default=0.15,
                        help="Minimum duration in seconds before a short segment is merged.")

    # Geodesic reconstruction accuracy and cost.
    parser.add_argument("--path-samples", type=int, default=80,
                        help="Phase samples stored along each reconstructed primitive path.")
    parser.add_argument("--log-map-max-nfev", type=int, default=60,
                        help="Maximum endpoint-shooting residual evaluations per primitive.")
    parser.add_argument("--endpoint-tol", type=float, default=2e-3,
                        help="Maximum accepted geodesic endpoint error in joint-space radians.")
    parser.add_argument("--geodesic-max-step", type=float, default=0.03,
                        help="Maximum integration step in normalized path phase.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = vars(args).copy()
    for key in ["subject", "task", "dataset_path", "output_dir", "overwrite"]:
        params.pop(key)
    result = extract_reach_grasp_primitives(
        args.subject, args.task, args.dataset_path, args.output_dir,
        params, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    if result["failed_repetitions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

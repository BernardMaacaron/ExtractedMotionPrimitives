from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nni


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_riemPrim_params.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NNI bridge for riemPrim parameter evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subjects", nargs="+", default=["sub-01"],
                        help="ReachGrasp subjects used to score each NNI trial.")
    parser.add_argument("--tasks", nargs="+", default=["FroRea"],
                        help="ReachGrasp tasks used to score each NNI trial.")
    parser.add_argument("--dataset-path", type=str, required=True, help="ReachGrasp dataset root.")
    parser.add_argument("--python", default=sys.executable,
                        help="Evaluator interpreter; defaults to the active NNI trial environment.")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "results/ReachGrasp/riemannian/evaluate_riemPrim_params/nni",
                        help="Root for evaluator artifacts from all NNI trials.")
    parser.add_argument("--max-repetitions", type=int,
                        help="Maximum repetitions per subject/task; omit to evaluate all repetitions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = nni.get_next_parameter()

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(params, handle)
        params_path = Path(handle.name)

    env = os.environ.copy()
    python_path = [str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    cmd = [
        args.python,
        str(EVALUATOR),
        "--subjects", *args.subjects,
        "--tasks", *args.tasks,
        "--dataset-path", args.dataset_path,
        "--output-root", str(args.output_root),
        "--params-json", str(params_path),
    ]
    if args.max_repetitions is not None:
        cmd.extend(["--max-repetitions", str(args.max_repetitions)])

    try:
        print(f"Evaluator: {args.python}; subjects={args.subjects}; tasks={args.tasks}; "
              f"repetition limit={args.max_repetitions}; parameters={params}", flush=True)
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE)
        last_line = ""
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.strip():
                last_line = line
        returncode = process.wait()
    finally:
        params_path.unlink(missing_ok=True)

    if returncode != 0:
        raise RuntimeError(f"Evaluator failed with exit code {returncode}")
    metrics = json.loads(last_line)

    nni.report_final_result(metrics)


if __name__ == "__main__":
    main()

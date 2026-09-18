from __future__ import annotations

import argparse
import json
import inspect
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from riemannian import extract_primitives, extract_reach_grasp_primitives
from motion_primitives.paths import PROJECT_ROOT
from motion_primitives.randg import RandG_dir, get_available_subjects, get_available_tasks, get_repetition_count


@dataclass(frozen=True, slots=True)
class ExtractionJob:
    subject: str
    task: str
    expected_repetitions: int


def build_jobs(dataset_path: str, subjects: list[str] | None, tasks: list[str] | None) -> list[ExtractionJob]:
    selected_subjects = subjects or sorted(get_available_subjects(dataset_path=dataset_path))
    jobs = []
    for subject in selected_subjects:
        selected_tasks = tasks or sorted(get_available_tasks(
            subject, modality="motion", dataset_path=dataset_path))
        for task in selected_tasks:
            jobs.append(ExtractionJob(
                subject, task, get_repetition_count(
                    subject, task, dataset_path=dataset_path, extract_movements=True)))
    return jobs


def summarize_job(output_root: Path, job: ExtractionJob) -> dict[str, Any]:
    completed = []
    failures = []
    for repetition in range(job.expected_repetitions):
        status_path = output_root / job.subject / job.task / f"rep-{repetition:02d}" / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status["status"] == "completed":
            completed.append(status)
        else:
            failures.append(status)

    n_primitives = sum(status["n_primitives"] for status in completed)
    weighted_error = sum(
        status["mean_endpoint_error"] * status["n_primitives"] for status in completed)
    return {
        **asdict(job),
        "completed_repetitions": len(completed),
        "pending_repetitions": job.expected_repetitions - len(completed),
        "failed_repetitions": failures,
        "n_primitives": n_primitives,
        "mean_endpoint_error": weighted_error / n_primitives if n_primitives else None,
        "runtime_seconds": sum(status["runtime_seconds"] for status in completed),
        "status": "completed" if len(completed) == job.expected_repetitions else "pending",
    }


def extract_job(payload: tuple[ExtractionJob, str, Path, dict[str, Any], bool]) -> dict[str, Any]:
    job, dataset_path, output_root, params, overwrite = payload
    return extract_reach_grasp_primitives(
        job.subject, job.task, dataset_path, output_root, params, overwrite=overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Riemannian primitives for the full ReachGrasp dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Inputs and output location.
    parser.add_argument("--params-file", type=Path,
                        default=PROJECT_ROOT / "results/ReachGrasp/riemannian/plot_riemPrim_nni_results/elbow_parameters.json",
                        help="JSON file whose 'parameters' object controls primitive extraction.")
    parser.add_argument("--dataset-path", default=RandG_dir, help="ReachGrasp dataset root.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/ReachGrasp/riemannian/full",
                        help="Root for per-subject/task/repetition artifacts and the summary manifest.")
    parser.add_argument("--subjects", nargs="*", help="Subjects to extract; omit to discover all subjects.")
    parser.add_argument("--tasks", nargs="*", help="Tasks to extract; omit to discover each subject's tasks.")

    # Parallelism and run scope.
    parser.add_argument("--jobs", type=int, default=4, help="Number of subject/task worker processes.")
    parser.add_argument("--worker-cpu-threads", type=int, default=1,
                        help="CPU thread limit applied inside each worker process.")
    parser.add_argument("--limit-jobs", type=int,
                        help="Run only the first N selected subject/task jobs; useful for a smoke run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute repetitions even when completed artifacts already exist.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved workload and parameters without extracting primitives.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.params_file.read_text())
    defaults = {key: parameter.default for key, parameter in inspect.signature(extract_primitives).parameters.items()
                if parameter.default is not inspect.Parameter.empty and key != "show_progress"}
    params = defaults | selection["parameters"]
    jobs = build_jobs(args.dataset_path, args.subjects, args.tasks)
    if args.limit_jobs is not None:
        jobs = jobs[:args.limit_jobs]

    summaries = {
        (job.subject, job.task): summarize_job(args.output_dir, job) for job in jobs}
    pending = [
        job for job in jobs
        if args.overwrite or summaries[(job.subject, job.task)]["pending_repetitions"]]

    print(f"Selected subject/actions: {len(jobs)}")
    print(f"Selected repetitions: {sum(job.expected_repetitions for job in jobs)}")
    print(f"Already completed: {sum(item['completed_repetitions'] for item in summaries.values())}")
    print(f"Pending subject/actions: {len(pending)}")
    print(f"Workers: {args.jobs}")
    print(f"Parameters: {args.params_file}")
    print(f"Effective extraction settings: {json.dumps(params, sort_keys=True)}")
    print(f"Selection provenance: {json.dumps({key: value for key, value in selection.items() if key != 'parameters'})}")
    print(f"Output: {args.output_dir}; overwrite={args.overwrite}; subset limit={args.limit_jobs}")
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    subject_action_failures = []

    def write_manifest() -> None:
        subject_actions = sorted(summaries.values(), key=lambda item: (item["subject"], item["task"]))
        manifest = {
            "params_file": str(args.params_file),
            "parameters": params,
            "selection": selection,
            "dataset_path": args.dataset_path,
            "extract_movements": True,
            "total_subject_actions": len(jobs),
            "total_repetitions": sum(job.expected_repetitions for job in jobs),
            "completed_repetitions": sum(item["completed_repetitions"] for item in subject_actions),
            "pending_repetitions": sum(item["pending_repetitions"] for item in subject_actions),
            "failed_repetitions": [
                failure for item in subject_actions for failure in item["failed_repetitions"]],
            "failed_subject_actions": subject_action_failures,
            "subject_actions": subject_actions,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    write_manifest()
    os.environ["OMP_NUM_THREADS"] = str(args.worker_cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.worker_cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.worker_cpu_threads)
    payloads = [(job, args.dataset_path, args.output_dir, params, args.overwrite) for job in pending]

    if args.jobs == 1:
        def completed_results():
            for payload in payloads:
                try:
                    yield payload[0], extract_job(payload), None
                except Exception as exc:
                    yield payload[0], None, exc

        progress = tqdm(
            completed_results(), total=len(payloads),
            desc="Riemannian extraction", unit="subject-action")
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn"))
        future_jobs = {executor.submit(extract_job, payload): payload[0] for payload in payloads}

        def completed_results():
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                try:
                    yield job, future.result(), None
                except Exception as exc:
                    yield job, None, exc

        progress = tqdm(
            completed_results(), total=len(future_jobs),
            desc="Riemannian extraction", unit="subject-action")

    try:
        for job, result, error in progress:
            if error is None:
                summaries[(job.subject, job.task)] = summarize_job(args.output_dir, job)
                label = "FAILED" if result["failed_repetitions"] else "OK"
                tqdm.write(
                    f"[{label}] {job.subject} {job.task}: {job.expected_repetitions} movements, "
                    f"{result['newly_completed_repetitions']} newly completed, "
                    f"{result['resumed_repetitions']} resumed, {result['n_primitives']} primitives")
            else:
                failure = {**asdict(job), "status": "failed", "error": str(error)}
                subject_action_failures.append(failure)
                tqdm.write(f"[FAILED] {job.subject} {job.task}: {error}")
            write_manifest()
    finally:
        if args.jobs != 1:
            executor.shutdown()

    failed_repetitions = sum(len(item["failed_repetitions"]) for item in summaries.values())
    print(f"Completed repetitions: {sum(item['completed_repetitions'] for item in summaries.values())}")
    print(f"Failed repetitions: {failed_repetitions}")
    print(f"Manifest: {manifest_path}")
    if failed_repetitions or subject_action_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

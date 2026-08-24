"""Command line entry point for the dynamic, simulation-based evaluation.

Two commands:

``rescore``
    Recompute the structural metrics for the stored benchmark results and write
    them back into the existing result files, leaving every other metric alone.

``validate``
    Evidence that the metric measures what it claims to: it scores the dataset
    answers against deliberately good and bad completions and checks that the
    reference optimiser is never beaten by the answer it is meant to bound.

Examples::

    python -m sphyr.evaluate_dynamic rescore --workers 8
    python -m sphyr.evaluate_dynamic validate --subject full_easy --samples 20
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from sphyr.metrics.structural import (
    STRUCTURAL_METRIC_KEYS,
    get_structural_metrics,
)
from sphyr.metrics.utils import extract_grid_from_text, get_gravity_from_folder

RESULTS_SUFFIX = "_results.json"
AGGREGATED_SUFFIX = "_aggregated_results.json"

DATASET_DIR = os.path.join(
    "src", "sphyr", "dataset_creation", "huggingface", "2D", "datasets"
)


def find_result_files(results_root):
    """Group stored result files by subject.

    Grouping by subject rather than by model matters for speed: every model
    completed the same 100 samples, so processing a subject in one worker lets
    all of its models share one set of cached optimiser runs.
    """
    grouped = defaultdict(list)

    for model_dir in sorted(os.listdir(results_root)):
        model_path = os.path.join(results_root, model_dir)
        if not os.path.isdir(model_path) or model_dir == "plots":
            continue

        for file_name in sorted(os.listdir(model_path)):
            if not file_name.endswith(RESULTS_SUFFIX):
                continue
            if file_name.endswith(AGGREGATED_SUFFIX):
                continue

            subject = file_name[: -len(RESULTS_SUFFIX)]
            grouped[subject].append(
                {
                    "path": os.path.join(model_path, file_name),
                    "model": model_dir,
                    # Rotation lives in the directory name, and a rotated
                    # sample is loaded along its own rotated gravity.
                    "gravity_dir": get_gravity_from_folder(model_dir),
                }
            )

    return grouped


def rescore_record(record, gravity_dir, include_own_budget_reference):
    """Add the structural metrics to one stored result record."""
    input_grid = extract_grid_from_text(record["prompt"])
    output_grid = extract_grid_from_text(record["completion"])
    gt_grid = extract_grid_from_text(record["ground_truth"])

    metrics = get_structural_metrics(
        output_grid=output_grid,
        gt_grid=gt_grid,
        input_grid=input_grid,
        gravity_dir=gravity_dir,
        include_own_budget_reference=include_own_budget_reference,
    )

    record.update(metrics.as_dict())
    return record


def update_aggregates(path, records):
    """Refresh only the structural totals of the aggregate file next to it."""
    aggregate_path = path.replace(RESULTS_SUFFIX, AGGREGATED_SUFFIX)

    aggregates = {}
    if os.path.exists(aggregate_path):
        with open(aggregate_path, "r") as handle:
            aggregates = json.load(handle)

    for key in STRUCTURAL_METRIC_KEYS:
        aggregates[f"total_{key}"] = sum(
            float(record.get(key) or 0.0) for record in records
        )
    aggregates["evaluated_samples"] = len(records)

    with open(aggregate_path, "w") as handle:
        json.dump(aggregates, handle, indent=2)


def rescore_subject(task):
    """Rescore every stored file of one subject (runs in a worker process)."""
    subject, files, include_own_budget_reference, dry_run = task
    summaries = []

    for entry in files:
        with open(entry["path"], "r") as handle:
            records = json.load(handle)

        for record in records:
            rescore_record(
                record, entry["gravity_dir"], include_own_budget_reference
            )

        if not dry_run:
            with open(entry["path"], "w") as handle:
                json.dump(records, handle, indent=2)
            update_aggregates(entry["path"], records)

        summaries.append(
            {
                "model": entry["model"],
                "subject": subject,
                "samples": len(records),
                **{
                    key: (
                        sum(float(record.get(key) or 0.0) for record in records)
                        / max(len(records), 1)
                    )
                    for key in STRUCTURAL_METRIC_KEYS
                },
            }
        )

    return summaries


def print_leaderboard(summaries):
    """Mean topology score per model, averaged over its subjects."""
    per_model = defaultdict(list)
    for summary in summaries:
        per_model[summary["model"]].append(summary)

    print("\nDynamic evaluation - mean over subjects")
    header = (
        f"{'model':46s} {'topo':>6s} {'struct':>7s} {'material':>9s} "
        f"{'vs GT':>7s} {'carrying':>9s}"
    )
    print(header)
    print("-" * len(header))

    def mean(rows, key):
        return sum(row[key] for row in rows) / max(len(rows), 1)

    for model, rows in sorted(
        per_model.items(), key=lambda item: -mean(item[1], "topology_score")
    ):
        print(
            f"{model:46s} "
            f"{mean(rows, 'topology_score'):6.3f} "
            f"{mean(rows, 'structural_efficiency'):7.3f} "
            f"{mean(rows, 'material_efficiency'):9.3f} "
            f"{mean(rows, 'compliance_efficiency_vs_ground_truth'):7.3f} "
            f"{mean(rows, 'load_carrying'):9.3f}"
        )


def run_rescore(args):
    grouped = find_result_files(args.results_root)
    if not grouped:
        print(f"No result files found under {args.results_root}/")
        return

    file_count = sum(len(files) for files in grouped.values())
    print(
        f"Rescoring {file_count} result files across {len(grouped)} subjects "
        f"with {args.workers} worker(s)."
    )
    if args.dry_run:
        print("Dry run: metrics are computed but nothing is written back.")

    tasks = [
        (subject, files, args.design_efficiency, args.dry_run)
        for subject, files in sorted(grouped.items())
    ]

    started = time.time()
    summaries = []

    if args.workers <= 1:
        for task in tasks:
            summaries.extend(rescore_subject(task))
            print(f"  done: {task[0]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(rescore_subject, task): task[0] for task in tasks}
            for future in as_completed(futures):
                summaries.extend(future.result())
                print(f"  done: {futures[future]}")

    print(f"\nFinished in {time.time() - started:.0f}s")
    print_leaderboard(summaries)


def load_dataset_file(subject):
    path = os.path.join(DATASET_DIR, f"{subject}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"Dataset file not found: {path}")
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_probe_completions(ground_truth, rng, hard):
    """Completions whose relative quality is known in advance.

    A metric worth trusting has to rank these in the obvious order, and has to
    put the two degenerate answers - fill everything, fill nothing - below the
    dataset answer.
    """

    def perturb(count):
        grid = [row[:] for row in ground_truth]
        cells = [
            (row, col)
            for row in range(len(grid))
            for col in range(len(grid[0]))
            if grid[row][col] not in ("L", "S")
        ]
        for row, col in rng.sample(cells, min(count, len(cells))):
            if hard:
                grid[row][col] = f"{rng.choice([0.0, 0.2, 0.5, 0.7, 1.0]):.1f}"
            else:
                grid[row][col] = "0" if grid[row][col] == "1" else "1"
        return grid

    def fill(value):
        return [
            [value if cell not in ("L", "S") else cell for cell in row]
            for row in ground_truth
        ]

    return {
        "dataset answer": ground_truth,
        "2 cells flipped": perturb(2),
        "10 cells flipped": perturb(10),
        "filled solid": fill("1.0" if hard else "1"),
        "left empty": fill("0.0" if hard else "0"),
    }


def run_validate(args):
    records = load_dataset_file(args.subject)[: args.samples]
    hard = args.subject.endswith("hard")

    rows = defaultdict(list)
    optimiser_losses = 0
    started = time.time()

    for index, record in enumerate(records):
        ground_truth = record["ground_truth"]
        rng = random.Random(index)
        for name, completion in build_probe_completions(
            ground_truth, rng, hard
        ).items():
            metrics = get_structural_metrics(
                output_grid=completion,
                gt_grid=ground_truth,
                input_grid=record["input_grid"],
                include_own_budget_reference=args.design_efficiency,
            )
            rows[name].append(metrics)
            if name == "dataset answer" and metrics.structural_efficiency_raw > 1.0005:
                optimiser_losses += 1

    print(
        f"\n{args.subject}: {len(records)} samples, "
        f"{time.time() - started:.0f}s\n"
    )
    header = (
        f"{'completion':18s} {'topo score':>10s} {'stiffness':>10s} "
        f"{'material':>9s} {'vs GT':>7s} {'volume':>7s} {'carrying':>9s}"
    )
    print(header)
    print("-" * len(header))

    def mean(name, key):
        values = [getattr(metrics, key) for metrics in rows[name]]
        return sum(float(value or 0.0) for value in values) / max(len(values), 1)

    for name in rows:
        print(
            f"{name:18s} "
            f"{mean(name, 'topology_score'):10.3f} "
            f"{mean(name, 'structural_efficiency'):10.3f} "
            f"{mean(name, 'material_efficiency'):9.3f} "
            f"{mean(name, 'compliance_efficiency_vs_ground_truth'):7.3f} "
            f"{mean(name, 'volume_fraction'):7.3f} "
            f"{mean(name, 'load_carrying'):9.3f}"
        )

    print(
        f"\nReference optimiser beaten by the dataset answer on "
        f"{optimiser_losses}/{len(records)} samples "
        f"(a non-zero count means the yardstick is too weak)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    rescore = subparsers.add_parser(
        "rescore", help="add structural metrics to stored benchmark results"
    )
    rescore.add_argument("--results-root", default="results")
    rescore.add_argument("--workers", type=int, default=1)
    rescore.add_argument(
        "--design-efficiency",
        action="store_true",
        help="also re-optimise at each completion's own material budget (slow)",
    )
    rescore.add_argument("--dry-run", action="store_true")
    rescore.set_defaults(func=run_rescore)

    validate = subparsers.add_parser(
        "validate", help="check that the metric ranks known-good and known-bad answers"
    )
    validate.add_argument("--subject", default="full_easy")
    validate.add_argument("--samples", type=int, default=20)
    validate.add_argument("--design-efficiency", action="store_true")
    validate.set_defaults(func=run_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

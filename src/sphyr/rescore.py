"""Rescore stored answers under the masked-only rule, without model calls.

    python -m sphyr.rescore --workers 8
    python -m sphyr.rescore --runs perplexity-sonar --limit-records 5 --dry-run

Original files are backed up before atomic replacement. A dry run writes no
results. Existing optimization references are reused for unchanged tasks unless
--rebuild-references is supplied; all prediction metrics are recomputed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

from sphyr.evaluate_dynamic import rescore_record
from sphyr.metrics.utils import SCORING_POLICY, get_gravity_from_folder
from sphyr.scoring import Result, aggregate_results
from sphyr.tasks import RANDOM_SEED, build_sample, load_subject


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".rescore-tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@lru_cache(maxsize=8)
def feedback_samples(subject, version):
    records = load_subject(subject, source="local", version=version)
    random.Random(RANDOM_SEED).shuffle(records)
    return records


def rescore_episode(episode, subject, gravity, rebuild_references=False):
    """Recover legacy feedback tasks using the original deterministic shuffle.

    Subsequent migrations use the stored prompt/answer instead. Trajectories
    remain the originally sampled trajectories: no feedback is regenerated and
    no new revisions are requested from a model.
    """
    if "prompt" not in episode or "ground_truth" not in episode:
        version = episode["dataset_version"]
        records = feedback_samples(subject, version)
        index = episode["sample_index"]
        sample = build_sample(records, records[index], index, subject)
        episode["prompt"] = sample.prompt
        episode["ground_truth"] = sample.ground_truth
        episode["task_recovery"] = f"bundled {version} {subject}; seed {RANDOM_SEED}"
    episode.setdefault("feedback_generated_with_scoring_policy", "legacy-unrestored")
    for attempt in episode["trajectory"]:
        attempt.setdefault("subject", subject)
        attempt.setdefault("prompt", episode["prompt"])
        attempt.setdefault("ground_truth", episode["ground_truth"])
        if episode.get("load_direction") is not None:
            attempt.setdefault("load_direction", episode["load_direction"])
            attempt.setdefault("load_direction_source", episode.get("load_direction_source", "explicit"))
        rescore_record(attempt, gravity, reuse_reference=not rebuild_references)
    episode["scoring_policy"] = SCORING_POLICY
    return episode


def process_file(task):
    path_string, root_string, backup_string, dry_run, limit, rebuild, own_budget = task
    path, root = Path(path_string), Path(root_string)
    started = time.perf_counter()
    original_bytes = path.read_bytes()
    records = json.loads(original_bytes)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of results: {path}")
    subject = path.name.removesuffix("_results.json")
    gravity = get_gravity_from_folder(path.parent.name)
    selected = records[:limit] if limit else records
    changed = predictions = 0
    max_score_delta = 0.0
    for record in selected:
        previous = record.get("topology_score", 0.0)
        if "trajectory" in record:
            rescore_episode(record, subject, gravity, rebuild)
            predictions += len(record["trajectory"])
            changed += sum(bool(a["visible_cell_changes"]) for a in record["trajectory"])
        else:
            rescore_record(record, gravity, own_budget, reuse_reference=not rebuild)
            predictions += 1
            changed += bool(record["visible_cell_changes"])
            max_score_delta = max(max_score_delta, abs(record["topology_score"] - previous))
    if not dry_run:
        backup = Path(backup_string) / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        # Do not overwrite a backup when resuming a migration.
        if not backup.exists():
            backup.write_bytes(original_bytes)
        aggregate_path = path.with_name(subject + "_aggregated_results.json")
        aggregate_backup = backup.with_name(aggregate_path.name)
        if aggregate_path.exists() and not aggregate_backup.exists():
            shutil.copy2(aggregate_path, aggregate_backup)
        write_json(path, records)
        if not records or "trajectory" not in records[0]:
            aggregate = aggregate_results([Result.from_dict(r) for r in records])
            aggregate["evaluated_samples"] = len(records)
            write_json(aggregate_path, aggregate)
    return {
        "file": str(path.relative_to(root)),
        "records": len(selected),
        "predictions": predictions,
        "visible_cell_changes_in_predictions": changed,
        "max_topology_score_delta": max_score_delta,
        "seconds": round(time.perf_counter() - started, 3),
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "rescored_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if not dry_run else None,
    }


def rescore_stored_results(results_root="results", workers=1, dry_run=False,
                           runs=None, limit_records=None, backup_root=None,
                           rebuild_references=False, own_budget=False):
    if limit_records and not dry_run:
        raise ValueError("--limit-records is a timing option and requires --dry-run")
    root = Path(results_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(backup_root).resolve() if backup_root else root.parent / ".rescore" / stamp
    if backup == root or root in backup.parents:
        raise ValueError("Backup directory must be outside the scanned results directory")
    paths = sorted(p for p in root.glob("*/*_results.json")
                   if not p.name.endswith("_aggregated_results.json")
                   and p.parent.name != "plots"
                   and (not runs or p.parent.name in runs))
    if runs:
        missing = set(runs) - {p.parent.name for p in paths}
        if missing:
            raise ValueError(f"No result files found for: {sorted(missing)}")
    print(f"{'Timing' if dry_run else 'Rescoring'} {len(paths)} files, {workers} workers", flush=True)
    if not dry_run:
        backup.mkdir(parents=True, exist_ok=True)
        print(f"Original files: {backup}", flush=True)
    # Worker processes must not each start a full BLAS thread pool.
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    tasks = [(str(p), str(root), str(backup), dry_run, limit_records,
              rebuild_references, own_budget) for p in paths]
    started = time.perf_counter()
    summaries = []
    if workers == 1:
        pending = map(process_file, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        pending = (future.result() for future in as_completed(
            pool.submit(process_file, task) for task in tasks))
    try:
        for summary in pending:
            summaries.append(summary)
            print(f"  {len(summaries)}/{len(tasks)} {summary['file']}: "
                  f"{summary['predictions']} predictions, {summary['seconds']:.1f}s", flush=True)
            if not dry_run:
                write_json(backup / "manifest.json", {
                    "scoring_policy": SCORING_POLICY,
                    "reference_policy": "rebuilt" if rebuild_references else "reuse unchanged stored references",
                    "files": summaries,
                    "complete": len(summaries) == len(tasks),
                })
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"Finished: {sum(s['predictions'] for s in summaries)} predictions in "
          f"{time.perf_counter() - started:.1f}s", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--backup-root")
    parser.add_argument("--rebuild-references", action="store_true")
    args = parser.parse_args()
    rescore_stored_results(**vars(args))


if __name__ == "__main__":
    main()

"""Recover and verify v2 force metadata into separately versioned datasets.

Replay the generator's candidate RNG (seed 0), match boundary layouts in
acceptance order, then re-solve every accepted problem. Write nothing until
both binary and rounded-density answers match the bundled records exactly.
Historical tasks/results are never edited. New model replies are required.
"""

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sphyr.dataset_creation.generate_v2 import Sample, _answer_grid, _draw_problem, _solve
from sphyr.physics.boundary_conditions import get_problem_from_grid
from sphyr.tasks import DATASET_VERSIONS, SUBJECTS


def _layout(problem):
    return (problem.nely, problem.nelx, frozenset(problem.load_cells),
            frozenset(problem.support_cells))


def recover_candidates(records, seed=0, max_draws=18000):
    """Find candidate forces in the original generator's acceptance order."""
    rng = random.Random(seed)
    index, candidates = 0, []
    for draw in range(max_draws):
        problem = _draw_problem(rng, grid=len(records[0]["ground_truth"]))
        if problem is None:
            continue
        target = get_problem_from_grid(records[index]["ground_truth"])
        if _layout(problem) != _layout(target):
            continue
        candidates.append((draw, problem))
        index += 1
        if index == len(records):
            return candidates
    raise ValueError(f"Could not recover record {index}; seed/order may differ. No dataset written.")


def verify_candidate(task):
    problem, easy, hard = task
    solved = _solve(problem)
    if solved is None:
        raise ValueError(f"Recovered candidate has no usable solution: {problem}")
    solid, densities = solved
    import numpy as np
    sample = Sample(problem, tuple(map(tuple, solid)),
                    tuple(map(tuple, np.round(densities, 1))))
    if _answer_grid(sample, "easy") != easy or _answer_grid(sample, "hard") != hard:
        raise ValueError(f"Re-solved answers differ; force recovery not proven: {problem}")
    return True


def recover_version(version, workers=8, seed=0):
    source = DATASET_VERSIONS[version]
    destination = DATASET_VERSIONS[version + "-force"]
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    records = {
        subject: [json.loads(line) for line in (source / f"{subject}.jsonl").read_text().splitlines()]
        for subject in SUBJECTS
    }
    easy, hard = records["full_easy"], records["full_hard"]
    candidates = recover_candidates(easy, seed=seed)
    tasks = [(problem, a["ground_truth"], b["ground_truth"])
             for (_, problem), a, b in zip(candidates, easy, hard, strict=True)]
    print(f"{version}: verifying {len(tasks)} recovered load cases with {workers} workers", flush=True)
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(verify_candidate, tasks), 1):
            if i % 25 == 0:
                print(f"  verified {i}/{len(tasks)}", flush=True)
    directions = {a["index"]: list(problem.load_direction)
                  for (_, problem), a in zip(candidates, easy, strict=True)}
    # All variants must carry exactly the same answer for their sample ID.
    answers = {difficulty: {r["index"]: r["ground_truth"] for r in records[f"full_{difficulty}"]}
               for difficulty in ("easy", "hard")}
    encoded, source_hashes = {}, {}
    for subject, rows in records.items():
        difficulty = subject.rsplit("_", 1)[-1]
        if len(rows) != len(directions) or len({r["index"] for r in rows}) != len(directions):
            raise ValueError(f"Inconsistent sample IDs in {subject}")
        for row in rows:
            if row["ground_truth"] != answers[difficulty][row["index"]]:
                raise ValueError(f"Inconsistent answer in {subject}, ID {row['index']}")
        encoded[subject] = "".join(json.dumps({**row, "load_direction": directions[row["index"]]}) + "\n"
                                   for row in rows)
        source_hashes[f"{subject}.jsonl"] = hashlib.sha256((source / f"{subject}.jsonl").read_bytes()).hexdigest()
    destination.mkdir()
    for subject, content in encoded.items():
        (destination / f"{subject}.jsonl").write_text(content, encoding="utf-8")
    manifest = {
        "dataset_version": version + "-force", "source_version": version,
        "seed": seed, "verified_samples": len(tasks),
        "verification": "Exact match after re-solving both binary and rounded continuous answers",
        "candidate_draws": [draw for draw, _ in candidates], "source_sha256": source_hashes,
        "requires_new_model_evaluations": True,
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Written {destination}; original files unchanged", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", choices=["v2", "v2-20"], default=["v2", "v2-20"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for version in args.versions:
        recover_version(version, args.workers, args.seed)


if __name__ == "__main__":
    main()

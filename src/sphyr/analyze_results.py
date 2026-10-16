"""Analyses of the stored results that the paper's claims rest on.

Four commands.  The first three call no model and re-run no simulation --
they read the metrics already stored with every completion:

``alternatives``
    How often a completion is as good as the reference design while being a
    different structure: near-optimal on both stiffness and material, yet not
    the reference answer, and how far from it.  A verifier that rewarded
    resemblance to the optimiser's output would score none of these well.

``headline``
    The statistics the abstract and Section 4.1 quote over the paper's v1
    runs: how many valid completions carry their load while differing from
    the reference, how often cell-graph connectivity and compliance disagree,
    and how often a completion beats the reference design by spending more
    material than it.

``invalid``
    Why unreadable completions are unreadable: no grid at all, a grid that
    stops short, extra rows, or the right rows at the wrong widths.  These are
    different failures -- running out of output budget is not miscounting a
    row -- and only the last is a pure serialisation slip.

``formats``
    The same samples answered as a grid and as a sparse list of solid cells,
    side by side, to separate structural quality from grid serialisation.

``optima``
    The model-free version of ``alternatives``: drive the optimiser into
    different local optima of the same problem, score each through the
    verifier, and count the structurally different designs it rates as highly
    as the reference.  Answers whether the score *can* reward a different
    design, independently of whether any model produced one.

Examples::

    python -m sphyr.analyze_results alternatives
    python -m sphyr.analyze_results invalid --runs google_gemini-3.8-flash_v2-20
    python -m sphyr.analyze_results formats --runs anthropic_claude-opus-5_v2-20
    python -m sphyr.analyze_results optima --dataset-version v2 --samples 30
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from sphyr.metrics.structural import get_structural_metrics
from sphyr.metrics.utils import extract_grid_from_text, is_solid
from sphyr.physics.boundary_conditions import (
    get_densities_from_grid,
    get_free_cells_mask,
    get_problem_from_grid,
)
from sphyr.physics.simp import binarize_densities, optimize_topology, polish_design
from sphyr.tasks import grid_to_str, load_subject, str_to_grid

RESULTS_ROOT = Path("results")

# The runs behind the paper's v1 statistics over "the valid completions in our
# results": every v1 model run, ablations included, as stored when the
# statistic was first computed.  Later additions (the real DeepSeek-R1 column,
# the local model, the v2 runs) are deliberately not in it, so the number keeps
# meaning what the paper says it means.
PAPER_V1_RUNS = [
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-20250514",
    "claude-opus-4-20250514_3_rotations",
    "claude-opus-4-20250514_few_shot_1",
    "claude-opus-4-20250514_few_shot_3",
    "deepseek-reasoner",
    "deepseek-reasoner_3_rotations",
    "gemini-1.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-05-06_3_rotations",
    "gemini-2.5-pro-preview-05-06_physics_enhanced_prompt",
    "gemini-2.5-pro-preview-05-06_physics_neutral_prompt",
    "gpt-3.5-turbo-0125",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-2025-04-14_3_rotations",
    "gpt-4o-2024-08-06",
    "perplexity-sonar",
    "perplexity-sonar-reasoning",
    "perplexity-sonar_3_rotations",
]

# Within this fraction of the reference design on both factors counts as
# "as good as" it.  Tight enough that a structure meeting it is not a weaker
# design that happens to stand.
NEAR_OPTIMAL = 0.95

# Solid pieces connect through corners -- two elements meeting at a corner
# share a node in the finite element model -- while voids connect only through
# edges, the standard dual pair that keeps holes and pieces well defined.
SOLID_CONNECTIVITY = np.ones((3, 3), dtype=int)
VOID_CONNECTIVITY = ndimage.generate_binary_structure(2, 1)


def records(runs, results_root=RESULTS_ROOT):
    """Every stored completion of the named runs, as (run, record) pairs."""
    for run in runs:
        for path in sorted((Path(results_root) / run).glob("*_results.json")):
            if path.name.endswith("_aggregated_results.json"):
                continue
            with open(path) as handle:
                for record in json.load(handle):
                    yield run, record


def solid_mask(grid):
    """Which cells carry material, densities thresholded at one half."""
    return np.array(
        [
            [
                cell in ("L", "S") or (is_solid(cell) and float(cell) >= 0.5)
                for cell in row
            ]
            for row in grid
        ],
        dtype=bool,
    )


def topology(mask):
    """(solid pieces, enclosed holes) of a binary structure."""
    _, pieces = ndimage.label(mask, structure=SOLID_CONNECTIVITY)
    voids, void_count = ndimage.label(~mask, structure=VOID_CONNECTIVITY)
    border = set(np.unique(np.concatenate(
        [voids[0], voids[-1], voids[:, 0], voids[:, -1]]
    ))) - {0}
    return pieces, void_count - len(border)


def run_alternatives(args):
    valid = carrying_different = near_optimal = distinct_topology = 0
    stiffer_than_answer = 0
    distances = []

    for _, record in records(args.runs or PAPER_V1_RUNS, args.results_root):
        if not record.get("valid_output_grid"):
            continue
        valid += 1
        if not record.get("load_carrying") or record.get("exact_match"):
            continue
        carrying_different += 1

        structural = record.get("structural_efficiency") or 0.0
        material = record.get("material_efficiency") or 0.0
        if structural < NEAR_OPTIMAL or material < NEAR_OPTIMAL:
            continue
        near_optimal += 1

        output = solid_mask(extract_grid_from_text(record["completion"]))
        reference = solid_mask(str_to_grid(record["ground_truth"]))
        masked = np.array(
            [[cell == "V" for cell in row]
             for row in extract_grid_from_text(record["prompt"])]
        )
        distances.append(int((output != reference)[masked].sum()))
        if topology(output) != topology(reference):
            distinct_topology += 1

        compliance = record.get("compliance")
        answer = record.get("ground_truth_compliance")
        if compliance is not None and answer is not None and compliance <= answer:
            stiffer_than_answer += 1

    distances = np.array(distances) if distances else np.zeros(1)
    summary = {
        "valid": valid,
        "load_carrying_and_different": carrying_different,
        "near_optimal_and_different": near_optimal,
        "near_optimal_distinct_topology": distinct_topology,
        "near_optimal_at_least_as_stiff_as_dataset_answer": stiffer_than_answer,
        "masked_cells_differing_median": float(np.median(distances)),
        "masked_cells_differing_mean": float(distances.mean()),
        "masked_cells_differing_at_least_5": int((distances >= 5).sum()),
    }
    print(json.dumps(summary, indent=2))


# Material use within this fraction of the reference counts as equal.
EQUAL_MATERIAL = 0.01


def run_headline(args):
    valid = carrying_different = 0
    connected_not_carrying = carrying_not_connected = 0
    beats = {"more": [0, 0], "equal": [0, 0], "less": [0, 0]}

    for _, record in records(args.runs or PAPER_V1_RUNS, args.results_root):
        if not record.get("valid_output_grid"):
            continue
        valid += 1
        carrying = bool(record.get("load_carrying"))
        connected = bool(record.get("load_support_connected"))
        carrying_different += carrying and not record.get("exact_match")
        connected_not_carrying += connected and not carrying
        carrying_not_connected += carrying and not connected

        compliance = record.get("compliance")
        optimum = record.get("optimal_compliance")
        volume = record.get("volume_ratio")
        if not carrying or not compliance or not optimum or volume is None:
            continue
        if volume > 1 + EQUAL_MATERIAL:
            group = beats["more"]
        elif volume < 1 - EQUAL_MATERIAL:
            group = beats["less"]
        else:
            group = beats["equal"]
        group[0] += 1
        group[1] += compliance < optimum

    print(
        json.dumps(
            {
                "valid": valid,
                "load_carrying_and_different": carrying_different,
                "load_carrying_and_different_pct": 100.0 * carrying_different / max(valid, 1),
                "connectivity_disagrees_with_compliance": connected_not_carrying
                + carrying_not_connected,
                "connected_but_not_carrying": connected_not_carrying,
                "carrying_but_not_connected": carrying_not_connected,
                "beats_reference_design_pct": {
                    material: 100.0 * hits / max(total, 1)
                    for material, (total, hits) in beats.items()
                },
            },
            indent=2,
        )
    )


def failure_kind(record):
    """Classify one unreadable completion by what is wrong with its grid."""
    expected = str_to_grid(record["ground_truth"])
    grid = extract_grid_from_text(record.get("completion") or "")
    if not grid:
        return "no grid"
    if len(grid) < len(expected):
        return "stops short"
    if len(grid) > len(expected):
        return "extra rows"
    if any(len(row) != len(want) for row, want in zip(grid, expected)):
        return "wrong row width"
    return "bad cell values"


def run_invalid(args):
    for run in args.runs:
        kinds, reasons, total = Counter(), Counter(), 0
        for _, record in records([run], args.results_root):
            total += 1
            if record.get("valid_output_grid"):
                continue
            kinds[failure_kind(record)] += 1
            if record.get("finish_reason"):
                reasons[record["finish_reason"]] += 1
        invalid = sum(kinds.values())
        print(f"{run}: {invalid}/{total} unreadable")
        for kind, count in kinds.most_common():
            print(f"  {kind:16s} {count}")
        if reasons:
            print(f"  finish reasons: {dict(reasons)}")


def mean(rows, key, condition=None):
    rows = [r for r in rows if condition is None or condition(r)]
    if not rows:
        return float("nan")
    return 100.0 * sum(float(r.get(key) or 0.0) for r in rows) / len(rows)


def sparse_counterpart(run):
    """The sparse-format run that replays a grid run's samples."""
    if "_v2" in run:
        stem, version = run.split("_v2", 1)
        return f"{stem}_sparse_v2{version}"
    return f"{run}_sparse"


def by_task(run, results_root):
    """A run's records keyed by subject and masked grid, so formats pair up."""
    keyed = {}
    for _, record in records([run], results_root):
        task = repr(extract_grid_from_text(record["prompt"]))
        keyed.setdefault(record["subject"], {})[task] = record
    return keyed


def run_formats(args):
    """Grid against sparse answers, per subject, on the samples both share.

    A run is named ``GRID`` (paired with its ``_sparse`` counterpart) or
    ``GRID:SPARSE`` to pair two directories explicitly, e.g. a rerun of the
    grid format against the sparse run.
    """
    ok = lambda r: r.get("valid_output_grid")
    if not args.latex:
        print(
            f"{'grid run':44s} {'subject':22s} {'n':>4s} "
            f"{'valid':>6s} {'TS':>6s} {'TS|ok':>6s} {'LC':>6s} | "
            f"{'valid':>6s} {'TS':>6s} {'TS|ok':>6s} {'LC':>6s}  (grid | sparse)"
        )
    for spec in args.runs:
        grid_run, _, sparse_run = spec.partition(":")
        grid_rows = by_task(grid_run, args.results_root)
        sparse_rows = by_task(sparse_run or sparse_counterpart(grid_run), args.results_root)

        for subject in sorted(set(grid_rows) & set(sparse_rows)):
            shared = sorted(set(grid_rows[subject]) & set(sparse_rows[subject]))
            if not shared:
                continue
            grid = [grid_rows[subject][key] for key in shared]
            sparse = [sparse_rows[subject][key] for key in shared]
            values = [
                mean(grid, "valid_output_grid"),
                mean(grid, "topology_score"),
                mean(grid, "topology_score", ok),
                mean(grid, "load_carrying"),
                mean(sparse, "valid_output_grid"),
                mean(sparse, "topology_score"),
                mean(sparse, "topology_score", ok),
                mean(sparse, "load_carrying"),
            ]
            if args.latex:
                cells = " & ".join(f"{v:.1f}" for v in values)
                subject_tex = subject.replace("_", "\\_")
                print(f"{grid_run} & \\texttt{{{subject_tex}}} & {len(shared)} & {cells} \\\\")
            else:
                cells = " ".join(f"{v:6.1f}" for v in values[:4])
                cells += " | " + " ".join(f"{v:6.1f}" for v in values[4:])
                print(f"{grid_run:44s} {subject:22s} {len(shared):4d} {cells}")


# How the optimiser is pushed off the reference: each run forbids a random
# share of the reference answer's solid masked cells and re-optimises the rest
# under the same budget.  Random starts alone are not enough -- the discrete
# polish pulls every start back into the same basin.  Each forbidden share is
# drawn several times, plus one unconstrained run.
OPTIMA_FORBIDDEN_SHARES = (0.1, 0.2, 0.3)
OPTIMA_DRAWS = 4

# Two designs closer than this many cells are counted as one.
OPTIMA_MIN_SEPARATION = 3


def alternative_optima(input_grid, gt_grid, rng):
    """Distinct binary designs for one sample, each with its verifier score.

    Every design spends exactly the reference answer's material in the masked
    cells, so material efficiency is 1 and the score is stiffness alone.  The
    forbidden cells only steer the optimiser; the score is always computed on
    the sample as posed, with every masked cell available.
    """
    problem = get_problem_from_grid(gt_grid)
    shape = (problem.nely, problem.nelx)
    structural = problem.structural_cells_mask()
    free = get_free_cells_mask(input_grid, shape) & ~structural
    gt_densities, _ = get_densities_from_grid(gt_grid)
    gt_densities[structural] = 1.0
    target = int(round(gt_densities[free].sum()))
    volume = float(gt_densities.mean())
    answer_solid = np.argwhere(free & (gt_densities > 0.5))

    designs = []
    for share in (0.0,) + tuple(
        share for share in OPTIMA_FORBIDDEN_SHARES for _ in range(OPTIMA_DRAWS)
    ):
        forbidden = np.zeros(shape, dtype=bool)
        count = int(round(share * len(answer_solid)))
        if count:
            picks = answer_solid[rng.choice(len(answer_solid), count, replace=False)]
            forbidden[tuple(picks.T)] = True
        allowed = free & ~forbidden

        # Masked cells outside the design space are pinned empty, so they
        # neither carry load nor eat into the budget.
        pinned = gt_densities.copy()
        pinned[free] = 0.0

        relaxed = optimize_topology(
            problem=problem,
            volume_fraction=volume,
            free_cells=allowed,
            fixed_densities=pinned,
        ).densities
        binary = binarize_densities(
            relaxed, free_cells=allowed, target_solid_cells=target
        )
        polished, _ = polish_design(problem=problem, densities=binary, free_cells=allowed)
        designs.append(np.asarray(polished).reshape(shape) > 0.5)

    reference = gt_densities > 0.5
    kept = []
    for design in designs:
        if any((design != other)[free].sum() < OPTIMA_MIN_SEPARATION for other, _ in kept):
            continue
        grid = [
            [
                ("1" if design[r, c] else "0") if free[r, c] else gt_grid[r][c]
                for c in range(shape[1])
            ]
            for r in range(shape[0])
        ]
        score = get_structural_metrics(grid, gt_grid, input_grid).topology_score
        kept.append((design, score))

    return [
        {
            "score": score,
            "cells_from_answer": int((design != reference)[free].sum()),
            "distinct_topology": topology(design) != topology(reference),
        }
        for design, score in kept
    ]


def run_optima(args):
    rng = np.random.default_rng(0)
    rows = load_subject(args.subject, version=args.dataset_version)[: args.samples]
    per_sample = []

    for index, row in enumerate(rows):
        input_grid = str_to_grid(grid_to_str(row["input_grid"]))
        gt_grid = str_to_grid(grid_to_str(row["ground_truth"]))
        answer = get_structural_metrics(gt_grid, gt_grid, input_grid).topology_score
        designs = alternative_optima(input_grid, gt_grid, rng)
        # "As good as the reference" means scoring at least what the dataset's
        # own answer scores, within the near-optimal margin.
        good = [
            d
            for d in designs
            if d["score"] >= NEAR_OPTIMAL * answer and d["cells_from_answer"] >= OPTIMA_MIN_SEPARATION
        ]
        strict = [d for d in good if d["score"] >= answer]
        per_sample.append(
            {
                "sample": index,
                "answer_score": answer,
                "at_least_answer_and_different": len(strict),
                "at_least_answer_distinct_topology": sum(
                    d["distinct_topology"] for d in strict
                ),
                "distinct_designs": len(designs),
                "as_good_and_different": len(good),
                "as_good_distinct_topology": sum(d["distinct_topology"] for d in good),
                "max_cells_from_answer": max((d["cells_from_answer"] for d in good), default=0),
                "masked_cells": int(sum(cell == "V" for line in input_grid for cell in line)),
            }
        )
        print(json.dumps(per_sample[-1]))

    with_alternative = sum(1 for s in per_sample if s["as_good_and_different"])
    with_topology = sum(1 for s in per_sample if s["as_good_distinct_topology"])
    strict_alternative = sum(1 for s in per_sample if s["at_least_answer_and_different"])
    strict_topology = sum(
        1 for s in per_sample if s["at_least_answer_distinct_topology"]
    )
    print(
        json.dumps(
            {
                "subject": args.subject,
                "dataset_version": args.dataset_version,
                "samples": len(per_sample),
                "samples_with_an_as_good_different_design": with_alternative,
                "samples_with_an_as_good_different_topology": with_topology,
                "samples_with_a_different_design_scoring_at_least_the_answer": strict_alternative,
                "samples_with_a_different_topology_scoring_at_least_the_answer": strict_topology,
                "median_max_cells_from_answer": float(
                    np.median([s["max_cells_from_answer"] for s in per_sample if s["as_good_and_different"]] or [0])
                ),
            },
            indent=2,
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    alternatives = subparsers.add_parser(
        "alternatives", help="near-optimal completions that differ from the reference"
    )
    alternatives.add_argument(
        "--runs", nargs="+", help="result directories; default: the paper's v1 runs"
    )
    alternatives.set_defaults(func=run_alternatives)

    headline = subparsers.add_parser(
        "headline", help="the v1 statistics quoted in the abstract and Section 4.1"
    )
    headline.add_argument(
        "--runs", nargs="+", help="result directories; default: the paper's v1 runs"
    )
    headline.set_defaults(func=run_headline)

    invalid = subparsers.add_parser("invalid", help="why unreadable completions fail")
    invalid.add_argument("--runs", nargs="+", required=True)
    invalid.set_defaults(func=run_invalid)

    formats = subparsers.add_parser(
        "formats", help="grid against sparse answers on shared samples"
    )
    formats.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help=(
            "grid-answer runs, each paired with its _sparse counterpart, or "
            "GRID:SPARSE to pair two result directories explicitly"
        ),
    )
    formats.add_argument("--latex", action="store_true", help="print table rows")
    formats.set_defaults(func=run_formats)

    optima = subparsers.add_parser(
        "optima", help="different designs the verifier rates as highly as the reference"
    )
    optima.add_argument("--subject", default="full_easy")
    optima.add_argument("--dataset-version", default="v2")
    optima.add_argument("--samples", type=int, default=30)
    optima.set_defaults(func=run_optima)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Generate SPhyR-v2: a benchmark whose samples are not near-twins.

SPhyR-v1 was produced by sweeping load and support positions one cell at a
time along fixed edges.  That makes the samples a dense grid over a small
space: of the 1,295 distinct boundary conditions in a subject, 1,294 have
another one a single cell away, and adjacent load cases usually share an
optimal structure.  A benchmark built that way rewards interpolating between
neighbouring pre-solved problems, which is not the ability it means to measure.

v2 changes the sampling rather than the task.  Load and support groups are
placed on any edge in any combination, the load direction varies, and a
candidate is rejected unless its boundary conditions *and* its solution are far
from every sample already accepted.  The task formats, grid size, tokens and
record schema are unchanged, so the existing prompts, metrics and environment
work against v2 without modification.

Ground truth comes from the repository's own SIMP optimizer rather than from an
external solver, so the dataset can be regenerated from source.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sphyr.physics.boundary_conditions import MASK_TOKEN, StructuralProblem
from sphyr.physics.simp import binarize_densities, optimize_reference_design

# Default grid. Larger grids enlarge the boundary-condition space, which is
# what lets more samples coexist at a given separation; solve cost grows
# sub-linearly in cells (0.9s at 10x10, 2.6s at 20x20, 4.5s at 30x30).
GRID = 10

# Minimum separation between accepted samples.  ``BC_DISTANCE`` counts cells
# where the load/support layouts differ and ``SOLUTION_DISTANCE`` counts cells
# where the solved structures differ; a candidate must clear both.  v1 would
# score 1 and often 0 on these.
BC_DISTANCE = 6
SOLUTION_DISTANCE = 8

# Material budget.  Loads and supports are solid by definition and can occupy a
# tenth of the grid on their own, so the budget has to leave the optimiser room
# to build; v1's 0.1 target is not reachable under this parameterisation.
VOLUME_FRACTION = 0.3

# Load directions, as (row, col) components: straight along each axis and the
# four diagonals, so no single orientation dominates the dataset.
LOAD_DIRECTIONS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)

EDGES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class Sample:
    problem: StructuralProblem
    structure: tuple
    densities: tuple = ()


def _edge_cells(edge, n=GRID):
    if edge == "top":
        return [(0, c) for c in range(n)]
    if edge == "bottom":
        return [(n - 1, c) for c in range(n)]
    if edge == "left":
        return [(r, 0) for r in range(n)]
    return [(r, n - 1) for r in range(n)]


def _draw_group(rng, edge, size, grid=GRID):
    """A contiguous run of ``size`` cells somewhere along ``edge``."""
    cells = _edge_cells(edge, grid)
    start = rng.randrange(0, len(cells) - size + 1)
    return cells[start : start + size]


def _draw_problem(rng, grid=GRID):
    """Sample one boundary condition from the full space v2 draws over."""
    load_edge = rng.choice(EDGES)
    support_edge = rng.choice([e for e in EDGES if e != load_edge])

    load_cells = _draw_group(rng, load_edge, rng.randint(1, 3), grid)

    # Supports may be one run or two disjoint runs, on the same edge or on the
    # remaining edges, which v1 never varied.
    support_cells = _draw_group(rng, support_edge, rng.randint(2, 5), grid)
    if rng.random() < 0.35:
        second_edge = rng.choice([e for e in EDGES if e != load_edge])
        support_cells = support_cells + _draw_group(
            rng, second_edge, rng.randint(2, 4), grid
        )

    support_cells = [c for c in dict.fromkeys(support_cells) if c not in load_cells]
    if len(support_cells) < 2:
        return None

    return StructuralProblem(
        nely=grid,
        nelx=grid,
        load_cells=tuple(load_cells),
        support_cells=tuple(support_cells),
        load_direction=rng.choice(LOAD_DIRECTIONS),
    )


def _bc_signature(problem):
    grid = problem.nely
    """The load/support layout as a flat grid of tokens, for distance checks."""
    layout = np.full((grid, grid), ".", dtype=object)
    for r, c in problem.load_cells:
        layout[r, c] = "L"
    for r, c in problem.support_cells:
        layout[r, c] = "S"
    return layout.ravel()


def _distance(a, b):
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def _solve(problem):
    """Optimal structure for a problem, or None if it does not stand.

    Returns the binarised design used by the ``easy`` subjects and the
    continuous density field used by the ``hard`` ones, so both difficulties
    describe the same sample.
    """
    result = optimize_reference_design(problem, volume_fraction=VOLUME_FRACTION)
    densities = np.asarray(result.densities)

    solid = binarize_densities(
        densities,
        target_solid_cells=int(round(VOLUME_FRACTION * problem.element_count)),
    )
    solid = np.asarray(solid)
    if solid.ndim == 1:
        solid = solid.reshape(problem.nely, problem.nelx)

    # A problem whose optimum is degenerate is not a useful task.
    if not np.isfinite(result.compliance) or result.compliance > 1e6:
        return None
    return (
        (solid > 0.5).astype(int),
        densities.reshape(problem.nely, problem.nelx),
    )


def _to_grid(problem, structure):
    """Render a solved sample as the token grid the benchmark uses."""
    grid = [[str(int(v)) for v in row] for row in structure]
    for r, c in problem.load_cells:
        grid[r][c] = "L"
    for r, c in problem.support_cells:
        grid[r][c] = "S"
    return grid


def generate_samples(
    count, seed=0, max_attempts_per_sample=60, verbose=True, grid=GRID
):
    """Draw ``count`` boundary conditions that are far apart and solve them."""
    rng = random.Random(seed)
    accepted, signatures, structures = [], [], []
    attempts = 0

    while len(accepted) < count and attempts < count * max_attempts_per_sample:
        attempts += 1
        problem = _draw_problem(rng, grid)
        if problem is None:
            continue

        signature = _bc_signature(problem)
        if any(_distance(signature, s) < BC_DISTANCE for s in signatures):
            continue

        solved = _solve(problem)
        if solved is None:
            continue
        structure, densities = solved

        flat = structure.ravel()
        if any(_distance(flat, s) < SOLUTION_DISTANCE for s in structures):
            continue

        accepted.append(
            Sample(
                problem=problem,
                structure=tuple(map(tuple, structure)),
                densities=tuple(map(tuple, np.round(densities, 1))),
            )
        )
        signatures.append(signature)
        structures.append(flat)

        if verbose and len(accepted) % 25 == 0:
            print(f"  accepted {len(accepted)}/{count} (attempts {attempts})")

    return accepted


def _structural_positions(problem):
    return set(problem.load_cells) | set(problem.support_cells)


def _mask_cells(grid, positions):
    masked = [list(row) for row in grid]
    for r, c in positions:
        masked[r][c] = MASK_TOKEN
    return masked


def _free_positions(problem):
    grid = problem.nely
    """Cells a mask may cover: everything that is not a load or a support."""
    structural = _structural_positions(problem)
    return [
        (r, c) for r in range(grid) for c in range(grid) if (r, c) not in structural
    ]


def _scaled_count(kind, count, grid):
    """Scale a variant's mask size so the masked *fraction* is grid-independent.

    The variant names describe the 10x10 benchmark: ``5_random_cell`` masks 5%
    of the grid and ``3_random_row`` masks 30% of it.  Carrying the literal
    counts to a larger grid would quietly make every task easier --- 10 cells
    is a tenth of a 10x10 grid but a fortieth of a 20x20 one --- so counts
    scale with area for cells and with side length for rows and columns, and
    the ladder means the same thing at either size.
    """
    factor = grid / 10.0
    if kind == "cell":
        return max(1, int(round(count * factor * factor)))
    return max(1, int(round(count * factor)))


def _mask_for(variant, problem, rng):
    """Positions masked for one task variant.

    ``full`` masks every non-structural cell.  v1 masked rows 1-8 instead,
    which left the whole load and support rows visible; here the only cells a
    model is shown are the load and support cells themselves, so a full
    structure prediction really is made from the boundary conditions alone.
    """
    free = _free_positions(problem)
    if variant == "full":
        return free

    kind, count = variant
    count = _scaled_count(kind, count, problem.nely)

    if kind == "cell":
        return rng.sample(free, min(count, len(free)))

    axis = 0 if kind == "row" else 1
    lines = sorted({p[axis] for p in free})
    chosen = rng.sample(lines, min(count, len(lines)))
    return [p for p in free if p[axis] in chosen]


VARIANTS = {
    "1_random_cell": ("cell", 1),
    "5_random_cell": ("cell", 5),
    "10_random_cell": ("cell", 10),
    "1_random_row": ("row", 1),
    "3_random_row": ("row", 3),
    "1_random_column": ("column", 1),
    "3_random_column": ("column", 3),
    "full": "full",
}


def _answer_grid(sample, difficulty):
    problem = sample.problem
    if difficulty == "easy":
        values = [[str(int(v)) for v in row] for row in sample.structure]
    else:
        values = [[f"{float(v):.1f}" for v in row] for row in sample.densities]
    for r, c in problem.load_cells:
        values[r][c] = "L"
    for r, c in problem.support_cells:
        values[r][c] = "S"
    return values


def write_subjects(samples, out_dir, seed=0):
    """Write the sixteen subject files, in the v1 record schema."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    for variant_name, variant in VARIANTS.items():
        for difficulty in ("easy", "hard"):
            rng = random.Random(f"{seed}-{variant_name}-{difficulty}")
            path = out_dir / f"{variant_name}_{difficulty}.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                for index, sample in enumerate(samples):
                    answer = _answer_grid(sample, difficulty)
                    masked = _mask_cells(
                        answer, _mask_for(variant, sample.problem, rng)
                    )
                    handle.write(
                        json.dumps(
                            {
                                "index": index,
                                "input_grid": masked,
                                "ground_truth": answer,
                            }
                        )
                        + "\n"
                    )
            written[path.name] = len(samples)
    return written

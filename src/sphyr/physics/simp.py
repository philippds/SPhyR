"""SIMP topology optimisation with an optimality-criteria update.

This is the standard density-based minimum-compliance formulation (Bendsoe &
Sigmund; the "99/88-line" family of codes) specialised to SPhyR grids:

    minimise    c(rho) = f^T u(rho)
    subject to  K(rho) u = f
                sum(rho) <= volume_fraction * n_elements
                0 <= rho <= 1

Two SPhyR-specific extensions matter for benchmarking:

* Load and support cells are passive solid - they are structure by definition
  and are never optimised away.
* Only the cells a model was asked to fill in (the masked cells of the input
  grid) are design variables.  Every other cell is pinned to the density the
  sample prescribes.  Re-optimising exactly the masked region is what makes the
  optimiser a fair reference for a masked completion rather than for a whole
  structure the model never had to invent.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix

from sphyr.physics.fea import simp_stiffness

DEFAULT_PENAL = 3.0
# The grid is only 10x10 and its members are one cell wide, so the density
# filter is deliberately narrow: a wider one washes single-cell members out and
# leaves the optimiser with designs the dataset itself beats.  1.2 is the
# smallest radius that still suppressed checkerboarding on the SPhyR samples.
DEFAULT_FILTER_RADIUS = 1.2
DEFAULT_MAX_ITERATIONS = 500
DEFAULT_TOLERANCE = 1e-3
DEFAULT_MOVE_LIMIT = 0.2

# Penalisation continuation: starting nearly convex and stiffening the
# penalisation keeps the optimiser out of the grey local minima that a cold
# start at p = 3 falls into on a mesh this coarse.
DEFAULT_PENAL_SCHEDULE = (1.0, 2.0, 3.0)
DEFAULT_MAX_POLISH_PASSES = 3
DEFAULT_MAX_POLISH_EVALUATIONS = 6000
DEFAULT_POLISH_SOURCE_LIMIT = 16
DEFAULT_POLISH_TARGET_LIMIT = 24


@dataclass
class OptimizationResult:
    """Outcome of a SIMP run."""

    densities: np.ndarray
    compliance: float
    volume_fraction: float
    iterations: int
    converged: bool


@lru_cache(maxsize=64)
def build_density_filter(nely, nelx, filter_radius):
    """Cone-shaped density filter matrix H and its row sums."""
    element_count = nely * nelx
    reach = int(np.ceil(filter_radius)) - 1

    rows = []
    cols = []
    values = []

    for row in range(nely):
        for col in range(nelx):
            element = row * nelx + col
            for neighbor_row in range(max(row - reach, 0), min(row + reach + 1, nely)):
                for neighbor_col in range(
                    max(col - reach, 0), min(col + reach + 1, nelx)
                ):
                    distance = np.hypot(row - neighbor_row, col - neighbor_col)
                    weight = filter_radius - distance
                    if weight <= 0.0:
                        continue
                    rows.append(element)
                    cols.append(neighbor_row * nelx + neighbor_col)
                    values.append(weight)

    filter_matrix = coo_matrix(
        (values, (rows, cols)), shape=(element_count, element_count)
    ).tocsr()

    return filter_matrix, np.asarray(filter_matrix.sum(axis=1)).ravel()


def _resolve_design_space(problem, free_cells, fixed_densities):
    """Split the grid into design variables and pinned cells.

    Load and support cells are forced solid and removed from the design space
    regardless of what the caller passed in.
    """
    structural = problem.structural_cells_mask()

    if free_cells is None:
        free = ~structural
    else:
        free = np.asarray(free_cells, dtype=bool).reshape(problem.nely, problem.nelx)
        free = free & ~structural

    if fixed_densities is None:
        pinned = structural.astype(float)
    else:
        pinned = np.clip(
            np.asarray(fixed_densities, dtype=float).reshape(problem.nely, problem.nelx),
            0.0,
            1.0,
        ).copy()

    pinned[structural] = 1.0

    return free, pinned


def _get_free_budget(problem, free, pinned, volume_fraction):
    """Material (in cells) left for the design variables after pinned cells."""
    free_flat = free.ravel()
    pinned_flat = pinned.ravel()
    free_count = int(free_flat.sum())
    pinned_volume = float(pinned_flat[~free_flat].sum())
    total_budget = float(volume_fraction) * problem.element_count
    free_budget = min(max(total_budget - pinned_volume, 0.0), float(free_count))
    return free_budget, free_count


def get_compliance(model, densities, penal=DEFAULT_PENAL):
    """Compliance of a density field under an already-built FEA model."""
    _, compliance = model.solve(simp_stiffness(np.ravel(densities), penal=penal))
    return compliance


def optimize_topology(
    problem,
    volume_fraction,
    free_cells=None,
    fixed_densities=None,
    penal=DEFAULT_PENAL,
    filter_radius=DEFAULT_FILTER_RADIUS,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    tolerance=DEFAULT_TOLERANCE,
    move_limit=DEFAULT_MOVE_LIMIT,
    initial_densities=None,
):
    """Minimise compliance for ``problem`` under a global volume constraint.

    ``volume_fraction`` is the budget for the whole grid, including the cells
    pinned by ``fixed_densities``; the optimiser gets whatever is left of that
    budget after the pinned material has been paid for.  ``initial_densities``
    warm-starts the design, which is what makes penalisation continuation work.
    """
    model = problem.build_model()
    free, pinned = _resolve_design_space(problem, free_cells, fixed_densities)

    element_count = problem.element_count
    free_flat = free.ravel()
    pinned_flat = pinned.ravel()

    free_budget, free_count = _get_free_budget(problem, free, pinned, volume_fraction)
    pinned_volume = float(pinned_flat[~free_flat].sum())

    if free_count == 0:
        densities = pinned_flat.reshape(problem.nely, problem.nelx)
        _, compliance = model.solve(simp_stiffness(densities.ravel(), penal=penal))
        return OptimizationResult(
            densities=densities,
            compliance=compliance,
            volume_fraction=float(densities.mean()),
            iterations=0,
            converged=True,
        )

    filter_matrix, filter_sums = build_density_filter(
        problem.nely, problem.nelx, float(filter_radius)
    )

    design = pinned_flat.copy()
    if initial_densities is None:
        design[free_flat] = free_budget / free_count
    else:
        warm_start = np.clip(
            np.asarray(initial_densities, dtype=float).ravel(), 1e-3, 1.0
        )
        design[free_flat] = warm_start[free_flat]

    physical = design.copy()

    compliance = np.inf
    iteration = 0
    converged = False

    while iteration < max_iterations:
        iteration += 1

        moduli = simp_stiffness(physical, penal=penal)
        displacements, compliance = model.solve(moduli)
        if not np.isfinite(compliance):
            break

        unit_energies = model.element_strain_energies(displacements)

        # d c / d rho for E(rho) = e_min + rho^penal * (e_0 - e_min).
        compliance_sensitivity = (
            -penal * np.maximum(physical, 1e-9) ** (penal - 1.0) * unit_energies
        )
        volume_sensitivity = np.ones(element_count)

        # Chain rule through the density filter; pinned cells do not respond.
        compliance_sensitivity = filter_matrix.T @ (
            compliance_sensitivity * free_flat / filter_sums
        )
        volume_sensitivity = filter_matrix.T @ (
            volume_sensitivity * free_flat / filter_sums
        )

        new_design, physical = _optimality_criteria_update(
            design=design,
            compliance_sensitivity=compliance_sensitivity,
            volume_sensitivity=volume_sensitivity,
            free_flat=free_flat,
            pinned_flat=pinned_flat,
            filter_matrix=filter_matrix,
            filter_sums=filter_sums,
            target_volume=pinned_volume + free_budget,
            move_limit=move_limit,
        )

        change = float(np.max(np.abs(new_design[free_flat] - design[free_flat])))
        design = new_design

        if change < tolerance:
            converged = True
            break

    _, compliance = model.solve(simp_stiffness(physical, penal=penal))

    return OptimizationResult(
        densities=physical.reshape(problem.nely, problem.nelx),
        compliance=compliance,
        volume_fraction=float(physical.mean()),
        iterations=iteration,
        converged=converged,
    )


def _optimality_criteria_update(
    design,
    compliance_sensitivity,
    volume_sensitivity,
    free_flat,
    pinned_flat,
    filter_matrix,
    filter_sums,
    target_volume,
    move_limit,
):
    """Bisection on the Lagrange multiplier of the volume constraint."""
    lower, upper = 1e-9, 1e9
    new_design = design.copy()
    physical = design.copy()

    while (upper - lower) / (lower + upper) > 1e-6:
        middle = 0.5 * (lower + upper)

        ratio = np.sqrt(
            np.maximum(-compliance_sensitivity, 0.0)
            / np.maximum(volume_sensitivity, 1e-12)
            / middle
        )
        candidate = np.clip(design * ratio, design - move_limit, design + move_limit)
        candidate = np.clip(candidate, 0.0, 1.0)

        new_design = np.where(free_flat, candidate, pinned_flat)

        physical = (filter_matrix @ new_design) / filter_sums
        physical = np.where(free_flat, physical, pinned_flat)

        if physical.sum() > target_volume:
            lower = middle
        else:
            upper = middle

    return new_design, physical


def binarize_densities(
    densities, free_cells=None, preserve_volume=True, target_solid_cells=None
):
    """Threshold a density field into a 0/1 design.

    With ``preserve_volume`` the densest cells are kept solid until the
    material budget is used up.  The budget defaults to the material the field
    itself holds, but ``target_solid_cells`` should be passed whenever the
    field was produced under a volume constraint: an optimiser run usually
    stops slightly *below* its budget, and rounding to the field instead of to
    the budget quietly throws away a whole cell of material on a grid this
    small.  Rounding is always downwards, so the result never overspends.
    """
    densities = np.asarray(densities, dtype=float)
    if free_cells is None:
        free = np.ones_like(densities, dtype=bool)
    else:
        free = np.asarray(free_cells, dtype=bool)

    binary = (densities >= 0.5).astype(float)
    binary[~free] = densities[~free]

    if not preserve_volume or not free.any():
        return binary

    if target_solid_cells is None:
        target_solid = int(np.floor(densities[free].sum() + 1e-9))
    else:
        target_solid = int(np.floor(target_solid_cells + 1e-9))
    candidates = np.sort(densities[free])[::-1]

    if target_solid <= 0:
        threshold = np.inf
    elif target_solid >= candidates.size:
        threshold = -np.inf
    else:
        threshold = candidates[target_solid - 1]

    binary[free] = (densities[free] >= threshold).astype(float)
    return binary


def polish_design(
    problem,
    densities,
    free_cells,
    penal=DEFAULT_PENAL,
    max_passes=DEFAULT_MAX_POLISH_PASSES,
    max_evaluations=DEFAULT_MAX_POLISH_EVALUATIONS,
    source_limit=DEFAULT_POLISH_SOURCE_LIMIT,
    target_limit=DEFAULT_POLISH_TARGET_LIMIT,
):
    """Discrete best-improvement search over single material swaps.

    SIMP solves a relaxed problem, so its rounded design is rarely the best
    discrete one.  Moving one cell of material at a time - always keeping the
    material budget constant - repairs the rounding on a grid this small.

    An exhaustive pass costs ``solid x void`` solves, so the neighbourhood is
    ranked by strain energy first: material is taken from the cells that carry
    the least load and offered to the empty cells that sit next to the ones
    carrying the most.  Returns the improved design and its compliance.
    """
    model = problem.build_model()
    design = np.array(densities, dtype=float).copy()
    free = np.asarray(free_cells, dtype=bool)

    best_compliance = get_compliance(model, design, penal=penal)
    evaluations = 0

    for _ in range(max_passes):
        displacements, _ = model.solve(simp_stiffness(design, penal=penal))
        energies = (
            model.element_strain_energies(displacements)
            .reshape(problem.nely, problem.nelx)
        )

        solid = _rank_cells((design > 0.5) & free, energies, source_limit, lowest=True)
        void = _rank_cells(
            (design <= 0.5) & free,
            _neighbour_energy(energies, design > 0.5),
            target_limit,
            lowest=False,
        )
        if not solid or not void:
            break

        best_swap = None
        for source in solid:
            for target in void:
                if evaluations >= max_evaluations:
                    break
                candidate = design.copy()
                candidate[source] = 0.0
                candidate[target] = 1.0
                compliance = get_compliance(model, candidate, penal=penal)
                evaluations += 1
                if compliance < best_compliance - 1e-12:
                    best_compliance = compliance
                    best_swap = (source, target)
            if evaluations >= max_evaluations:
                break

        if best_swap is None:
            break

        source, target = best_swap
        design[source] = 0.0
        design[target] = 1.0

    return design, best_compliance


def _rank_cells(mask, scores, limit, lowest):
    """Cells of ``mask`` ordered by ``scores``, truncated to ``limit``."""
    cells = list(zip(*np.where(mask)))
    if not cells:
        return cells
    cells.sort(key=lambda cell: scores[cell], reverse=not lowest)
    if limit is None or limit <= 0:
        return cells
    return cells[:limit]


def _neighbour_energy(energies, solid_mask):
    """Strain energy of the busiest solid neighbour of each cell.

    Empty cells next to highly stressed material are where extra material is
    most likely to help, so this ranks the targets of a swap.
    """
    padded = np.where(solid_mask, energies, 0.0)
    stacked = np.zeros_like(energies)
    for row_shift in (-1, 0, 1):
        for col_shift in (-1, 0, 1):
            if row_shift == 0 and col_shift == 0:
                continue
            shifted = np.roll(np.roll(padded, row_shift, axis=0), col_shift, axis=1)
            if row_shift == 1:
                shifted[0, :] = 0.0
            elif row_shift == -1:
                shifted[-1, :] = 0.0
            if col_shift == 1:
                shifted[:, 0] = 0.0
            elif col_shift == -1:
                shifted[:, -1] = 0.0
            stacked = np.maximum(stacked, shifted)
    return stacked


def optimize_reference_design(
    problem,
    volume_fraction,
    free_cells=None,
    fixed_densities=None,
    penal=DEFAULT_PENAL,
    filter_radius=DEFAULT_FILTER_RADIUS,
    penal_schedule=DEFAULT_PENAL_SCHEDULE,
    polish=True,
    seed_designs=(),
    max_iterations=DEFAULT_MAX_ITERATIONS,
    tolerance=DEFAULT_TOLERANCE,
):
    """Best design the optimiser can find at a given material budget.

    A single SIMP run is not a trustworthy yardstick on a 10x10 mesh: it has
    grey local minima, and which one it falls into depends on the starting
    point.  So several designs are generated and the stiffest one that stays
    within budget wins:

    * a cold run at every penalisation in ``penal_schedule``,
    * a warm-started continuation chain through the same schedule,
    * every design in ``seed_designs`` (typically the reference answer of the
      sample) together with a SIMP run warm-started from it,
    * a volume-preserving binarisation of each of the above,
    * a discrete swap polish of the best candidate.

    Seeding with the reference answer is what makes the yardstick meaningful:
    the optimum a completion is compared against is then, by construction, at
    least as good as the answer the dataset ships.
    """
    model = problem.build_model()
    free, pinned = _resolve_design_space(problem, free_cells, fixed_densities)
    free_budget, _ = _get_free_budget(problem, free, pinned, volume_fraction)
    budget = float(volume_fraction) * problem.element_count + 1e-9

    def run(stage_penal, initial=None):
        return optimize_topology(
            problem=problem,
            volume_fraction=volume_fraction,
            free_cells=free_cells,
            fixed_densities=fixed_densities,
            penal=stage_penal,
            filter_radius=filter_radius,
            max_iterations=max_iterations,
            tolerance=tolerance,
            initial_densities=initial,
        )

    designs = []
    last_result = None

    for stage_penal in penal_schedule:
        last_result = run(stage_penal)
        designs.append(last_result.densities)

    warm_start = None
    for stage_penal in penal_schedule:
        last_result = run(stage_penal, initial=warm_start)
        warm_start = last_result.densities
        designs.append(warm_start)

    for seed in seed_designs:
        seed = np.clip(np.asarray(seed, dtype=float), 0.0, 1.0).reshape(
            problem.nely, problem.nelx
        )
        seed = np.where(free, seed, pinned)
        designs.append(seed)
        designs.append(run(penal, initial=seed).densities)

    candidates = []
    for design in designs:
        candidates.append(design)
        candidates.append(
            binarize_densities(
                design, free_cells=free, target_solid_cells=free_budget
            )
        )

    best_design = None
    best_compliance = np.inf
    for candidate in candidates:
        if np.sum(candidate) > budget:
            continue
        compliance = get_compliance(model, candidate, penal=penal)
        if compliance < best_compliance:
            best_compliance = compliance
            best_design = candidate

    if best_design is None:
        best_design = candidates[-1]
        best_compliance = get_compliance(model, best_design, penal=penal)

    if polish:
        polished, polished_compliance = polish_design(
            problem=problem,
            densities=binarize_densities(
                best_design, free_cells=free, target_solid_cells=free_budget
            ),
            free_cells=free,
            penal=penal,
        )
        if polished_compliance < best_compliance and np.sum(polished) <= budget:
            best_design, best_compliance = polished, polished_compliance

    return OptimizationResult(
        densities=np.array(best_design, dtype=float).reshape(
            problem.nely, problem.nelx
        ),
        compliance=float(best_compliance),
        volume_fraction=float(np.mean(best_design)),
        iterations=last_result.iterations if last_result is not None else 0,
        converged=last_result.converged if last_result is not None else False,
    )

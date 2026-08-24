"""Dynamic, simulation-based evaluation of a completed SPhyR grid.

Instead of comparing a completion to the reference answer cell by cell, the
metrics in this module put the completion through the same physics the dataset
was generated with:

1. The load case encoded in the sample (loads, supports, load direction) is
   rebuilt and the completion is solved as a linear elastic structure, giving
   its compliance - the work done by the load, i.e. the inverse of stiffness.
2. A SIMP topology optimiser is then re-run on exactly the masked region, given
   exactly the material the model spent there, producing the stiffest structure
   that could have been drawn into that hole.
3. The completion is scored by how close it gets to that optimum.

A completion that differs from the reference answer but carries the load just
as efficiently now scores as well as the reference answer, and a completion
that matches the reference answer almost everywhere but severs the load path
scores zero.
"""

from dataclasses import asdict, dataclass

import numpy as np

from sphyr.physics.analysis import analyze_structure
from sphyr.physics.boundary_conditions import (
    InvalidProblemError,
    get_densities_from_grid,
    get_free_cells_mask,
    get_problem_from_grid,
)
from sphyr.physics.simp import (
    DEFAULT_FILTER_RADIUS,
    DEFAULT_PENAL,
    optimize_reference_design,
)

_REFERENCE_OPTIMUM_CACHE = {}

# Structural metrics that are meaningful to average over a subject.  Raw
# compliances are deliberately left out: they are per-sample physical
# quantities whose scale depends on the load case, so a mean over samples says
# nothing useful.
STRUCTURAL_METRIC_KEYS = (
    "topology_score",
    "structural_efficiency",
    "material_efficiency",
    "compliance_efficiency_vs_ground_truth",
    "load_carrying",
    "volume_ratio",
)


@dataclass
class StructuralMetrics:
    """Physics-based scores for one completion.

    Ratios are clipped to [0, 1] and are "higher is better"; compliance values
    are raw physical quantities and are "lower is better".
    """

    physically_valid: bool = False
    load_carrying: bool = False
    compliance: float = float("inf")
    ground_truth_compliance: float = float("inf")
    optimal_compliance: float = float("inf")
    own_budget_optimal_compliance: float = None
    topology_score: float = 0.0
    structural_efficiency: float = 0.0
    structural_efficiency_raw: float = 0.0
    design_efficiency: float = None
    material_efficiency: float = 0.0
    compliance_efficiency_vs_ground_truth: float = 0.0
    volume_fraction: float = 0.0
    ground_truth_volume_fraction: float = 0.0
    volume_ratio: float = 0.0
    max_displacement: float = 0.0
    unfilled_cells: int = 0
    unparsable_cells: int = 0

    def as_dict(self):
        """Serialisable form; non-finite compliances become ``None``.

        ``inf`` is not valid JSON, and an unsupported design legitimately has
        infinite compliance, so it is written out as a null instead.
        """
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, float) and not np.isfinite(value):
                values[key] = None
        return values


def _clipped_ratio(reference, actual):
    """``reference / actual`` clipped to [0, 1], safe against inf and zero."""
    if not np.isfinite(actual) or actual <= 0.0:
        return 0.0
    if not np.isfinite(reference):
        return 0.0
    return float(min(max(reference / actual, 0.0), 1.0))


def get_reference_optimum(
    problem,
    free_cells,
    pinned_densities,
    volume_fraction,
    penal=DEFAULT_PENAL,
    filter_radius=DEFAULT_FILTER_RADIUS,
    seed_designs=(),
):
    """SIMP optimum for one masked region at one material budget.

    Results are cached: the same sample is completed by many models, and many
    completions land on the same material budget, so the optimiser usually runs
    only a handful of times per sample.
    """
    key = (
        problem,
        free_cells.tobytes(),
        np.round(pinned_densities, 6).tobytes(),
        round(float(volume_fraction), 6),
        float(penal),
        float(filter_radius),
        tuple(np.round(seed, 6).tobytes() for seed in seed_designs),
    )

    if key not in _REFERENCE_OPTIMUM_CACHE:
        _REFERENCE_OPTIMUM_CACHE[key] = optimize_reference_design(
            problem=problem,
            volume_fraction=volume_fraction,
            free_cells=free_cells,
            fixed_densities=pinned_densities,
            penal=penal,
            filter_radius=filter_radius,
            seed_designs=seed_designs,
        )

    return _REFERENCE_OPTIMUM_CACHE[key]


def get_structural_metrics(
    output_grid,
    gt_grid,
    input_grid=None,
    gravity_dir=(1, 0),
    penal=DEFAULT_PENAL,
    filter_radius=DEFAULT_FILTER_RADIUS,
    include_own_budget_reference=False,
):
    """Evaluate a completion by simulating it and re-optimising its mask.

    Boundary conditions are always read from ``gt_grid`` so that a completion
    cannot improve its score by relocating loads or supports.  ``input_grid``
    identifies the masked cells; without it the whole non-structural grid is
    treated as the masked region.

    ``include_own_budget_reference`` adds ``design_efficiency``, which re-runs
    the optimiser at the material budget the completion actually chose.  It is
    off by default because it needs one optimiser run per distinct budget
    instead of one per sample.

    The headline number is ``topology_score``: stiffness relative to the best
    design achievable on the reference material budget, discounted by any
    material the completion spent beyond that budget.  Both halves are needed -
    stiffness alone is maximised by filling the grid solid, and thrift alone by
    building nothing.
    """
    metrics = StructuralMetrics()

    try:
        problem = get_problem_from_grid(gt_grid, gravity_dir=gravity_dir)
    except (InvalidProblemError, IndexError, TypeError):
        return metrics

    shape = (problem.nely, problem.nelx)
    if len(output_grid) != shape[0] or any(len(row) != shape[1] for row in output_grid):
        return metrics

    structural_mask = problem.structural_cells_mask()

    output_densities, output_report = get_densities_from_grid(output_grid)
    gt_densities, _ = get_densities_from_grid(gt_grid)
    output_densities[structural_mask] = 1.0
    gt_densities[structural_mask] = 1.0

    metrics.physically_valid = True
    metrics.unfilled_cells = output_report["unfilled_cells"]
    metrics.unparsable_cells = output_report["unparsable_cells"]

    output_analysis = analyze_structure(problem, output_densities, penal=penal)
    gt_analysis = analyze_structure(problem, gt_densities, penal=penal)

    metrics.compliance = output_analysis.compliance
    metrics.ground_truth_compliance = gt_analysis.compliance
    metrics.load_carrying = output_analysis.load_carrying
    metrics.max_displacement = output_analysis.max_displacement
    metrics.volume_fraction = output_analysis.volume_fraction
    metrics.ground_truth_volume_fraction = gt_analysis.volume_fraction

    free_cells = get_free_cells_mask(input_grid, shape)
    if free_cells is None:
        free_cells = ~structural_mask

    # The yardstick: the stiffest design available on the reference budget,
    # seeded with the reference answer so it can never be worse than it.
    optimum = get_reference_optimum(
        problem=problem,
        free_cells=free_cells,
        pinned_densities=gt_densities,
        volume_fraction=gt_analysis.volume_fraction,
        penal=penal,
        filter_radius=filter_radius,
        seed_designs=(gt_densities,),
    )
    metrics.optimal_compliance = optimum.compliance

    if include_own_budget_reference:
        own_budget_optimum = get_reference_optimum(
            problem=problem,
            free_cells=free_cells,
            pinned_densities=gt_densities,
            volume_fraction=output_analysis.volume_fraction,
            penal=penal,
            filter_radius=filter_radius,
            seed_designs=(gt_densities,),
        )
        metrics.own_budget_optimal_compliance = own_budget_optimum.compliance

    if gt_analysis.volume_fraction > 0.0:
        metrics.volume_ratio = float(
            output_analysis.volume_fraction / gt_analysis.volume_fraction
        )

    if not output_analysis.load_carrying:
        return metrics

    metrics.structural_efficiency = _clipped_ratio(
        optimum.compliance, output_analysis.compliance
    )
    if metrics.own_budget_optimal_compliance is not None:
        metrics.design_efficiency = _clipped_ratio(
            metrics.own_budget_optimal_compliance, output_analysis.compliance
        )
    metrics.compliance_efficiency_vs_ground_truth = _clipped_ratio(
        gt_analysis.compliance, output_analysis.compliance
    )

    if np.isfinite(optimum.compliance) and output_analysis.compliance > 0.0:
        metrics.structural_efficiency_raw = float(
            optimum.compliance / output_analysis.compliance
        )

    # Spending more material than the reference answer is discounted in
    # proportion to the overspend; spending less is never penalised here, it
    # already shows up as a loss of stiffness.
    if output_analysis.volume_fraction > 0.0:
        metrics.material_efficiency = float(
            min(
                1.0,
                gt_analysis.volume_fraction / output_analysis.volume_fraction,
            )
        )

    metrics.topology_score = float(
        metrics.structural_efficiency * metrics.material_efficiency
    )

    return metrics


def clear_reference_cache():
    """Drop cached optimiser runs (mainly useful in tests)."""
    _REFERENCE_OPTIMUM_CACHE.clear()

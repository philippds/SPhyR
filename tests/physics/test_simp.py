import numpy as np
import pytest

from sphyr.physics.analysis import analyze_structure
from sphyr.physics.boundary_conditions import (
    InvalidProblemError,
    get_densities_from_grid,
    get_free_cells_mask,
    get_problem_from_grid,
)
from sphyr.physics.simp import (
    binarize_densities,
    optimize_reference_design,
    optimize_topology,
    polish_design,
)

COLUMN_GRID = [
    ["L", "L", "0", "0"],
    ["V", "V", "V", "V"],
    ["V", "V", "V", "V"],
    ["S", "S", "0", "0"],
]


def build_problem(grid=COLUMN_GRID):
    return get_problem_from_grid(grid)


def test_problem_reads_loads_and_supports_from_the_grid():
    problem = build_problem()

    assert problem.load_cells == ((0, 0), (0, 1))
    assert problem.support_cells == ((3, 0), (3, 1))
    assert problem.element_count == 16


def test_problem_rejects_a_grid_without_supports():
    with pytest.raises(InvalidProblemError):
        get_problem_from_grid([["L", "0"], ["0", "0"]])


def test_load_vector_carries_a_unit_resultant_along_gravity():
    problem = build_problem()
    forces = problem.force_vector()

    assert forces.reshape(-1, 2)[:, 1].sum() == pytest.approx(1.0)
    assert forces.reshape(-1, 2)[:, 0].sum() == pytest.approx(0.0)


def test_rotated_gravity_pushes_along_the_rotated_axis():
    problem = get_problem_from_grid(COLUMN_GRID, gravity_dir=(0, 1))
    forces = problem.force_vector().reshape(-1, 2)

    assert forces[:, 0].sum() == pytest.approx(1.0)
    assert forces[:, 1].sum() == pytest.approx(0.0)


def test_densities_treat_loads_and_supports_as_solid():
    densities, report = get_densities_from_grid(
        [["L", "0.5"], ["S", "nonsense"]]
    )

    assert densities[0][0] == 1.0
    assert densities[1][0] == 1.0
    assert densities[0][1] == pytest.approx(0.5)
    assert report["unparsable_cells"] == 1


def test_masked_cells_are_reported_as_unfilled():
    _, report = get_densities_from_grid([["L", "V"], ["S", "0"]])
    assert report["unfilled_cells"] == 1


def test_optimizer_respects_the_volume_budget():
    problem = build_problem()
    result = optimize_topology(problem, volume_fraction=0.5)

    assert result.volume_fraction <= 0.5 + 1e-6


def test_optimizer_keeps_loads_and_supports_solid():
    problem = build_problem()
    result = optimize_topology(problem, volume_fraction=0.3)

    for row, col in problem.load_cells + problem.support_cells:
        assert result.densities[row, col] == pytest.approx(1.0)


def test_optimizer_only_moves_the_free_cells():
    problem = build_problem()
    pinned = np.zeros((4, 4))
    pinned[1, 3] = 0.4

    free = np.zeros((4, 4), dtype=bool)
    free[2, :] = True

    result = optimize_topology(
        problem, volume_fraction=0.5, free_cells=free, fixed_densities=pinned
    )

    assert result.densities[1, 3] == pytest.approx(0.4)
    assert result.densities[1, 0] == pytest.approx(0.0)


def test_more_material_buys_a_stiffer_optimum():
    problem = build_problem()

    lean = optimize_reference_design(problem, volume_fraction=0.3, polish=False)
    generous = optimize_reference_design(problem, volume_fraction=0.6, polish=False)

    assert generous.compliance < lean.compliance


def test_reference_design_is_never_worse_than_its_seed():
    grid = [
        ["L", "L", "0", "0"],
        ["V", "V", "V", "V"],
        ["V", "V", "V", "V"],
        ["S", "S", "0", "0"],
    ]
    problem = get_problem_from_grid(grid)
    seed = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    seed_compliance = analyze_structure(problem, seed).compliance

    result = optimize_reference_design(
        problem, volume_fraction=float(seed.mean()), seed_designs=(seed,)
    )

    assert result.compliance <= seed_compliance + 1e-9


def test_binarisation_spends_the_budget_without_exceeding_it():
    densities = np.array([[0.9, 0.6], [0.4, 0.1]])
    binary = binarize_densities(densities, target_solid_cells=2)

    assert binary.sum() == pytest.approx(2.0)
    assert binary[0, 0] == 1.0 and binary[1, 1] == 0.0


def test_binarisation_rounds_the_budget_down():
    densities = np.array([[0.9, 0.6], [0.4, 0.1]])
    assert binarize_densities(densities, target_solid_cells=2.9).sum() == 2.0


def test_polish_never_makes_a_design_worse():
    problem = build_problem()
    free = get_free_cells_mask(COLUMN_GRID, (4, 4))
    design = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    before = analyze_structure(problem, design).compliance

    _, after = polish_design(problem, design, free_cells=free)

    assert after <= before + 1e-12


def test_polish_preserves_the_material_budget():
    problem = build_problem()
    free = get_free_cells_mask(COLUMN_GRID, (4, 4))
    design = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    polished, _ = polish_design(problem, design, free_cells=free)

    assert polished.sum() == pytest.approx(design.sum())

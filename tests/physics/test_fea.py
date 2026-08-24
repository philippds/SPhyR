import numpy as np
import pytest

from sphyr.physics.fea import (
    FEAModel,
    get_element_stiffness_matrix,
    simp_stiffness,
)


def analytic_element_stiffness(nu=0.3):
    """Closed-form Q4 stiffness matrix used by the 99/88-line SIMP codes."""
    k = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )
    order = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 7, 6, 5, 4, 3, 2],
            [2, 7, 0, 5, 6, 3, 4, 1],
            [3, 6, 5, 0, 7, 2, 1, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 3, 2, 1, 0, 7, 6],
            [6, 3, 4, 1, 2, 7, 0, 5],
            [7, 2, 1, 4, 3, 6, 5, 0],
        ]
    )
    return k[order] / (1 - nu**2)


def test_element_stiffness_matches_reference_implementation():
    assert get_element_stiffness_matrix(0.3) == pytest.approx(
        analytic_element_stiffness(0.3), abs=1e-12
    )


def test_element_stiffness_has_three_rigid_body_modes():
    stiffness = get_element_stiffness_matrix(0.3)
    assert np.linalg.matrix_rank(stiffness, tol=1e-10) == 5
    assert stiffness == pytest.approx(stiffness.T)


def test_rigid_body_translation_produces_no_force():
    stiffness = get_element_stiffness_matrix(0.3)
    translation = np.array([1.0, 0.0] * 4)
    assert stiffness @ translation == pytest.approx(np.zeros(8), abs=1e-12)


def test_simp_stiffness_interpolates_between_void_and_solid():
    moduli = simp_stiffness([0.0, 0.5, 1.0], penal=3.0, e_0=1.0, e_min=1e-9)
    assert moduli[0] == pytest.approx(1e-9)
    assert moduli[1] == pytest.approx(0.125, rel=1e-6)
    assert moduli[2] == pytest.approx(1.0)


def build_cantilever(nely=4, nelx=8):
    """Tip-loaded cantilever clamped along its left edge."""
    node_count = (nely + 1) * (nelx + 1)
    fixed = [
        dof
        for row in range(nely + 1)
        for dof in (2 * (row * (nelx + 1)), 2 * (row * (nelx + 1)) + 1)
    ]
    forces = np.zeros(2 * node_count)
    forces[2 * (nely * (nelx + 1) + nelx) + 1] = 1.0
    return FEAModel(nely=nely, nelx=nelx, fixed_dofs=fixed, forces=forces)


def test_compliance_decreases_when_material_is_added():
    model = build_cantilever()
    sparse = np.full(model.element_count, 0.4)
    dense = np.full(model.element_count, 0.8)

    _, sparse_compliance = model.solve(simp_stiffness(sparse))
    _, dense_compliance = model.solve(simp_stiffness(dense))

    assert dense_compliance < sparse_compliance


def test_compliance_is_scale_invariant_under_uniform_stiffness():
    """Halving every modulus doubles the compliance of a linear structure."""
    model = build_cantilever()
    densities = np.full(model.element_count, 1.0)

    _, reference = model.solve(simp_stiffness(densities))
    _, halved = model.solve(0.5 * simp_stiffness(densities))

    assert halved == pytest.approx(2.0 * reference, rel=1e-6)


def test_empty_structure_is_effectively_unsupported():
    model = build_cantilever()
    solid = np.ones(model.element_count)
    void = np.zeros(model.element_count)

    _, solid_compliance = model.solve(simp_stiffness(solid))
    _, void_compliance = model.solve(simp_stiffness(void))

    assert void_compliance > 1e6 * solid_compliance


def test_solve_without_supports_reports_infinite_compliance():
    forces = np.zeros(2 * 3 * 3)
    forces[1] = 1.0
    model = FEAModel(nely=2, nelx=2, fixed_dofs=[], forces=forces)

    _, compliance = model.solve(np.ones(4))

    assert not np.isfinite(compliance)


def test_element_dofs_reference_the_expected_nodes():
    model = FEAModel(nely=2, nelx=3, fixed_dofs=[0], forces=np.zeros(2 * 12))
    # Element 0 spans nodes 0, 1 (top) and 4, 5 (bottom) of a 4-column grid.
    assert list(model.element_dofs[0]) == [0, 1, 2, 3, 10, 11, 8, 9]

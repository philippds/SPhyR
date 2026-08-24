"""Linear-elastic finite element analysis for structured 2D grids.

Every grid cell is modelled as a unit-square bilinear (Q4) plane-stress element.
Element stiffness is interpolated from the cell density with the SIMP law

    E(rho) = E_min + rho ** penal * (E_0 - E_min)

so that a density of 0 leaves a numerically negligible (but non-singular)
stiffness behind and a density of 1 recovers the full material stiffness.

Node numbering is row-major over the (nely + 1) x (nelx + 1) node grid, with
row 0 at the top of the grid.  Each node owns two degrees of freedom, laid out
as ``[u_x, u_y]``; ``+x`` points along increasing column index and ``+y`` along
increasing row index, i.e. "down" the printed grid.
"""

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Relative equilibrium residual above which a solve is treated as singular.
RESIDUAL_TOLERANCE = 1e-6

GAUSS_POINTS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

# Local node order of an element: top-left, top-right, bottom-right, bottom-left.
ELEMENT_NODE_COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def get_constitutive_matrix(nu=0.3):
    """Plane-stress constitutive matrix for unit Young's modulus."""
    return np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    ) / (1.0 - nu**2)


def get_strain_displacement_matrix(xi, eta, node_coords=ELEMENT_NODE_COORDS):
    """Strain-displacement matrix B (3x8) of a Q4 element at (xi, eta)."""
    d_shape = 0.25 * np.array(
        [
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        ]
    )

    jacobian = d_shape @ node_coords
    d_shape_xy = np.linalg.solve(jacobian, d_shape)

    b_matrix = np.zeros((3, 8))
    b_matrix[0, 0::2] = d_shape_xy[0]
    b_matrix[1, 1::2] = d_shape_xy[1]
    b_matrix[2, 0::2] = d_shape_xy[1]
    b_matrix[2, 1::2] = d_shape_xy[0]

    return b_matrix, np.linalg.det(jacobian)


def get_element_stiffness_matrix(nu=0.3):
    """Stiffness matrix (8x8) of a unit square Q4 element with unit modulus.

    Integrated with 2x2 Gauss quadrature; identical to the closed-form matrix
    used by the classic 99/88-line topology optimisation codes.
    """
    constitutive = get_constitutive_matrix(nu)
    stiffness = np.zeros((8, 8))

    for xi in GAUSS_POINTS:
        for eta in GAUSS_POINTS:
            b_matrix, det_jacobian = get_strain_displacement_matrix(xi, eta)
            stiffness += b_matrix.T @ constitutive @ b_matrix * det_jacobian

    return stiffness


def simp_stiffness(densities, penal=3.0, e_0=1.0, e_min=1e-9):
    """SIMP-interpolated Young's modulus of each element."""
    densities = np.clip(np.asarray(densities, dtype=float), 0.0, 1.0)
    return e_min + densities**penal * (e_0 - e_min)


class FEAModel:
    """Reusable assembly/solve machinery for a fixed grid and load case.

    The mesh connectivity, the sparse assembly indices and the element
    stiffness matrix only depend on the grid shape, so they are built once and
    reused across every density field evaluated on that grid.
    """

    def __init__(self, nely, nelx, fixed_dofs, forces, nu=0.3):
        self.nely = nely
        self.nelx = nelx
        self.element_count = nely * nelx
        self.node_count = (nely + 1) * (nelx + 1)
        self.dof_count = 2 * self.node_count

        self.element_stiffness = get_element_stiffness_matrix(nu)
        self.element_dofs = self._build_element_dofs()

        self.forces = np.asarray(forces, dtype=float).reshape(self.dof_count)

        fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=int))
        self.fixed_dofs = fixed_dofs
        self.free_dofs = np.setdiff1d(np.arange(self.dof_count), fixed_dofs)

        rows = np.repeat(self.element_dofs, 8, axis=1).ravel()
        cols = np.tile(self.element_dofs, (1, 8)).ravel()
        self._assembly_rows = rows
        self._assembly_cols = cols
        self._element_stiffness_flat = self.element_stiffness.ravel()

    def _build_element_dofs(self):
        """Global DOF indices (element_count x 8) in local node order."""
        rows, cols = np.divmod(np.arange(self.element_count), self.nelx)
        top_left = rows * (self.nelx + 1) + cols
        top_right = top_left + 1
        bottom_left = top_left + (self.nelx + 1)
        bottom_right = bottom_left + 1

        nodes = np.column_stack([top_left, top_right, bottom_right, bottom_left])
        element_dofs = np.empty((self.element_count, 8), dtype=int)
        element_dofs[:, 0::2] = 2 * nodes
        element_dofs[:, 1::2] = 2 * nodes + 1
        return element_dofs

    def assemble(self, element_moduli):
        """Assemble the global stiffness matrix for the given element moduli."""
        element_moduli = np.asarray(element_moduli, dtype=float).reshape(-1)
        values = (element_moduli[:, None] * self._element_stiffness_flat[None, :]).ravel()

        stiffness = coo_matrix(
            (values, (self._assembly_rows, self._assembly_cols)),
            shape=(self.dof_count, self.dof_count),
        ).tocsc()

        return stiffness

    def solve(self, element_moduli):
        """Solve K u = f and return (displacements, compliance).

        Compliance is ``f . u``; an unsolvable system (no supports, no load, or
        a numerically singular stiffness matrix) yields ``inf`` so that callers
        can treat it as a structurally useless design instead of crashing.
        """
        if self.free_dofs.size == 0 or not np.any(self.forces):
            return np.zeros(self.dof_count), np.inf

        stiffness = self.assemble(element_moduli)
        free = self.free_dofs

        displacements = np.zeros(self.dof_count)
        free_stiffness = stiffness[free, :][:, free].tocsc()
        free_forces = self.forces[free]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                solution = spsolve(free_stiffness, free_forces)
            except Exception:
                return displacements, np.inf

        solution = np.asarray(solution, dtype=float)
        if not np.all(np.isfinite(solution)):
            return displacements, np.inf

        # A structure with no load path leaves the stiffness matrix singular.
        # The sparse solver does not always say so - it can return a huge but
        # finite vector - so the solution is only trusted if it actually
        # satisfies the equilibrium equations.
        residual = np.linalg.norm(free_stiffness @ solution - free_forces)
        if residual > RESIDUAL_TOLERANCE * max(np.linalg.norm(free_forces), 1.0):
            return displacements, np.inf

        displacements[free] = solution
        compliance = float(self.forces @ displacements)

        if not np.isfinite(compliance) or compliance < 0.0:
            return displacements, np.inf

        return displacements, compliance

    def element_strain_energies(self, displacements):
        """Per-element compliance contribution for unit modulus.

        Multiplying by the element modulus gives the actual strain energy
        density used both for reporting and for compliance sensitivities.
        """
        element_displacements = displacements[self.element_dofs]
        return np.einsum(
            "ij,jk,ik->i",
            element_displacements,
            self.element_stiffness,
            element_displacements,
        )

    def max_displacement(self, displacements):
        """Largest nodal displacement magnitude."""
        nodal = displacements.reshape(-1, 2)
        return float(np.max(np.linalg.norm(nodal, axis=1)))

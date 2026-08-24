"""Translation of SPhyR grids into well-posed structural problems.

An SPhyR grid encodes its own boundary conditions: ``L`` cells carry the
applied load, ``S`` cells are anchored to ground, ``V`` marks a masked cell the
model has to fill in and every other cell holds a density in [0, 1].

The load is applied as a unit resultant, spread evenly over the nodes of the
``L`` cells, so that compliance values are comparable across samples with
differently sized load patches.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sphyr.physics.fea import FEAModel

LOAD_TOKEN = "L"
SUPPORT_TOKEN = "S"
MASK_TOKEN = "V"

STRUCTURAL_TOKENS = (LOAD_TOKEN, SUPPORT_TOKEN)


class InvalidProblemError(ValueError):
    """Raised when a grid does not describe a solvable structural problem."""


@dataclass(frozen=True)
class StructuralProblem:
    """Boundary conditions of one SPhyR sample.

    Frozen (and therefore hashable) so that the assembled FEA model and the
    reference optimisations can be cached across the many completions that
    share a single sample.
    """

    nely: int
    nelx: int
    load_cells: tuple
    support_cells: tuple
    load_direction: tuple = (1.0, 0.0)
    total_load: float = 1.0

    @property
    def element_count(self):
        return self.nely * self.nelx

    def node_index(self, row, col):
        return row * (self.nelx + 1) + col

    def cell_nodes(self, row, col):
        return (
            self.node_index(row, col),
            self.node_index(row, col + 1),
            self.node_index(row + 1, col),
            self.node_index(row + 1, col + 1),
        )

    def fixed_dofs(self):
        """All degrees of freedom clamped by the support cells."""
        dofs = set()
        for row, col in self.support_cells:
            for node in self.cell_nodes(row, col):
                dofs.add(2 * node)
                dofs.add(2 * node + 1)
        return np.array(sorted(dofs), dtype=int)

    def force_vector(self):
        """Nodal load vector with a unit resultant over the load cells."""
        forces = np.zeros(2 * (self.nely + 1) * (self.nelx + 1))
        if not self.load_cells:
            return forces

        direction = np.array(self.load_direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise InvalidProblemError("load direction must be non-zero")
        direction = direction / norm

        share = self.total_load / (4.0 * len(self.load_cells))
        for row, col in self.load_cells:
            for node in self.cell_nodes(row, col):
                forces[2 * node] += share * direction[1]
                forces[2 * node + 1] += share * direction[0]

        return forces

    def structural_cells_mask(self):
        """Cells that are structure by definition (loads and supports)."""
        mask = np.zeros((self.nely, self.nelx), dtype=bool)
        for row, col in self.load_cells + self.support_cells:
            mask[row, col] = True
        return mask

    def build_model(self):
        return _build_model(self)


@lru_cache(maxsize=256)
def _build_model(problem):
    """Cached FEA model for a problem (mesh and assembly indices are reused)."""
    return FEAModel(
        nely=problem.nely,
        nelx=problem.nelx,
        fixed_dofs=problem.fixed_dofs(),
        forces=problem.force_vector(),
    )


def get_problem_from_grid(grid, gravity_dir=(1, 0), total_load=1.0):
    """Extract the boundary conditions encoded in a grid.

    ``gravity_dir`` is a (row, col) step, matching
    :func:`sphyr.metrics.utils.get_gravity_from_folder`, so rotated variants of
    a sample are loaded along their own rotated gravity direction.
    """
    if not grid or not grid[0]:
        raise InvalidProblemError("empty grid")

    nely = len(grid)
    nelx = len(grid[0])
    if any(len(row) != nelx for row in grid):
        raise InvalidProblemError("ragged grid")

    load_cells = []
    support_cells = []
    for row in range(nely):
        for col in range(nelx):
            token = str(grid[row][col]).strip()
            if token == LOAD_TOKEN:
                load_cells.append((row, col))
            elif token == SUPPORT_TOKEN:
                support_cells.append((row, col))

    if not load_cells:
        raise InvalidProblemError("grid contains no load ('L') cells")
    if not support_cells:
        raise InvalidProblemError("grid contains no support ('S') cells")

    return StructuralProblem(
        nely=nely,
        nelx=nelx,
        load_cells=tuple(load_cells),
        support_cells=tuple(support_cells),
        load_direction=(float(gravity_dir[0]), float(gravity_dir[1])),
        total_load=total_load,
    )


def get_densities_from_grid(grid, mask_density=0.0):
    """Convert a grid into a density field plus a bookkeeping report.

    Loads and supports are solid by definition.  Masked ('V') and unparsable
    cells fall back to ``mask_density`` and are counted, so that a completion
    which never filled its holes can be told apart from one that filled them
    with zeros.
    """
    nely = len(grid)
    nelx = len(grid[0])

    densities = np.zeros((nely, nelx), dtype=float)
    unfilled_cells = 0
    unparsable_cells = 0

    for row in range(nely):
        for col in range(nelx):
            token = str(grid[row][col]).strip()
            if token in STRUCTURAL_TOKENS:
                densities[row, col] = 1.0
            elif token == MASK_TOKEN:
                densities[row, col] = mask_density
                unfilled_cells += 1
            else:
                try:
                    densities[row, col] = min(max(float(token), 0.0), 1.0)
                except ValueError:
                    densities[row, col] = mask_density
                    unparsable_cells += 1

    return densities, {
        "unfilled_cells": unfilled_cells,
        "unparsable_cells": unparsable_cells,
    }


def get_free_cells_mask(input_grid, shape):
    """Boolean mask of the cells a model was actually asked to fill in.

    Falls back to "everything that is not a load or a support" when the input
    grid is unavailable or does not line up with the completion.
    """
    nely, nelx = shape
    mask = np.zeros((nely, nelx), dtype=bool)

    if not input_grid or len(input_grid) != nely:
        return None
    if any(len(row) != nelx for row in input_grid):
        return None

    for row in range(nely):
        for col in range(nelx):
            if str(input_grid[row][col]).strip() == MASK_TOKEN:
                mask[row, col] = True

    return mask

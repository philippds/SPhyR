"""Structural analysis of a density field on an SPhyR grid.

Everything here is a physical measurement of a design: no reference answer is
involved, the numbers come out of the finite element solve.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sphyr.physics.fea import simp_stiffness
from sphyr.physics.simp import DEFAULT_PENAL

# A design whose compliance exceeds this multiple of the fully solid design has
# no usable load path: the load is carried by the numerically negligible void
# stiffness rather than by material.
SINGULARITY_RATIO = 1e3


@dataclass
class StructureAnalysis:
    """Physical response of one design under the sample load case."""

    compliance: float
    volume_fraction: float
    max_displacement: float
    load_carrying: bool
    solid_compliance: float

    def as_dict(self):
        return {
            "compliance": self.compliance,
            "volume_fraction": self.volume_fraction,
            "max_displacement": self.max_displacement,
            "load_carrying": self.load_carrying,
        }


@lru_cache(maxsize=256)
def get_solid_compliance(problem, penal=DEFAULT_PENAL):
    """Compliance of the fully solid grid: the stiffest design possible.

    Used both as a lower bound reference and as the yardstick for deciding
    whether a design carries its load at all.
    """
    model = problem.build_model()
    densities = np.ones(problem.element_count)
    _, compliance = model.solve(simp_stiffness(densities, penal=penal))
    return compliance


def analyze_structure(problem, densities, penal=DEFAULT_PENAL):
    """Solve the load case for ``densities`` and report the response."""
    model = problem.build_model()
    densities = np.clip(np.asarray(densities, dtype=float).ravel(), 0.0, 1.0)

    displacements, compliance = model.solve(simp_stiffness(densities, penal=penal))
    solid_compliance = get_solid_compliance(problem, penal=penal)

    load_carrying = bool(
        np.isfinite(compliance)
        and np.isfinite(solid_compliance)
        and compliance <= SINGULARITY_RATIO * solid_compliance
    )

    return StructureAnalysis(
        compliance=float(compliance),
        volume_fraction=float(densities.mean()),
        max_displacement=model.max_displacement(displacements),
        load_carrying=load_carrying,
        solid_compliance=float(solid_compliance),
    )

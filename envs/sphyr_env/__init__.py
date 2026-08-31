"""SPhyR: a spatial-physical reasoning environment, packaged for OpenEnv."""

from .client import SPhyREnv
from .models import SPhyRAction, SPhyRObservation

__all__ = ["SPhyRAction", "SPhyRObservation", "SPhyREnv"]

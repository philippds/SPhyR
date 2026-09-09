"""Data models for the SPhyR Environment.

SPhyR asks an agent to complete a masked structural material distribution.
An episode is a single step: ``reset()`` hands out one masked grid, ``step()``
submits the completed grid and returns the benchmark's full metric set with a
scalar reward.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SPhyRAction(Action):
    """Action for the SPhyR environment - the agent's completed grid."""

    grid: str = Field(
        ...,
        description=(
            "The completed grid: one row per line, cells separated by spaces. "
            "Surrounding prose or code fences are tolerated; the last grid "
            "block in the text is taken as the answer."
        ),
    )


class SPhyRObservation(Observation):
    """Observation from the SPhyR environment.

    The task fields (``prompt`` through ``masked_cells``) are populated by
    ``reset()``; the outcome fields (``completion`` through ``metrics``) by
    ``step()``.
    """

    prompt: Optional[str] = Field(
        default=None,
        description="The prompt describing the completion task (None after step)",
    )
    grid: Optional[str] = Field(
        default=None,
        description="The masked input grid, with 'L' loads, 'S' supports and 'V' voids to fill",
    )
    subject: Optional[str] = Field(
        default=None,
        description="The SPhyR subject this sample came from, e.g. '10_random_cell_easy'",
    )
    sample_index: Optional[int] = Field(
        default=None,
        description="Index of the sample within the shuffled subject",
    )
    rows: Optional[int] = Field(default=None, description="Grid height in cells")
    cols: Optional[int] = Field(default=None, description="Grid width in cells")
    cell_values: Optional[str] = Field(
        default=None,
        description="'binary' if masked cells take 0/1, 'density' if they take 0.0-1.0",
    )
    masked_cells: Optional[int] = Field(
        default=None,
        description="Number of 'V' cells the agent has to fill",
    )
    rotation_count: Optional[int] = Field(
        default=None,
        description="Quarter turns applied to the sample; the load case rotates with it",
    )
    completion: Optional[str] = Field(
        default=None,
        description="The grid text the agent submitted (revealed after step)",
    )
    ground_truth: Optional[str] = Field(
        default=None,
        description="The dataset's reference answer (revealed after step)",
    )
    valid_grid: Optional[bool] = Field(
        default=None,
        description="Whether the submitted grid had the right shape and cell values",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "The full SPhyR metric set for the completion, including the "
            "simulation-based topology_score, structural_efficiency, "
            "material_efficiency, compliance and load_carrying"
        ),
    )

"""Scoring for a SPhyR completion: one grid in, every benchmark metric out.

The environment calls :func:`calculate_all_metrics` from ``step()`` and the
offline rescoring path calls it over stored result files, so the numbers a
model is trained against and the numbers in the paper come from one place.
"""

from dataclasses import dataclass

from sphyr.metrics.physics_approximation import (
    get_force_path_cost_average_efficiency_ratio,
)
from sphyr.metrics.reconstruction import (
    get_difference_ratio,
    get_exact_match,
    get_penalized_difference_ratio,
    get_relative_difference_ratio,
)
from sphyr.metrics.structural import (
    STRUCTURAL_METRIC_KEYS,
    get_structural_metrics,
)
from sphyr.metrics.topology import (
    get_difficulty_score,
    get_isolated_clusters_count,
    is_load_supported,
    is_load_supported_force_directional,
)
from sphyr.metrics.utils import get_grid_shape_and_value_validity

# The scalar an RL loop optimises.  ``topology_score`` is the headline dynamic
# metric: stiffness relative to the best design achievable on the reference
# material budget, discounted by any material spent beyond it.
REWARD_METRIC = "topology_score"


@dataclass
class Result:
    subject: str
    prompt: str
    ground_truth: str
    completion: str
    exact_match: bool
    difference_ratio: float  # lower is better
    penalized_difference_ratio: float  # lower is better
    relative_difference_ratio: float  # lower is better
    valid_output_grid: bool
    load_support_connected: bool
    load_support_connected_force_directional: bool
    isolated_clusters_count: int  # lower is better
    force_path_cost_average_efficiency_ratio: float
    difficulty_score: float  # higher is more difficult
    difficulty_weighted_difference_ratio: float  # lower is better
    difficulty_weighted_relative_difference_ratio: float  # lower is better

    # Dynamic, simulation-based metrics.  Defaulted so that result files
    # written before the structural evaluation existed still load.
    topology_score: float = 0.0  # higher is better
    structural_efficiency: float = 0.0  # higher is better
    material_efficiency: float = 0.0  # higher is better
    compliance_efficiency_vs_ground_truth: float = 0.0  # higher is better
    load_carrying: bool = False
    compliance: float = None  # lower is better
    volume_ratio: float = 0.0  # 1.0 matches the reference material usage

    @staticmethod
    def from_dict(result_dict):
        return Result(
            subject=result_dict["subject"],
            prompt=result_dict["prompt"],
            ground_truth=result_dict["ground_truth"],
            completion=result_dict["completion"],
            exact_match=result_dict["exact_match"],
            difference_ratio=result_dict["difference_ratio"],
            penalized_difference_ratio=result_dict["penalized_difference_ratio"],
            relative_difference_ratio=result_dict["relative_difference_ratio"],
            valid_output_grid=result_dict["valid_output_grid"],
            load_support_connected=result_dict["load_support_connected"],
            load_support_connected_force_directional=result_dict[
                "load_support_connected_force_directional"
            ],
            isolated_clusters_count=result_dict["isolated_clusters_count"],
            force_path_cost_average_efficiency_ratio=result_dict[
                "force_path_cost_average_efficiency_ratio"
            ],
            difficulty_score=result_dict["difficulty_score"],
            difficulty_weighted_difference_ratio=result_dict[
                "difficulty_weighted_difference_ratio"
            ],
            difficulty_weighted_relative_difference_ratio=result_dict[
                "difficulty_weighted_relative_difference_ratio"
            ],
            topology_score=result_dict.get("topology_score", 0.0),
            structural_efficiency=result_dict.get("structural_efficiency", 0.0),
            material_efficiency=result_dict.get("material_efficiency", 0.0),
            compliance_efficiency_vs_ground_truth=result_dict.get(
                "compliance_efficiency_vs_ground_truth", 0.0
            ),
            load_carrying=result_dict.get("load_carrying", False),
            compliance=result_dict.get("compliance"),
            volume_ratio=result_dict.get("volume_ratio", 0.0),
        )


def calculate_all_metrics(
    input_grid,
    output_grid,
    gt_grid,
    subject,
    prompt,
    ground_truth,
    output,
    gravity_dir=(1, 0),
):
    """Score one completion against its sample.

    Grids are nested lists of cell tokens.  ``gravity_dir`` is the direction
    force flows through this sample, which rotates with the sample itself.
    A completion whose shape or values are invalid is scored as a miss on
    every metric rather than raising, because an unusable answer is a real
    outcome the benchmark has to report.
    """
    valid_grid = get_grid_shape_and_value_validity(output_grid, gt_grid)

    difficulty_score = get_difficulty_score(
        input_grid=input_grid,
        gt_grid=gt_grid,
    )

    structural_metrics = {}

    exact_match = False
    difference_ratio = 0.0
    penalized_difference_ratio = 0.0
    relative_difference_ratio = 0.0
    load_support_connected = False
    load_support_connected_force_directional = False
    isolated_clusters_count = 0
    force_path_cost_average_efficiency_ratio = 0

    if valid_grid:
        exact_match = get_exact_match(output_grid, gt_grid)
        difference_ratio = get_difference_ratio(output_grid, gt_grid)
        penalized_difference_ratio = get_penalized_difference_ratio(
            output_grid, gt_grid
        )
        relative_difference_ratio = get_relative_difference_ratio(output_grid, gt_grid)

        load_support_connected = is_load_supported(output_grid)
        load_support_connected_force_directional = is_load_supported_force_directional(
            output_grid, gravity_dir=gravity_dir
        )
        isolated_clusters_count = get_isolated_clusters_count(output_grid)

        force_path_cost_average_efficiency_ratio = (
            get_force_path_cost_average_efficiency_ratio(
                output_grid, gt_grid, gravity_dir=gravity_dir
            )
        )

        # Dynamic evaluation: simulate the completion and re-optimise its mask.
        structural_metrics = get_structural_metrics(
            output_grid=output_grid,
            gt_grid=gt_grid,
            input_grid=input_grid,
            gravity_dir=gravity_dir,
        ).as_dict()

    result_dict = {
        "subject": subject,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "completion": output,
        "exact_match": exact_match,
        "difference_ratio": difference_ratio,
        "relative_difference_ratio": relative_difference_ratio,
        "penalized_difference_ratio": penalized_difference_ratio,
        "load_support_connected": load_support_connected,
        "load_support_connected_force_directional": load_support_connected_force_directional,
        "isolated_clusters_count": isolated_clusters_count,
        "force_path_cost_average_efficiency_ratio": force_path_cost_average_efficiency_ratio,
        "difficulty_score": difficulty_score,
        "difficulty_weighted_difference_ratio": difficulty_score * difference_ratio,
        "difficulty_weighted_relative_difference_ratio": difficulty_score
        * relative_difference_ratio,
        "valid_output_grid": valid_grid,
    }

    result_dict.update(structural_metrics)

    return result_dict


def reward_from_metrics(metrics):
    """The scalar reward for a scored completion.

    Zero for anything unusable, so an unparsable or misshapen grid is never
    worth more than a structure that at least stands up.
    """
    if not metrics.get("valid_output_grid"):
        return 0.0

    value = metrics.get(REWARD_METRIC)

    if value is None:
        return 0.0

    return float(value)


def aggregate_results(results: list[Result]) -> dict:

    total_exact_match = 0
    total_difference_ratio = 0.0
    total_penalized_difference_ratio = 0.0
    total_relative_difference_ratio = 0.0
    total_valid_output_grid = 0
    total_load_support_connected = 0
    total_load_support_connected_force_directional = 0
    total_isolated_clusters_count = 0
    total_force_path_cost_average_efficiency_ratio = 0.0
    total_difficulty_score = 0.0
    total_difficulty_weighted_difference_ratio = 0.0
    total_difficulty_weighted_relative_difference_ratio = 0.0

    for result in results:
        total_exact_match += int(result.exact_match)
        total_difference_ratio += result.difference_ratio
        total_penalized_difference_ratio += result.penalized_difference_ratio
        total_relative_difference_ratio += result.relative_difference_ratio
        total_valid_output_grid += int(result.valid_output_grid)
        total_load_support_connected += int(result.load_support_connected)
        total_load_support_connected_force_directional += int(
            result.load_support_connected_force_directional
        )
        total_isolated_clusters_count += result.isolated_clusters_count
        total_force_path_cost_average_efficiency_ratio += (
            result.force_path_cost_average_efficiency_ratio
        )
        total_difficulty_score += result.difficulty_score
        total_difficulty_weighted_difference_ratio += (
            result.difficulty_weighted_difference_ratio
        )
        total_difficulty_weighted_relative_difference_ratio += (
            result.difficulty_weighted_relative_difference_ratio
        )

    aggregated = {
        "total_exact_match": total_exact_match,
        "total_difference_ratio": total_difference_ratio,
        "total_penalized_difference_ratio": total_penalized_difference_ratio,
        "total_relative_difference_ratio": total_relative_difference_ratio,
        "total_valid_output_grid": total_valid_output_grid,
        "total_load_support_connected": total_load_support_connected,
        "total_load_support_connected_force_directional": total_load_support_connected_force_directional,
        "total_isolated_clusters_count": total_isolated_clusters_count,
        "total_force_path_cost_average_efficiency_ratio": total_force_path_cost_average_efficiency_ratio,
        "total_difficulty_score": total_difficulty_score,
        "total_difficulty_weighted_difference_ratio": total_difficulty_weighted_difference_ratio,
        "total_difficulty_weighted_relative_difference_ratio": total_difficulty_weighted_relative_difference_ratio,
    }

    for key in STRUCTURAL_METRIC_KEYS:
        aggregated[f"total_{key}"] = sum(
            float(getattr(result, key) or 0.0) for result in results
        )

    return aggregated

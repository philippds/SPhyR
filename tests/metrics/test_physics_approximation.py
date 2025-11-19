import pytest

from sphyr.metrics.physics_approximation import (
    get_total_force_path_cost_average_efficiency_ratio,
)


@pytest.mark.parametrize(
    "gt_grid, completion_grid, expected_ratio, gravity_dir",
    [
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            1.0,
            (1, 0),
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["1", "1", "0"],
                ["0", "S", "0"],
            ],
            1.0,
            (1, 0),
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["S", "0", "0"],
            ],
            [
                ["0", "L", "0"],
                ["1", "1", "0"],
                ["S", "0", "0"],
            ],
            0.8037,
            (1, 0),
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["S", "0", "0"],
            ],
            [
                ["0", "L", "0"],
                ["1", "0", "0"],
                ["S", "0", "0"],
            ],
            0.8037,
            (1, 0),
        ),
        (
            [
                ["0", "0", "L"],
                ["0", "1", "0"],
                ["S", "0", "0"],
            ],
            [
                ["0", "1", "L"],
                ["1", "0", "0"],
                ["S", "0", "0"],
            ],
            0.7724,
            (1, 0),
        ),
    ],
)
def test_get_total_force_path_cost_average_efficiency_ratio(
    gt_grid, completion_grid, gravity_dir, expected_ratio
):
    ratio = get_total_force_path_cost_average_efficiency_ratio(
        completion_grid, gt_grid, gravity_dir
    )
    assert ratio == pytest.approx(
        expected_ratio, rel=1e-3
    ), f"Expected {expected_ratio}, got {ratio}"

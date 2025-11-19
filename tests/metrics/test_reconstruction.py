import pytest

from sphyr.run_eval import get_exact_match
from sphyr.metrics.reconstruction import (
    get_difference_ratio,
    get_penalized_difference_ratio,
    get_relative_difference_ratio,
)


@pytest.mark.parametrize(
    "gt_grid, completion_grid, expected_exact_match",
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
            True,
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
            False,
        ),
    ],
)
def test_get_exact_match(gt_grid, completion_grid, expected_exact_match):
    exact_match = get_exact_match(completion_grid, gt_grid)
    assert (
        exact_match == expected_exact_match
    ), f"Expected {expected_exact_match}, got {exact_match}"


@pytest.mark.parametrize(
    "gt_grid, completion_grid, expected_difference_ratio",
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
            0.0,
        ),
        (
            [
                ["0", "L", "0"],
                ["1", "1", "0"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            0.5,
        ),
    ],
)
def test_get_difference_ratio(gt_grid, completion_grid, expected_difference_ratio):
    difference_ratio = get_difference_ratio(completion_grid, gt_grid)
    assert (
        difference_ratio == expected_difference_ratio
    ), f"Expected {expected_difference_ratio}, got {difference_ratio}"


@pytest.mark.parametrize(
    "gt_grid, completion_grid, expected_relative_difference_ratio",
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
        ),
        (
            [
                ["0", "L", "0"],
                ["1", "1", "1"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            1 / 3,
        ),
        (
            [
                ["0", "L", "0"],
                ["0.8", "1", "0.8"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0.4", "0.5", "0.4"],
                ["0", "S", "0"],
            ],
            0.5,
        ),
        (
            [
                ["0", "L", "0"],
                ["0.8", "1", "0.8"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0.4", "1.5", "0.4"],
                ["0", "S", "0"],
            ],
            0.5,
        ),
        (
            [
                ["0", "L", "0"],
                ["0.8", "1", "0.8"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0.4", "2", "0.4"],
                ["0", "S", "0"],
            ],
            0.3076,
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["1", "1", "1"],
                ["0", "S", "0"],
            ],
            -1.0,
        ),
    ],
)
def test_get_relative_difference_ratio(
    gt_grid, completion_grid, expected_relative_difference_ratio
):
    difference_ratio = get_relative_difference_ratio(completion_grid, gt_grid)
    assert difference_ratio == pytest.approx(
        expected_relative_difference_ratio, rel=1e-3
    ), f"Expected {expected_relative_difference_ratio}, got {difference_ratio}"


@pytest.mark.parametrize(
    "gt_grid, completion_grid, expected_penalized_difference_ratio",
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
        ),
        (
            [
                ["0", "L", "0"],
                ["1", "1", "1"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            1 / 3,
        ),
        (
            [
                ["0", "L", "0"],
                ["0.8", "1", "0.8"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0.4", "0.5", "0.4"],
                ["0", "S", "0"],
            ],
            0.5,
        ),
        (
            [
                ["0", "L", "0"],
                ["0.8", "1", "0.8"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0.4", "1.5", "0.4"],
                ["0", "S", "0"],
            ],
            0.5,
        ),
        (
            [
                ["0", "L", "0"],
                ["0.8", "1", "0.8"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["0.4", "2", "0.4"],
                ["0", "S", "0"],
            ],
            0.3076,
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            [
                ["0", "L", "0"],
                ["1", "1", "1"],
                ["0", "S", "0"],
            ],
            -1.0,
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            [
                ["0", "1", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            -2.0,
        ),
    ],
)
def test_get_penalized_difference_ratio(
    gt_grid, completion_grid, expected_penalized_difference_ratio
):
    difference_ratio = get_penalized_difference_ratio(completion_grid, gt_grid)
    assert difference_ratio == pytest.approx(
        expected_penalized_difference_ratio, rel=1e-3
    ), f"Expected {expected_penalized_difference_ratio}, got {difference_ratio}"

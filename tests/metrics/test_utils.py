import pytest
from sphyr.metrics.utils import get_grid_shape_and_value_validity


@pytest.mark.parametrize(
    "completion_grid, expected_validity",
    [
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            True,
        ),
        (
            [
                ["0", "X", "0"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            False,
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "1", "0"],
                ["0", "P", "0"],
            ],
            False,
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "-1", "0"],
                ["0", "S", "0"],
            ],
            False,
        ),
        (
            [
                ["0", "L", "0"],
                ["0", "2", "0"],
                ["0", "S", "0"],
            ],
            False,
        ),
    ],
)
def test_get_grid_shape_and_value_validity(completion_grid, expected_validity):

    gt_grid = [
        ["0", "L", "0"],
        ["0", "1", "0"],
        ["0", "S", "0"],
    ]

    validity = get_grid_shape_and_value_validity(completion_grid, gt_grid)
    assert (
        validity == expected_validity
    ), f"Expected {expected_validity}, got {validity}"

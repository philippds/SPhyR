import pytest

from sphyr.metrics.topology import (
    get_difficulty_score,
    get_isolated_clusters_count,
    is_load_supported,
    is_load_supported_force_directional,
)


@pytest.mark.parametrize(
    "input_grid, gt_grid, completion_grid, expected_difficulty_score",
    [
        (
            [
                ["0", "L", "0"],
                ["0", "V", "0"],
                ["0", "S", "0"],
            ],
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
            2.0,
        ),
        (
            [
                ["0", "L", "0"],
                ["V", "1", "0"],
                ["0", "S", "0"],
            ],
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
            3.0,
        ),
        (
            [
                ["0", "L", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "1", "0", "V", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "S", "0", "0", "0"],
            ],
            [
                ["0", "L", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "S", "0", "0", "0"],
            ],
            [
                ["0", "L", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "1", "0", "0", "0"],
                ["0", "S", "0", "0", "0"],
            ],
            1.0,
        ),
        (
            [
                ["0", "L", "L", "L", "0"],
                ["0", "1", "1", "1", "0"],
                ["0", "1", "V", "1", "0"],
                ["0", "1", "1", "1", "0"],
                ["0", "S", "S", "S", "0"],
            ],
            [
                ["0", "L", "L", "L", "0"],
                ["0", "1", "1", "0", "0"],
                ["0", "1", "0", "1", "0"],
                ["0", "1", "1", "0", "0"],
                ["0", "S", "S", "S", "0"],
            ],
            [
                ["0", "L", "L", "L", "0"],
                ["0", "1", "1", "0", "0"],
                ["0", "1", "0", "1", "0"],
                ["0", "1", "1", "0", "0"],
                ["0", "S", "S", "S", "0"],
            ],
            3.0,
        ),
    ],
)
def test_get_difficulty_score(
    input_grid, gt_grid, completion_grid, expected_difficulty_score
):
    difficulty_score = get_difficulty_score(input_grid, completion_grid, gt_grid)
    assert (
        difficulty_score == expected_difficulty_score
    ), f"Expected {expected_difficulty_score}, got {difficulty_score}"


@pytest.mark.parametrize(
    "output_grid, expected_load_support_connected",
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
                ["0", "L", "0"],
                ["1", "1", "0"],
                ["0", "S", "S"],
            ],
            True,
        ),
        (
            [
                ["0", "L", "L"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            True,
        ),
        (
            [
                ["0", "L", "L"],
                ["1", "0", "0"],
                ["0", "S", "0"],
            ],
            True,
        ),
        (
            [
                ["0", "0", "L"],
                ["1", "0", "0"],
                ["0", "S", "0"],
            ],
            False,
        ),
    ],
)
def test_is_load_supported(output_grid, expected_load_support_connected):
    load_support_connected = is_load_supported(output_grid)
    assert (
        load_support_connected == expected_load_support_connected
    ), f"Expected {expected_load_support_connected}, got {load_support_connected}"


@pytest.mark.parametrize(
    "output_grid, expected_load_support_connected",
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
                ["0", "L", "0"],
                ["1", "1", "0"],
                ["0", "S", "S"],
            ],
            True,
        ),
        (
            [
                ["0", "L", "L"],
                ["0", "1", "0"],
                ["0", "S", "0"],
            ],
            True,
        ),
        (
            [
                ["0", "L", "L"],
                ["1", "0", "0"],
                ["0", "S", "0"],
            ],
            True,
        ),
        (
            [
                ["0", "0", "L"],
                ["1", "0", "0"],
                ["0", "S", "0"],
            ],
            False,
        ),
        (
            [
                ["1", "1", "1", "0", "1", "L"],
                ["1", "0", "1", "0", "1", "0"],
                ["1", "0", "1", "1", "1", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["S", "0", "0", "0", "0", "0"],
            ],
            False,
        ),
    ],
)
def test_is_load_supported_force_directional(
    output_grid, expected_load_support_connected
):
    load_support_connected = is_load_supported_force_directional(output_grid)
    assert (
        load_support_connected == expected_load_support_connected
    ), f"Expected {expected_load_support_connected}, got {load_support_connected}"


@pytest.mark.parametrize(
    "output_grid, expected_isolated_clusters_count",
    [
        (
            [
                ["L", "0", "0", "0", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["1", "0", "1", "0", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["S", "0", "0", "0", "0", "0"],
            ],
            1,
        ),
        (
            [
                ["L", "0", "0", "0", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["1", "0", "1", "1", "0", "0"],
                ["1", "0", "0", "1", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["S", "0", "0", "0", "0", "0"],
            ],
            1,
        ),
        (
            [
                ["L", "0", "0", "0", "0", "1"],
                ["1", "0", "0", "0", "0", "1"],
                ["1", "0", "1", "1", "0", "1"],
                ["1", "0", "0", "1", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["S", "0", "0", "0", "0", "0"],
            ],
            2,
        ),
        (
            [
                ["L", "0", "0", "1", "0", "0"],
                ["1", "0", "0", "0", "0", "0"],
                ["1", "0", "1", "1", "0", "0"],
                ["1", "0", "0", "1", "0", "1"],
                ["1", "0", "0", "0", "0", "0"],
                ["S", "0", "0", "0", "0", "0"],
            ],
            3,
        ),
    ],
)
def test_get_isolated_clusters_count(output_grid, expected_isolated_clusters_count):
    isolated_clusters_count = get_isolated_clusters_count(output_grid)
    assert (
        isolated_clusters_count == expected_isolated_clusters_count
    ), f"Expected {expected_isolated_clusters_count}, got {isolated_clusters_count}"

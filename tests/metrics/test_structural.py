import json

import pytest

from sphyr.metrics.structural import get_structural_metrics

INPUT_GRID = [
    ["L", "L", "0", "0"],
    ["V", "V", "V", "V"],
    ["V", "V", "V", "V"],
    ["S", "S", "0", "0"],
]

GROUND_TRUTH = [
    ["L", "L", "0", "0"],
    ["1", "1", "0", "0"],
    ["1", "1", "0", "0"],
    ["S", "S", "0", "0"],
]

DISCONNECTED = [
    ["L", "L", "0", "0"],
    ["0", "0", "0", "0"],
    ["1", "1", "0", "0"],
    ["S", "S", "0", "0"],
]

SOLID = [
    ["L", "L", "1", "1"],
    ["1", "1", "1", "1"],
    ["1", "1", "1", "1"],
    ["S", "S", "1", "1"],
]

EMPTY = [
    ["L", "L", "0", "0"],
    ["0", "0", "0", "0"],
    ["0", "0", "0", "0"],
    ["S", "S", "0", "0"],
]


def score(completion, **kwargs):
    return get_structural_metrics(
        output_grid=completion,
        gt_grid=GROUND_TRUTH,
        input_grid=INPUT_GRID,
        **kwargs,
    )


def test_the_dataset_answer_carries_its_load():
    metrics = score(GROUND_TRUTH)

    assert metrics.physically_valid
    assert metrics.load_carrying
    assert metrics.topology_score > 0.5
    assert metrics.compliance_efficiency_vs_ground_truth == pytest.approx(1.0)


def test_a_severed_load_path_scores_zero():
    metrics = score(DISCONNECTED)

    assert metrics.physically_valid
    assert not metrics.load_carrying
    assert metrics.topology_score == 0.0
    assert metrics.structural_efficiency == 0.0


def test_an_empty_completion_scores_zero():
    assert score(EMPTY).topology_score == 0.0


def test_filling_everything_solid_is_penalised_for_material():
    metrics = score(SOLID)

    # Stiffest possible answer, so the stiffness half of the score saturates.
    assert metrics.structural_efficiency == pytest.approx(1.0)
    # But it spends far more material than the reference answer.
    assert metrics.material_efficiency < 0.6
    assert metrics.topology_score < score(GROUND_TRUTH).topology_score


def test_an_alternative_load_path_is_not_punished_for_being_different():
    """The point of the dynamic metric: a different but equally good answer.

    Mirroring the reference column onto the other pair of columns changes every
    masked cell, yet carries the load through the same amount of material.
    """
    mirrored_ground_truth = [
        ["0", "0", "L", "L"],
        ["0", "0", "1", "1"],
        ["0", "0", "1", "1"],
        ["0", "0", "S", "S"],
    ]
    mirrored_input = [
        ["0", "0", "L", "L"],
        ["V", "V", "V", "V"],
        ["V", "V", "V", "V"],
        ["0", "0", "S", "S"],
    ]
    shifted_completion = [
        ["0", "0", "L", "L"],
        ["0", "1", "1", "1"],
        ["0", "1", "1", "1"],
        ["0", "0", "S", "S"],
    ]

    metrics = get_structural_metrics(
        output_grid=shifted_completion,
        gt_grid=mirrored_ground_truth,
        input_grid=mirrored_input,
    )

    assert metrics.load_carrying
    assert metrics.topology_score > 0.5


def test_a_wrong_shape_completion_is_invalid():
    metrics = score([["L", "L"], ["S", "S"]])

    assert not metrics.physically_valid
    assert metrics.topology_score == 0.0


def test_unfilled_cells_are_counted():
    unfinished = [row[:] for row in GROUND_TRUTH]
    unfinished[1][0] = "V"

    assert score(unfinished).unfilled_cells == 1


def test_boundary_conditions_come_from_the_reference_not_the_completion():
    """Moving the supports in the completion must not create a free win."""
    relocated = [
        ["L", "L", "0", "0"],
        ["0", "0", "0", "0"],
        ["S", "S", "0", "0"],
        ["S", "S", "0", "0"],
    ]

    metrics = score(relocated)

    # The real supports are still the bottom row, so the gap above them leaves
    # the load unsupported no matter what the completion calls those cells.
    assert not metrics.load_carrying


def test_design_efficiency_is_only_computed_on_request():
    assert score(GROUND_TRUTH).design_efficiency is None

    metrics = score(GROUND_TRUTH, include_own_budget_reference=True)
    assert metrics.design_efficiency > 0.0


def test_a_severed_load_path_is_orders_of_magnitude_softer():
    """Void keeps a token stiffness, so a broken design is soft, not singular."""
    metrics = score(DISCONNECTED)

    assert metrics.compliance > 1e3 * score(GROUND_TRUTH).compliance
    assert not metrics.load_carrying


def test_metrics_serialise_to_valid_json():
    values = score(DISCONNECTED).as_dict()

    assert all(value != float("inf") for value in values.values())
    assert json.loads(json.dumps(values))["load_carrying"] is False

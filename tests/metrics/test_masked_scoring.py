"""The submitted design may change only the cells exposed by the mask."""

import copy

import pytest

from sphyr.metrics import structural
from sphyr.metrics.utils import complete_masked_grid
from sphyr.scoring import Result, aggregate_results, calculate_all_metrics


GIVEN = [["L", "0", "0"], ["V", "V", "0"], ["S", "0", "0"]]
ANSWER = [["L", "0", "0"], ["1", "0", "0"], ["S", "0", "0"]]


def score(grid, **kwargs):
    return calculate_all_metrics(
        input_grid=GIVEN,
        output_grid=grid,
        gt_grid=ANSWER,
        subject="test_easy",
        prompt="original task",
        ground_truth="\n".join(" ".join(row) for row in ANSWER),
        output="\n".join(" ".join(row) for row in grid),
        **kwargs,
    )


def test_visible_material_and_boundary_changes_cannot_change_any_design_metric():
    tampered = copy.deepcopy(ANSWER)
    tampered[0] = ["0", "1", "S"]
    tampered[2][0] = "0"
    raw = copy.deepcopy(tampered)
    expected = score(ANSWER)
    actual = score(tampered)
    excluded = {
        "completion", "visible_cell_changes", "visible_numeric_cell_changes",
        "visible_boundary_cell_changes", "visible_cells_preserved",
    }
    assert {k: v for k, v in actual.items() if k not in excluded} == {
        k: v for k, v in expected.items() if k not in excluded
    }
    assert actual["completion"] != expected["completion"]
    assert tampered == raw
    assert actual["visible_cell_changes"] == 4
    assert actual["visible_numeric_cell_changes"] == 2
    assert actual["visible_boundary_cell_changes"] == 2
    assert not actual["visible_cells_preserved"]
    assert actual["load_support_connected"]
    assert actual["exact_match"]
    totals = aggregate_results([Result.from_dict(actual)])
    assert totals["total_visible_cell_changes"] == 4
    assert totals["total_visible_cells_preserved"] == 0


def test_masked_errors_are_not_replaced_with_the_reference_answer():
    severed = copy.deepcopy(ANSWER)
    severed[1][0] = "0"
    metrics = score(severed)
    assert not metrics["exact_match"]
    assert not metrics["load_carrying"]
    assert metrics["topology_score"] == 0


def test_invalid_grids_cannot_be_repaired_into_valid_answers():
    metrics = score([["1", "1"]])
    assert not metrics["valid_output_grid"]
    assert metrics["topology_score"] == 0
    assert metrics["scored_grid"] == ""
    with pytest.raises(ValueError, match="shape"):
        complete_masked_grid(GIVEN, [["1"]])


def test_equivalent_numeric_spellings_do_not_count_as_copying_errors():
    grid = copy.deepcopy(ANSWER)
    grid[0][1] = "0.0"
    assert score(grid)["visible_cells_preserved"]


def test_stored_reference_reuses_only_the_unchanged_reference_not_prediction_scores(monkeypatch):
    reference = score(ANSWER)["optimal_compliance"]

    def unexpected_optimization(*args, **kwargs):
        raise AssertionError("the unchanged reference should be reused")

    monkeypatch.setattr(structural, "get_reference_optimum", unexpected_optimization)
    assert score(ANSWER, reference_compliance=reference)["topology_score"] > 0
    severed = copy.deepcopy(ANSWER)
    severed[1][0] = "0"
    assert score(severed, reference_compliance=reference)["topology_score"] == 0
    with pytest.raises(ValueError, match="finite and positive"):
        score(ANSWER, reference_compliance=-1)


def test_direct_structural_scoring_also_preserves_given_material():
    tampered = copy.deepcopy(ANSWER)
    tampered[0][1:] = ["1", "1"]
    a = structural.get_structural_metrics(ANSWER, ANSWER, GIVEN)
    b = structural.get_structural_metrics(tampered, ANSWER, GIVEN)
    assert a.as_dict() == b.as_dict()

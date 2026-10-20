"""Offline migration preserves raw evidence and agrees with live scoring."""

import json

import pytest

from sphyr.rescore import rescore_stored_results
from sphyr.scoring import calculate_all_metrics


GIVEN = [["L", "0", "0"], ["V", "V", "0"], ["S", "0", "0"]]
ANSWER = [["L", "0", "0"], ["1", "0", "0"], ["S", "0", "0"]]


def render(grid):
    return "\n".join(" ".join(row) for row in grid)


def test_rescore_preserves_raw_answers_backs_up_and_is_idempotent(tmp_path):
    root = tmp_path / "results"
    model = root / "test-model"
    model.mkdir(parents=True)
    tampered = [["0", "1", "1"], ["1", "0", "0"], ["0", "0", "0"]]
    original = calculate_all_metrics(
        GIVEN, ANSWER, ANSWER, "full_easy", render(GIVEN), render(ANSWER), render(tampered)
    )
    original["topology_score"] = 0.0
    original["exact_match"] = False
    original["score"] = 999
    path = model / "full_easy_results.json"
    contents = json.dumps([original]).encode()
    path.write_bytes(contents)
    backup = tmp_path / "backup"

    rescore_stored_results(root, backup_root=backup)
    result = json.loads(path.read_text())[0]
    assert result["completion"] == render(tampered)
    assert result["prompt"] == original["prompt"]
    assert result["ground_truth"] == original["ground_truth"]
    assert result["scored_grid"] == render(ANSWER)
    assert result["exact_match"]
    assert result["visible_cell_changes"] == 4
    assert "score" not in result
    assert (backup / "test-model" / path.name).read_bytes() == contents
    aggregate = json.loads((model / "full_easy_aggregated_results.json").read_text())
    assert aggregate["total_exact_match"] == 1
    assert aggregate["total_visible_cell_changes"] == 4

    rescored_bytes = path.read_bytes()
    rescore_stored_results(root, backup_root=backup)
    assert path.read_bytes() == rescored_bytes
    assert (backup / "test-model" / path.name).read_bytes() == contents


def test_timing_run_never_modifies_results_or_creates_backups(tmp_path):
    root = tmp_path / "results"
    model = root / "test-model"
    model.mkdir(parents=True)
    path = model / "full_easy_results.json"
    path.write_text(json.dumps([{
        "subject": "full_easy", "prompt": render(GIVEN),
        "ground_truth": render(ANSWER), "completion": render(ANSWER),
    }]))
    original = path.read_bytes()
    backup = tmp_path / "backup"
    rescore_stored_results(root, dry_run=True, limit_records=1, backup_root=backup)
    assert path.read_bytes() == original
    assert not backup.exists()
    with pytest.raises(ValueError, match="dry-run"):
        rescore_stored_results(root, limit_records=1)


def test_invalid_answer_clears_stale_physics_instead_of_retaining_old_score(tmp_path):
    root = tmp_path / "results"
    model = root / "test-model"
    model.mkdir(parents=True)
    path = model / "full_easy_results.json"
    path.write_text(json.dumps([{
        "subject": "full_easy", "prompt": render(GIVEN),
        "ground_truth": render(ANSWER), "completion": "1 1",
        "topology_score": 1, "load_carrying": True,
    }]))
    rescore_stored_results(root)
    result = json.loads(path.read_text())[0]
    assert not result["valid_output_grid"]
    assert not result["load_carrying"]
    assert result["topology_score"] == 0

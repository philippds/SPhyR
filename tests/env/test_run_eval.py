"""The runner loop: reset, act, step, write a result file the paper can read."""

import json

import pytest

from sphyr import run_eval
from sphyr.policies import ConstantFillPolicy
from sphyr.scoring import Result, aggregate_results


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS_ROOT", str(tmp_path))
    return tmp_path


def test_evaluate_subject_writes_a_complete_result_file(results_root):
    with run_eval.open_session(sample_count=2) as session:
        run_eval.evaluate_subject(
            session=session,
            policy=ConstantFillPolicy("1"),
            model="test-model",
            subject="full_easy",
            sample_count=2,
        )

    results_file = results_root / "test-model" / "full_easy_results.json"
    aggregated_file = results_root / "test-model" / "full_easy_aggregated_results.json"

    assert results_file.exists()
    assert aggregated_file.exists()

    records = json.loads(results_file.read_text())
    assert len(records) == 2

    for record in records:
        assert record["subject"] == "full_easy"
        assert record["prompt"]
        assert record["completion"]
        assert record["ground_truth"]
        assert record["valid_output_grid"] is True
        # Filling solid always carries the load and always overspends.
        assert record["load_carrying"] is True
        assert record["volume_ratio"] > 1.0

    # The stored records have to survive the round trip the paper scripts make.
    aggregated = aggregate_results([Result.from_dict(r) for r in records])
    assert aggregated == json.loads(aggregated_file.read_text())
    assert aggregated["total_valid_output_grid"] == 2


def test_evaluate_subject_resumes_from_stored_results(results_root, capsys):
    with run_eval.open_session(sample_count=2) as session:
        run_eval.evaluate_subject(
            session=session,
            policy=ConstantFillPolicy("1"),
            model="test-model",
            subject="full_easy",
            sample_count=2,
        )

        def refuse(observation):
            raise AssertionError("a stored sample should not be asked for again")

        run_eval.evaluate_subject(
            session=session,
            policy=refuse,
            model="test-model",
            subject="full_easy",
            sample_count=2,
        )

    assert "Existing results: 2" in capsys.readouterr().out


def test_a_failing_policy_leaves_the_run_resumable(results_root):
    def fail(observation):
        raise RuntimeError("provider is down")

    with run_eval.open_session(sample_count=2) as session:
        run_eval.evaluate_subject(
            session=session,
            policy=fail,
            model="test-model",
            subject="full_easy",
            sample_count=2,
        )

    # Nothing scored, nothing aggregated, and nothing that would be mistaken
    # for a finished run.
    assert not (results_root / "test-model" / "full_easy_aggregated_results.json").exists()


def test_rescore_recomputes_metrics_in_place(results_root):
    with run_eval.open_session(sample_count=2) as session:
        run_eval.evaluate_subject(
            session=session,
            policy=ConstantFillPolicy("1"),
            model="test-model",
            subject="full_easy",
            sample_count=2,
        )

    results_file = results_root / "test-model" / "full_easy_results.json"
    original = json.loads(results_file.read_text())

    # Simulate a stale file from an older revision of the benchmark.
    stale = [dict(record, topology_score=0.0, score=1.23) for record in original]
    results_file.write_text(json.dumps(stale))

    run_eval.rescore_stored_results(str(results_root))

    rescored = json.loads(results_file.read_text())

    assert [r["topology_score"] for r in rescored] == [
        r["topology_score"] for r in original
    ]
    assert all("score" not in r for r in rescored)

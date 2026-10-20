"""The generation load case must survive prompting, rotation and replay."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from sphyr.dataset_creation.generate_v2 import Sample as GeneratedSample, write_subjects
from sphyr.dataset_creation.recover_v2_directions import recover_candidates, verify_candidate
from sphyr.evaluate_dynamic import rescore_record
from sphyr.physics.analysis import analyze_structure
from sphyr.physics.boundary_conditions import get_densities_from_grid, get_problem_from_grid
from sphyr.tasks import (
    DATASET_VERSIONS,
    build_sample, load_subject, replay_samples, rotate_load_direction,
    str_to_grid, validate_load_direction,
)
from sphyr_env import SPhyRAction
from sphyr_env.server.sphyr_environment import SPhyREnvironment


@pytest.mark.parametrize("direction", [None, [], [0, 0], [1], [1, 2, 3],
                                      [float("nan"), 1], [float("inf"), 0], [True, 0], ["1", 0]])
def test_invalid_force_metadata_is_rejected(direction):
    with pytest.raises(ValueError, match="load_direction"):
        validate_load_direction(direction)


@pytest.mark.parametrize("direction", [(1, 0), (-1, 0), (0, 1), (0, -1),
                                      (1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_all_directions_rotate_in_grid_coordinates(direction):
    dr, dc = direction
    assert rotate_load_direction(direction, 1) == (dc, -dr)
    assert rotate_load_direction(direction, 4) == direction
    assert rotate_load_direction(direction, 2) == (-dr, -dc)


def test_writer_serializes_generation_force_for_every_subject(tmp_path):
    problem = get_problem_from_grid([["L", "0"], ["0", "S"]], gravity_dir=(-1, 1))
    sample = GeneratedSample(problem, ((1, 0), (0, 1)), ((1., 0.), (0., 1.)))
    write_subjects([sample], tmp_path)
    for path in tmp_path.glob("*.jsonl"):
        assert json.loads(path.read_text())["load_direction"] == [-1, 1]
    with pytest.raises(FileExistsError, match="overwrite"):
        write_subjects([sample], tmp_path)


def test_prompt_and_few_shot_examples_have_their_own_rotated_force():
    a = {"input_grid": [["L", "V"], ["0", "S"]],
         "ground_truth": [["L", "1"], ["0", "S"]], "load_direction": [1, 1]}
    b = {"input_grid": [["L", "0"], ["V", "S"]],
         "ground_truth": [["L", "0"], ["1", "S"]], "load_direction": [-1, 0]}
    sample = build_sample([a, b], a, 0, "full_easy", rotation_count=1, few_shot_count=1)
    assert sample.load_direction == (1, -1)
    assert "(1, -1)" in sample.prompt
    assert "(0, 1)" in sample.prompt
    assert sample.prompt.index("(1, -1)") > sample.prompt.index("(0, 1)")


def test_explicit_force_reaches_simulation_feedback_results_and_rescore():
    records = load_subject("full_easy", version="v2-force")[:1]
    with patch("sphyr_env.server.sphyr_environment.load_subject", return_value=records):
        env = SPhyREnvironment(dataset_version="v2-force", sample_count=1, feedback_rounds=1)
        observation = env.reset(subject="full_easy")
        assert observation.load_direction == (1, 1)
        assert "(1, 1)" in observation.prompt
        answer = env._current.ground_truth
        scored = env.step(SPhyRAction(grid=answer))
    grid = str_to_grid(answer)
    expected = analyze_structure(get_problem_from_grid(grid, gravity_dir=(1, 1)),
                                 get_densities_from_grid(grid)[0]).compliance
    vertical = analyze_structure(get_problem_from_grid(grid), get_densities_from_grid(grid)[0]).compliance
    assert scored.metrics["compliance"] == pytest.approx(expected)
    assert not np.isclose(expected, vertical)
    assert "(1, 1)" in scored.prompt
    assert scored.metrics["load_direction_source"] == "explicit"
    original = dict(scored.metrics)
    rescored = rescore_record(dict(original), gravity_dir=(1, 0))
    assert rescored["compliance"] == pytest.approx(expected)
    assert rescored["topology_score"] == pytest.approx(original["topology_score"])
    for template in (None, "{GRID}"):
        replay = replay_samples("full_easy", [original], prompt_template=template)[0]
        assert replay.load_direction == (1, 1)
        assert replay.load_direction_explicit
        assert "(1, 1)" in replay.prompt


def test_legacy_prompts_and_loads_keep_their_original_meaning():
    record = load_subject("full_easy", version="v2")[0]
    sample = build_sample([record], record, 0, "full_easy", rotation_count=1)
    assert not sample.load_direction_explicit
    assert sample.load_direction == (0, -1)
    assert "Applied force at every" not in sample.prompt
    result = {"prompt": sample.prompt, "ground_truth": sample.ground_truth,
              "load_direction": [0, -1], "load_direction_source": "legacy-rotation"}
    replay = replay_samples("full_easy", [result], prompt_template="{GRID}")[0]
    assert replay.load_direction == (0, -1)
    assert "Applied force at every" not in replay.prompt


def test_recovery_verifies_answers_and_refuses_an_incorrect_force():
    easy = load_subject("full_easy", version="v2")[:1]
    hard = load_subject("full_hard", version="v2")[:1]
    _, problem = recover_candidates(easy)[0]
    assert problem.load_direction == (1, 1)
    assert verify_candidate((problem, easy[0]["ground_truth"], hard[0]["ground_truth"]))
    from dataclasses import replace
    with pytest.raises(ValueError, match="differ"):
        verify_candidate((replace(problem, load_direction=(1, 0)),
                          easy[0]["ground_truth"], hard[0]["ground_truth"]))


@pytest.mark.parametrize("version", ["v2", "v2-20"])
def test_corrected_bundled_tasks_only_add_verified_force_metadata(version):
    import hashlib
    from sphyr.dataset_creation.generate_v2 import LOAD_DIRECTIONS
    corrected = DATASET_VERSIONS[version + "-force"]
    provenance = json.loads((corrected / "provenance.json").read_text())
    assert provenance["verified_samples"] == 300
    for filename, expected_hash in provenance["source_sha256"].items():
        original_bytes = (DATASET_VERSIONS[version] / filename).read_bytes()
        assert hashlib.sha256(original_bytes).hexdigest() == expected_hash
        original = [json.loads(line) for line in original_bytes.decode().splitlines()]
        updated = [json.loads(line) for line in (corrected / filename).read_text().splitlines()]
        assert len(original) == len(updated) == 300
        directions = {tuple(row["load_direction"]) for row in updated}
        assert directions == set(LOAD_DIRECTIONS)
        assert [{k: v for k, v in row.items() if k != "load_direction"} for row in updated] == original


def test_corrected_dataset_rejects_legacy_replay():
    record = load_subject("full_easy", version="v2")[0]
    legacy = build_sample([record], record, 0, "full_easy")
    stored = [{"prompt": legacy.prompt, "ground_truth": legacy.ground_truth}]
    with patch("sphyr_env.server.sphyr_environment.published_records", return_value=stored):
        env = SPhyREnvironment(dataset_version="v2-force", replay_from="legacy")
        with pytest.raises(ValueError, match="cannot replay legacy"):
            env.reset(subject="full_easy")

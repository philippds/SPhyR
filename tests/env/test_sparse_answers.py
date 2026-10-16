"""The sparse answer format: the solid masked cells as a list, not a grid."""

import pytest

from sphyr.metrics.utils import extract_grid_from_text, parse_sparse_answer
from sphyr.tasks import (
    describe_masked_cells,
    replay_samples,
    resolve_prompt_template,
    str_to_grid,
)
from sphyr_env import SPhyRAction
from sphyr_env.server.sphyr_environment import SPhyREnvironment


@pytest.fixture(scope="module")
def env():
    return SPhyREnvironment(sample_count=3, prompt_style="sparse")


def solid_masked_cells(input_text, answer_text, with_values=False):
    """The sparse answer equivalent to a full grid answer."""
    lines = []
    for r, (masked_row, answer_row) in enumerate(
        zip(str_to_grid(input_text), str_to_grid(answer_text))
    ):
        for c, (masked, value) in enumerate(zip(masked_row, answer_row)):
            if masked == "V" and float(value) > 0:
                lines.append(f"{r},{c},{value}" if with_values else f"{r},{c}")
    return "\n".join(lines)


def test_the_prompt_keeps_the_grid_parsable(env):
    observation = env.reset(subject="10_random_cell_easy", sample_index=0)

    assert "Do not reproduce the grid" in observation.prompt
    assert describe_masked_cells(observation.grid) in observation.prompt
    # Stored records are rescored by parsing the grid back out of the prompt,
    # so the masked-cell list must not be mistaken for one.
    assert extract_grid_from_text(observation.prompt) == str_to_grid(observation.grid)


def test_listing_the_answers_solid_cells_scores_as_the_answer():
    # Two fresh environments, so both draw the same sample: the sample order
    # depends on everything an environment's RNG has served before.
    sparse_env = SPhyREnvironment(sample_count=3, prompt_style="sparse")
    grid_env = SPhyREnvironment(sample_count=3)

    task = sparse_env.reset(subject="full_easy", sample_index=0)
    answer = sparse_env._current.ground_truth
    sparse = sparse_env.step(SPhyRAction(grid=solid_masked_cells(task.grid, answer)))

    grid_env.reset(subject="full_easy", sample_index=0)
    assert grid_env._current.ground_truth == answer
    full = grid_env.step(SPhyRAction(grid=answer))

    assert sparse.valid_grid is True
    assert sparse.metrics["exact_match"] is True
    assert sparse.metrics["answer_format"] == "sparse"
    assert sparse.reward == pytest.approx(full.reward)


def test_density_answers_carry_their_values(env):
    task = env.reset(subject="full_hard", sample_index=0)
    answer = env._current.ground_truth

    scored = env.step(
        SPhyRAction(grid=solid_masked_cells(task.grid, answer, with_values=True))
    )

    assert scored.valid_grid is True
    assert scored.metrics["exact_match"] is True


def test_entries_outside_the_mask_are_dropped_not_fatal():
    masked = [["L", "V"], ["V", "S"]]

    grid, info = parse_sparse_answer("0,1\n0,0\n9,9\nsome prose 1,1", masked)

    assert grid == [["L", "1"], ["0", "S"]]
    assert info == {"sparse_entries": 1, "sparse_rejected_entries": 2}


def test_none_is_an_empty_structure_and_silence_is_unreadable():
    masked = [["L", "V"], ["V", "S"]]

    assert parse_sparse_answer("NONE", masked)[0] == [["L", "0"], ["0", "S"]]
    assert parse_sparse_answer("I am not sure.", masked)[0] == []


def test_a_density_entry_needs_a_value():
    masked = [["L", "V"], ["V", "S"]]

    grid, info = parse_sparse_answer("0,1,0.7\n1,0", masked, density=True)

    assert grid == [["L", "0.7"], ["0.0", "S"]]
    assert info["sparse_rejected_entries"] == 1


def test_masked_runs_are_collapsed():
    assert describe_masked_cells("V V V 1\n1 V 1 V") == "row 0: 0-2\nrow 1: 1, 3"


def test_a_replay_can_pose_the_stored_samples_sparsely():
    prompt = "Complete this grid:\n\n0 V\nL S\n"
    record = {"prompt": prompt, "ground_truth": "0 1\nL S"}

    stored = replay_samples("1_random_cell_easy", [record])[0]

    reposed = replay_samples(
        "1_random_cell_easy",
        [record],
        prompt_template=resolve_prompt_template("sparse"),
        answer_format="sparse",
    )[0]

    assert stored.prompt == prompt
    assert reposed.input == stored.input
    assert reposed.answer_format == "sparse"
    assert "row 0: 1" in reposed.prompt


def test_feedback_rounds_refuse_sparse_answers():
    with pytest.raises(ValueError, match="grid answers only"):
        SPhyREnvironment(sample_count=1).reset(
            subject="1_random_cell_easy", prompt_style="sparse", feedback_rounds=1
        )

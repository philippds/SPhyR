import json

import pytest

from sphyr.tasks import str_to_grid
from sphyr_env import SPhyRAction
from sphyr_env.server.sphyr_environment import SPhyREnvironment


@pytest.fixture(scope="module")
def env():
    return SPhyREnvironment(sample_count=5)


def fill_masked(grid_text, value):
    """Complete a masked grid by putting ``value`` in every 'V' cell."""
    return "\n".join(
        " ".join(value if cell == "V" else cell for cell in row)
        for row in str_to_grid(grid_text)
    )


def test_reset_serves_a_task(env):
    observation = env.reset(subject="10_random_cell_easy", sample_index=0)

    assert observation.done is False
    assert observation.reward == 0.0
    assert observation.subject == "10_random_cell_easy"
    assert observation.cell_values == "binary"
    assert observation.rows == 10
    assert observation.cols == 10
    assert observation.masked_cells > 0
    assert "V" in observation.grid
    assert observation.grid in observation.prompt

    # The answer is not handed out with the question.
    assert observation.ground_truth is None
    assert observation.metrics is None


def test_reset_is_addressable_and_deterministic(env):
    first = env.reset(subject="1_random_cell_easy", sample_index=3)
    again = env.reset(subject="1_random_cell_easy", sample_index=3)

    assert first.sample_index == again.sample_index
    assert first.grid == again.grid
    assert first.prompt == again.prompt


def test_sequential_resets_advance_and_wrap(env):
    env.reset(subject="1_random_cell_easy", sample_index=0)

    indices = [env.reset().sample_index for _ in range(5)]

    assert indices == [1, 2, 3, 4, 0]


def test_hard_subjects_ask_for_densities(env):
    observation = env.reset(subject="1_random_cell_hard", sample_index=0)

    assert observation.cell_values == "density"
    assert "floating point number" in observation.prompt


def test_unknown_subject_is_rejected(env):
    with pytest.raises(ValueError, match="Unknown subject"):
        env.reset(subject="not_a_subject")


def test_sample_index_out_of_range_is_rejected(env):
    with pytest.raises(ValueError, match="out of range"):
        env.reset(subject="1_random_cell_easy", sample_index=99)


def test_step_scores_the_dataset_answer(env):
    env.reset(subject="10_random_cell_easy", sample_index=0)
    answer = env._current.ground_truth

    observation = env.step(SPhyRAction(grid=answer))

    assert observation.done is True
    assert observation.valid_grid is True
    assert observation.ground_truth == answer
    assert observation.metrics["exact_match"] is True
    assert observation.metrics["load_carrying"] is True
    assert observation.reward > 0.0


def test_step_tolerates_prose_and_code_fences(env):
    env.reset(subject="10_random_cell_easy", sample_index=0)
    answer = env._current.ground_truth
    bare = env.step(SPhyRAction(grid=answer))

    env.reset(sample_index=0)
    wrapped = env.step(
        SPhyRAction(grid=f"Here is the completed grid:\n\n```\n{answer}\n```\n")
    )

    assert wrapped.valid_grid is True
    assert wrapped.reward == bare.reward


def test_a_reply_with_no_grid_scores_zero(env):
    env.reset(subject="10_random_cell_easy", sample_index=0)

    observation = env.step(SPhyRAction(grid="I cannot complete this grid."))

    assert observation.done is True
    assert observation.valid_grid is False
    assert observation.reward == 0.0


def test_a_grid_of_the_wrong_shape_scores_zero(env):
    env.reset(subject="10_random_cell_easy", sample_index=0)

    observation = env.step(SPhyRAction(grid="1 0 1\n0 1 0"))

    assert observation.valid_grid is False
    assert observation.reward == 0.0


def test_reward_ranks_the_answer_above_the_degenerate_completions(env):
    # A fully masked subject, so the degenerate answers are actually degenerate:
    # with only a handful of cells masked the surrounding structure carries the
    # load whatever is written into them.
    task = env.reset(subject="full_easy", sample_index=0)
    answer = env._current.ground_truth

    scored_answer = env.step(SPhyRAction(grid=answer))

    env.reset(sample_index=0)
    scored_solid = env.step(SPhyRAction(grid=fill_masked(task.grid, "1")))

    env.reset(sample_index=0)
    scored_empty = env.step(SPhyRAction(grid=fill_masked(task.grid, "0")))

    # Filling solid carries the load but wastes material; leaving it empty
    # severs the load path entirely.
    assert scored_answer.reward > scored_solid.reward
    assert scored_solid.reward > scored_empty.reward
    assert scored_solid.metrics["volume_ratio"] > 1.0
    assert scored_empty.metrics["load_carrying"] is False
    assert scored_empty.reward == 0.0


def test_a_lightly_masked_sample_can_be_left_empty_and_still_stand(env):
    # Worth pinning down: on a subject that masks a few cells, emptying them
    # neither severs the load path nor costs much stiffness, so the reward is
    # high.  The benchmark's difficulty lives in the fully masked subjects.
    task = env.reset(subject="10_random_cell_easy", sample_index=1)

    scored_empty = env.step(SPhyRAction(grid=fill_masked(task.grid, "0")))

    assert scored_empty.metrics["load_carrying"] is True
    assert scored_empty.metrics["volume_ratio"] < 1.0
    assert scored_empty.reward > 0.0


def test_rotation_rotates_the_load_case_with_the_sample(env):
    rotated = SPhyREnvironment(sample_count=2)
    task = rotated.reset(subject="full_easy", rotation_count=3, sample_index=0)

    observation = rotated.step(SPhyRAction(grid=rotated._current.ground_truth))

    assert task.rotation_count == 3
    # The dataset answer still stands up once its gravity has been rotated too.
    assert observation.metrics["load_carrying"] is True
    assert observation.reward > 0.0


def test_state_tracks_the_episode(env):
    env.reset(subject="1_random_cell_easy", sample_index=2, episode_id="episode-001")

    assert env.state.episode_id == "episode-001"
    assert env.state.step_count == 0
    assert env.state.subject == "1_random_cell_easy"
    assert env.state.sample_index == 2

    env.step(SPhyRAction(grid=env._current.ground_truth))

    assert env.state.step_count == 1


def test_metrics_are_a_serialisable_benchmark_record(env):
    env.reset(subject="10_random_cell_easy", sample_index=0)

    metrics = env.step(SPhyRAction(grid=env._current.ground_truth)).metrics

    # The result files the paper is built from are these dicts verbatim, so
    # every key they are read back by has to be present and JSON-clean.
    for key in (
        "subject",
        "prompt",
        "ground_truth",
        "completion",
        "exact_match",
        "difference_ratio",
        "penalized_difference_ratio",
        "relative_difference_ratio",
        "valid_output_grid",
        "load_support_connected",
        "load_support_connected_force_directional",
        "isolated_clusters_count",
        "force_path_cost_average_efficiency_ratio",
        "difficulty_score",
        "difficulty_weighted_difference_ratio",
        "difficulty_weighted_relative_difference_ratio",
        "topology_score",
        "structural_efficiency",
        "material_efficiency",
        "compliance_efficiency_vs_ground_truth",
        "load_carrying",
        "compliance",
        "volume_ratio",
    ):
        assert key in metrics

    json.dumps(metrics)


def test_prompt_style_selects_the_template(env):
    styled = SPhyREnvironment(sample_count=1)

    neutral = styled.reset(
        subject="1_random_cell_easy", prompt_style="physics_neutral", sample_index=0
    )
    enhanced = styled.reset(
        subject="1_random_cell_easy", prompt_style="physics_enhanced", sample_index=0
    )

    assert "a special marker" in neutral.prompt
    assert "must be transferred through continuous material paths" in enhanced.prompt


def test_few_shot_prompts_carry_worked_examples(env):
    few_shot = SPhyREnvironment(sample_count=2)

    observation = few_shot.reset(
        subject="1_random_cell_easy", few_shot_count=2, sample_index=0
    )

    assert observation.prompt.count("Corresponding completed output grid:") == 2

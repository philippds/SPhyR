"""The client's half of the OpenEnv contract: payloads out, observations in.

These run without a server; the wire format is what is being pinned down.
"""

from sphyr_env import SPhyRAction, SPhyREnv


def make_client():
    """A client instance without opening a connection."""
    return SPhyREnv.__new__(SPhyREnv)


def test_step_payload_carries_the_grid():
    payload = make_client()._step_payload(SPhyRAction(grid="1 0\n0 1"))

    assert payload == {"grid": "1 0\n0 1"}


def test_parse_result_rebuilds_a_reset_observation():
    result = make_client()._parse_result(
        {
            "observation": {
                "prompt": "complete the grid",
                "grid": "L V\nS 1",
                "subject": "full_easy",
                "sample_index": 4,
                "rows": 2,
                "cols": 2,
                "cell_values": "binary",
                "masked_cells": 1,
                "rotation_count": 0,
            },
            "reward": 0.0,
            "done": False,
        }
    )

    assert result.done is False
    assert result.reward == 0.0
    assert result.observation.prompt == "complete the grid"
    assert result.observation.subject == "full_easy"
    assert result.observation.sample_index == 4
    assert result.observation.masked_cells == 1
    assert result.observation.metrics is None


def test_parse_result_rebuilds_a_scored_observation():
    result = make_client()._parse_result(
        {
            "observation": {
                "subject": "full_easy",
                "completion": "1 1\nS 1",
                "ground_truth": "1 0\nS 1",
                "valid_grid": True,
                "metrics": {"topology_score": 0.75, "load_carrying": True},
            },
            "reward": 0.75,
            "done": True,
            "metadata": {"note": "scored"},
        }
    )

    assert result.done is True
    assert result.reward == 0.75
    assert result.observation.valid_grid is True
    assert result.observation.ground_truth == "1 0\nS 1"
    assert result.observation.metrics["topology_score"] == 0.75
    assert result.observation.metadata == {"note": "scored"}


def test_parse_state_keeps_the_sample_being_served():
    state = make_client()._parse_state(
        {
            "episode_id": "episode-001",
            "step_count": 1,
            "subject": "3_random_row_hard",
            "sample_index": 7,
            "sample_count": 100,
        }
    )

    assert state.episode_id == "episode-001"
    assert state.step_count == 1
    assert state.subject == "3_random_row_hard"
    assert state.sample_index == 7

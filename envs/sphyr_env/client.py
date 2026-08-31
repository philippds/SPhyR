"""SPhyR Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SPhyRAction, SPhyRObservation


class SPhyREnv(EnvClient[SPhyRAction, SPhyRObservation, State]):
    """
    Client for the SPhyR Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, so a training or evaluation loop keeps one dedicated environment
    session for its whole run.

    Example:
        >>> from sphyr_env import SPhyREnv, SPhyRAction
        >>>
        >>> env = SPhyREnv(base_url="http://localhost:8000")
        >>> result = env.reset(subject="10_random_cell_easy")
        >>> print(result.observation.prompt)
        >>>
        >>> result = env.step(SPhyRAction(grid=completed_grid_text))
        >>> print(f"Reward: {result.reward}")
        >>> print(f"Load carrying: {result.observation.metrics['load_carrying']}")
        >>>
        >>> result = env.reset()   # next sample from the same subject
        >>> env.close()

    Example with Docker:
        >>> client = SPhyREnv.from_docker_image("sphyr-env:latest")
        >>> try:
        ...     result = client.reset(subject="full_hard")
        ...     result = client.step(SPhyRAction(grid=completed_grid_text))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SPhyRAction) -> Dict[str, Any]:
        """
        Convert SPhyRAction to JSON payload for step message.

        Args:
            action: SPhyRAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "grid": action.grid,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SPhyRObservation]:
        """
        Parse server response into StepResult[SPhyRObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SPhyRObservation
        """
        obs_data = payload.get("observation", {})
        observation = SPhyRObservation(
            prompt=obs_data.get("prompt"),
            grid=obs_data.get("grid"),
            subject=obs_data.get("subject"),
            sample_index=obs_data.get("sample_index"),
            rows=obs_data.get("rows"),
            cols=obs_data.get("cols"),
            cell_values=obs_data.get("cell_values"),
            masked_cells=obs_data.get("masked_cells"),
            rotation_count=obs_data.get("rotation_count"),
            completion=obs_data.get("completion"),
            ground_truth=obs_data.get("ground_truth"),
            valid_grid=obs_data.get("valid_grid"),
            metrics=obs_data.get("metrics"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State with episode_id, step_count and the sample being served
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            subject=payload.get("subject"),
            sample_index=payload.get("sample_index"),
            sample_count=payload.get("sample_count"),
        )

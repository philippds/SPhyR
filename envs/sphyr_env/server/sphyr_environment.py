"""SPhyR Environment implementation.

Each episode presents one masked structural grid from the SPhyR dataset. The
agent submits the completed grid and is scored by the benchmark's dynamic,
simulation-based evaluation: the completion is solved as a linear elastic
structure and a SIMP optimiser is re-run on exactly the masked cells, so a
structure that routes the load differently but just as efficiently scores as
well as the dataset's own answer.

Design:
- Single-step episodes: reset() provides the task, step() scores it and returns
  done=True.
- Dataset persistence: a subject is loaded and shuffled once and reused across
  resets until the task configuration changes.
- Sequential iteration: samples are served in shuffled order, or addressed
  directly with reset(sample_index=...).
- Reproducibility: one RNG, seeded once, drives both the shuffle and few-shot
  example selection, so iterating the subjects in order reproduces the sample
  order the published results were generated with.
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from sphyr.metrics.utils import extract_grid_from_text
from sphyr.scoring import calculate_all_metrics, reward_from_metrics
from sphyr.tasks import (
    RANDOM_SEED,
    SAMPLE_COUNT,
    SUBJECTS,
    generate_prompts,
    gravity_for_rotation,
    load_subject,
    DEFAULT_DATASET_VERSION,
    format_feedback,
    published_records,
    replay_samples,
    resolve_prompt_template,
    str_to_grid,
)

try:
    from models import SPhyRAction, SPhyRObservation
except ImportError:
    from ..models import SPhyRAction, SPhyRObservation


DEFAULT_SUBJECT = "1_random_cell_easy"


class SPhyREnvironment(Environment):
    """SPhyR environment for single-step structural completion tasks.

    Example:
        >>> env = SPhyREnvironment()
        >>> obs = env.reset(subject="10_random_cell_easy")
        >>> print(obs.grid)          # masked grid with 'V' cells to fill
        >>>
        >>> obs = env.step(SPhyRAction(grid=completed_grid_text))
        >>> print(obs.reward)        # topology score in [0, 1]
        >>> print(obs.metrics["load_carrying"])
        >>> print(obs.done)          # True
        >>>
        >>> obs = env.reset()        # next sample from the same subject
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        subject: str = DEFAULT_SUBJECT,
        sample_count: int = SAMPLE_COUNT,
        rotation_count: int = 0,
        few_shot_count: int = 0,
        prompt_style: str = "default",
        dataset_source: Optional[str] = None,
        dataset_version: str = DEFAULT_DATASET_VERSION,
        feedback_rounds: int = 0,
        replay_from: Optional[str] = None,
        seed: int = RANDOM_SEED,
    ):
        """Configure the task the environment serves by default.

        Every argument can be overridden per episode on ``reset()``.
        """
        super().__init__()

        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._rng = random.Random(seed)

        self._subject = subject
        self._sample_count = sample_count
        self._rotation_count = rotation_count
        self._few_shot_count = few_shot_count
        self._prompt_style = prompt_style
        self._dataset_source = dataset_source
        self._dataset_version = dataset_version
        self._feedback_rounds = feedback_rounds
        self._replay_from = replay_from

        self._samples = []
        self._attempt = 0
        self._cursor = 0
        self._current = None
        self._built_config = None

    def _config(self):
        return (
            self._subject,
            self._sample_count,
            self._rotation_count,
            self._few_shot_count,
            self._prompt_style,
            self._dataset_source,
            self._dataset_version,
            self._replay_from,
        )

    def _build_samples(self):
        """Load and prompt-format the configured subject.

        With ``replay_from`` set the subject is not shuffled or sliced: the
        samples are the ones the named published run was scored on, in its
        order, so a new model is measured against the stored columns rather
        than against a fresh draw of the same size.
        """
        if self._replay_from:
            self._samples = replay_samples(
                subject=self._subject,
                records=published_records(self._subject, self._replay_from),
                rotation_count=self._rotation_count,
            )
            self._cursor = 0
            self._built_config = self._config()
            return

        records = load_subject(
            self._subject,
            source=self._dataset_source,
            version=self._dataset_version,
        )
        self._rng.shuffle(records)

        self._samples = generate_prompts(
            dataset=records,
            subject=self._subject,
            sample_count=self._sample_count,
            rotation_count=self._rotation_count,
            few_shot_count=self._few_shot_count,
            prompt_template=resolve_prompt_template(self._prompt_style),
            rng=self._rng,
        )
        self._cursor = 0
        self._built_config = self._config()

    def reset(
        self,
        subject: Optional[str] = None,
        sample_index: Optional[int] = None,
        sample_count: Optional[int] = None,
        rotation_count: Optional[int] = None,
        few_shot_count: Optional[int] = None,
        prompt_style: Optional[str] = None,
        dataset_source: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feedback_rounds: Optional[int] = None,
        replay_from: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> SPhyRObservation:
        """Start a new episode and return the grid to complete.

        Arguments left as ``None`` keep their current value, so a bare
        ``reset()`` serves the next sample of the subject already loaded.
        Changing any of them rebuilds the sample list.

        Args:
            subject: SPhyR subject to draw from, e.g. ``'full_hard'``.
            sample_index: Serve this sample instead of the next one.
            sample_count: How many samples of the subject to serve.
            rotation_count: Quarter turns applied to the sample and its load case.
            few_shot_count: Solved examples to include in the prompt.
            prompt_style: ``'default'``, ``'physics_enhanced'``,
                ``'physics_neutral'``, or a template containing a GRID placeholder.
            dataset_source: ``'local'``, ``'hub'``, or None to prefer the
                bundled dataset files.
            replay_from: Name of a directory under ``results/`` whose stored
                run should be replayed, e.g. ``'claude-opus-4-20250514'``.
                Serves exactly the samples that run was scored on.
            seed: Reseed the sample order. Rebuilds the sample list.
            episode_id: Optional episode ID.

        Returns:
            SPhyRObservation carrying the prompt and the masked grid.

        Raises:
            ValueError: If the subject is unknown or sample_index is out of range.
        """
        if subject is not None:
            if subject not in SUBJECTS:
                raise ValueError(
                    f"Unknown subject {subject!r}; expected one of {SUBJECTS}"
                )
            self._subject = subject
        if sample_count is not None:
            self._sample_count = sample_count
        if rotation_count is not None:
            self._rotation_count = rotation_count
        if few_shot_count is not None:
            self._few_shot_count = few_shot_count
        if prompt_style is not None:
            self._prompt_style = prompt_style
        if dataset_source is not None:
            self._dataset_source = dataset_source
        if dataset_version is not None:
            self._dataset_version = dataset_version
        if feedback_rounds is not None:
            self._feedback_rounds = feedback_rounds
        if replay_from is not None:
            self._replay_from = replay_from

        if seed is not None:
            self._rng = random.Random(seed)
            self._built_config = None

        if self._built_config != self._config():
            self._build_samples()

        if not self._samples:
            raise ValueError(f"Subject {self._subject!r} yielded no samples")

        if sample_index is not None:
            if not 0 <= sample_index < len(self._samples):
                raise ValueError(
                    f"sample_index {sample_index} out of range for "
                    f"{self._subject!r} ({len(self._samples)} samples)"
                )
            self._cursor = sample_index

        # Wrap around rather than ending: an episode is one sample, and a
        # training loop should be able to keep resetting.
        self._cursor %= len(self._samples)
        self._current = self._samples[self._cursor]
        self._attempt = 0
        self._cursor += 1

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            subject=self._subject,
            sample_index=self._current.index,
            sample_count=len(self._samples),
        )

        return SPhyRObservation(
            prompt=self._current.prompt,
            grid=self._current.input,
            subject=self._current.subject,
            sample_index=self._current.index,
            rows=self._current.rows,
            cols=self._current.cols,
            cell_values=self._current.cell_values,
            masked_cells=self._current.masked_cells,
            rotation_count=self._current.rotation_count,
            done=False,
            reward=0.0,
        )

    def step(self, action: SPhyRAction) -> SPhyRObservation:  # type: ignore[override]
        """Score the submitted grid, and end the episode unless a revision is allowed.

        Args:
            action: SPhyRAction carrying the completed grid as text.

        Returns:
            SPhyRObservation with the full metric set, the reference answer and
            a scalar reward, always done=True.
        """
        self._state.step_count += 1

        if self._current is None:
            return SPhyRObservation(done=True, reward=0.0)

        sample = self._current

        # Models wrap the grid in prose or code fences, so the last grid block
        # in the text is taken as the answer; a completion with no grid in it
        # at all falls through as an invalid one and scores zero.
        output_grid = extract_grid_from_text(action.grid) or str_to_grid(action.grid)

        metrics = calculate_all_metrics(
            input_grid=str_to_grid(sample.input),
            output_grid=output_grid,
            gt_grid=str_to_grid(sample.ground_truth),
            subject=sample.subject,
            prompt=sample.prompt,
            ground_truth=sample.ground_truth,
            output=action.grid,
            gravity_dir=gravity_for_rotation(sample.rotation_count),
        )

        self._attempt += 1
        done = self._attempt > self._feedback_rounds

        observation = SPhyRObservation(
            subject=sample.subject,
            sample_index=sample.index,
            rows=sample.rows,
            cols=sample.cols,
            cell_values=sample.cell_values,
            masked_cells=sample.masked_cells,
            rotation_count=sample.rotation_count,
            completion=action.grid,
            ground_truth=sample.ground_truth,
            valid_grid=metrics["valid_output_grid"],
            metrics=metrics,
            done=done,
            reward=reward_from_metrics(metrics),
        )

        if not done:
            # Another round is allowed: show the simulation's verdict and the
            # grid again, and withhold the answer so the episode stays a task.
            observation.prompt = resolve_prompt_template("feedback").format(
                FEEDBACK=format_feedback(metrics),
                GRID=sample.input,
                PREVIOUS=action.grid,
            )
            observation.grid = sample.input
            observation.ground_truth = None

        return observation

    @property
    def state(self) -> State:
        """Current episode state, including the subject and sample being served."""
        return self._state

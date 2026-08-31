"""Run the SPhyR benchmark by driving the OpenEnv environment.

Every experiment here is the same loop: reset the environment to get a masked
grid, ask a policy to complete it, step the environment to have the completion
simulated and scored, write the result.  The environment is the only thing that
knows how a task is built or how an answer is scored, so a run against a model
here and a run against an RL agent elsewhere are measured identically.

The environment runs in-process by default.  Point ``--base-url`` at a served
environment (see ``envs/sphyr_env``) to run against a container instead.

Examples::

    python -m sphyr.run_eval main
    python -m sphyr.run_eval main --models claude-opus-4-20250514
    python -m sphyr.run_eval rotations
    python -m sphyr.run_eval few-shot --few-shot-count 3
    python -m sphyr.run_eval rescore
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from tqdm import tqdm

from sphyr.metrics.utils import extract_grid_from_text, get_gravity_from_folder
from sphyr.policies import RUNNABLE_MODELS, get_policy, resolve_model
from sphyr.scoring import Result, aggregate_results, calculate_all_metrics
from sphyr.tasks import SAMPLE_COUNT, SUBJECTS

from sphyr_env import SPhyRAction

# The published main experiment covered ten models; the three OpenRouter no
# longer serves (see sphyr.policies.MODELS) are absent here because a run
# against them is no longer possible, not because they were dropped.
DEFAULT_MODELS = [
    "gpt-4.1-2025-04-14",
    "gemini-2.5-pro-preview-05-06",
    "deepseek-reasoner",
    "claude-opus-4-20250514",
    "gpt-4o-2024-08-06",
    "gpt-3.5-turbo-0125",
    "perplexity-sonar",
]

RESULTS_ROOT = "results"


class EnvSession:
    """One reset/step interface over an in-process or a served environment.

    The OpenEnv client returns a ``StepResult`` wrapping the observation while
    the environment returns the observation directly; the loop below should not
    have to care which it is talking to.
    """

    def __init__(self, env, remote, reset_defaults=None):
        self._env = env
        self._remote = remote
        self._reset_defaults = reset_defaults or {}

    def reset(self, **kwargs):
        result = self._env.reset(**{**self._reset_defaults, **kwargs})
        return result.observation if self._remote else result

    def step(self, action):
        result = self._env.step(action)
        return result.observation if self._remote else result

    def close(self):
        close = getattr(self._env, "close", None)
        if close is not None:
            close()


@contextmanager
def open_session(base_url=None, **reset_defaults):
    """Open an environment session, in-process or against a served environment.

    The task configuration travels on every ``reset()`` rather than in the
    constructor, so a served environment is configured the same way an
    in-process one is.
    """
    if base_url:
        from sphyr_env import SPhyREnv

        client = SPhyREnv(base_url=base_url).sync()
        client.connect()
        session = EnvSession(client, remote=True, reset_defaults=reset_defaults)
    else:
        # Imported here so that rescoring stored results does not need the
        # environment package at all.
        from sphyr_env.server.sphyr_environment import SPhyREnvironment

        session = EnvSession(
            SPhyREnvironment(), remote=False, reset_defaults=reset_defaults
        )

    try:
        yield session
    finally:
        session.close()


def load_existing_results(results_path):
    """Stored results for this model and subject, keyed by prompt."""
    if not os.path.exists(results_path):
        return {}

    with open(results_path, "r") as handle:
        return {record["prompt"]: record for record in json.load(handle)}


def results_dir_name(model, name_suffix=""):
    """The directory one model's results are stored under.

    An OpenRouter model id carries a vendor prefix (``openai/gpt-4.1``); flatten
    it so results land in one directory per model rather than nested under the
    vendor, where the rescoring scan would not find them.  Ollama tags carry a
    colon (``qwen3:8b``), which is not a legal filename character on Windows,
    so it is flattened the same way.
    """
    safe = model
    for character in "/:":
        safe = safe.replace(character, "_")
    return f"{safe}{name_suffix}"


def evaluate_subject(
    session,
    policy,
    model,
    subject,
    name_suffix="",
    sample_count=SAMPLE_COUNT,
    concurrency=1,
):
    """Run one model over one subject, resuming from whatever is already stored.

    The model calls are the only slow part -- a reasoning model can spend
    minutes on a single grid -- so they are the only part run in parallel.  The
    environment stays on the main thread throughout, for two reasons:

    * It holds the current sample between ``reset()`` and ``step()``, so
      concurrent episodes on one instance would score answers against each
      other's grids.
    * Its RNG is seeded once and its state carries across subject builds, so a
      per-thread environment would shuffle every subject after the first
      differently and quietly evaluate this model on different samples than the
      stored results for every other model.

    Addressing a ``sample_index`` consumes no randomness once the subject is
    built, so posing the tasks up front and re-posing them to score consumes
    exactly the sample order the sequential runner produced.
    """
    root_dir = f"{RESULTS_ROOT}/{results_dir_name(model, name_suffix)}"
    os.makedirs(root_dir, exist_ok=True)

    results_path = f"{root_dir}/{subject}_results.json"
    existing_results = load_existing_results(results_path)
    results = list(existing_results.values())

    print(
        f"Evaluating {model} for subject {subject} - "
        f"Existing results: {len(existing_results)}"
    )

    pending = []
    for sample_index in range(sample_count):
        observation = session.reset(subject=subject, sample_index=sample_index)
        if observation.prompt not in existing_results:
            pending.append((sample_index, observation))

    completions = {}
    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(policy, observation): sample_index
                for sample_index, observation in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                sample_index = futures[future]
                try:
                    completions[sample_index] = future.result()
                except Exception as error:
                    # A failed sample is skipped, not recorded: the run stays
                    # resumable and a rerun picks it up.
                    print(f"Error processing sample {sample_index}: {error}")

    for sample_index, _ in pending:
        action = completions.get(sample_index)
        if action is None:
            continue

        session.reset(subject=subject, sample_index=sample_index)
        scored = session.step(action)

        record = scored.metrics
        # Stamp what actually answered.  The published results carry no model
        # field, which is how a column produced by one model came to be
        # reported as another.  A baseline policy answers under a name that is
        # no OpenRouter model, so the slug is recorded only when there is one.
        record["model"] = model
        try:
            record["openrouter_model"] = resolve_model(model)
        except ValueError:
            pass
        results.append(record)
        existing_results[record["prompt"]] = record

        # Save after each sample: these runs are long and paid for by the call.
        with open(results_path, "w") as handle:
            json.dump(results, handle, indent=2)

    if len(results) >= sample_count:
        aggregated_results = aggregate_results([Result.from_dict(r) for r in results])

        print(f"{model} results for {subject}:")
        print(f"Total exact match: {aggregated_results['total_exact_match']}")
        print(f"Total topology score: {aggregated_results['total_topology_score']}")
        print(
            "Total structural efficiency: "
            f"{aggregated_results['total_structural_efficiency']}"
        )

        with open(f"{root_dir}/{subject}_aggregated_results.json", "w") as handle:
            json.dump(aggregated_results, handle)
    else:
        print(
            f"Skipping aggregation for {model}/{subject} - only {len(results)} "
            f"of {sample_count} samples completed."
        )


def run_experiment(
    models,
    subjects,
    name_suffix="",
    sample_count=SAMPLE_COUNT,
    base_url=None,
    concurrency=1,
    **task_config,
):
    """Run a set of models over a set of subjects in one environment session.

    One session for the whole experiment is deliberate: the environment seeds
    its sample order once, so every model sees the same samples in the same
    order, exactly as the published results were produced.
    """
    with open_session(
        base_url=base_url, sample_count=sample_count, **task_config
    ) as session:
        for subject in tqdm(subjects):
            for model in models:
                evaluate_subject(
                    session=session,
                    policy=get_policy(model),
                    model=model,
                    subject=subject,
                    name_suffix=name_suffix,
                    sample_count=sample_count,
                    concurrency=concurrency,
                )


def run_main_experiment(
    models=None,
    base_url=None,
    concurrency=1,
    replay_from=None,
    dataset_version=None,
    sample_count=SAMPLE_COUNT,
):
    run_experiment(
        models=models or DEFAULT_MODELS,
        subjects=SUBJECTS,
        sample_count=sample_count,
        base_url=base_url,
        concurrency=concurrency,
        **({"replay_from": replay_from} if replay_from else {}),
        **({"dataset_version": dataset_version} if dataset_version else {}),
    )


def run_rotation_comparison_experiment(
    models=None, rotations=3, base_url=None, concurrency=1, replay_from=None
):
    run_experiment(
        models=models
        or [
            "gpt-4.1-2025-04-14",
            "gemini-2.5-pro-preview-05-06",
            "deepseek-reasoner",
            "claude-opus-4-20250514",
            "perplexity-sonar",
        ],
        subjects=[
            "10_random_cell_easy",
            "3_random_row_easy",
            "3_random_column_easy",
            "full_easy",
        ],
        name_suffix=f"_{rotations}_rotations",
        rotation_count=rotations,
        base_url=base_url,
        concurrency=concurrency,
        **({"replay_from": replay_from} if replay_from else {}),
    )


def run_rotation_best_model_experiment(models=None, rotations=3, base_url=None):
    run_experiment(
        models=models or ["claude-opus-4-20250514"],
        subjects=[
            "1_random_cell_easy",
            "5_random_cell_easy",
            "1_random_row_easy",
            "1_random_column_easy",
            "1_random_cell_hard",
            "5_random_cell_hard",
            "10_random_cell_hard",
            "1_random_row_hard",
            "3_random_row_hard",
            "1_random_column_hard",
            "3_random_column_hard",
            "full_hard",
        ],
        name_suffix=f"_{rotations}_rotations",
        rotation_count=rotations,
        base_url=base_url,
    )


def run_few_shot_experiment(few_shot_count=1, models=None, base_url=None):
    run_experiment(
        models=models or ["claude-opus-4-20250514"],
        subjects=SUBJECTS,
        name_suffix=f"_few_shot_{few_shot_count}",
        few_shot_count=few_shot_count,
        base_url=base_url,
    )


def run_prompt_style_experiment(prompt_style, models=None, base_url=None):
    run_experiment(
        models=models or ["gemini-2.5-pro-preview-05-06"],
        subjects=SUBJECTS,
        name_suffix=f"_{prompt_style}_prompt",
        prompt_style=prompt_style,
        base_url=base_url,
    )


def rescore_stored_results(results_root=RESULTS_ROOT):
    """Recompute every metric for the stored results and rewrite them in place.

    The completions are already on disk, so this needs no model and no
    environment: the same scorer the environment calls from ``step()`` is
    applied to the grids parsed back out of each record.
    """
    print(f"Scanning {results_root}/ for existing results...")

    model_dirs = [
        name
        for name in sorted(os.listdir(results_root))
        if os.path.isdir(os.path.join(results_root, name)) and name != "plots"
    ]

    for model_dir in model_dirs:
        model_path = os.path.join(results_root, model_dir)
        print(f"\nProcessing model directory: {model_dir}")

        # Rotation is recorded in the directory name, not the file name, and
        # gravity has to rotate with the sample.
        gravity_dir = get_gravity_from_folder(model_dir)

        for file_name in sorted(os.listdir(model_path)):
            if file_name.endswith("_aggregated_results.json"):
                continue
            if not file_name.endswith("_results.json"):
                continue

            results_file = os.path.join(model_path, file_name)

            with open(results_file, "r") as handle:
                records = json.load(handle)

            for record in records:
                record.update(
                    calculate_all_metrics(
                        input_grid=extract_grid_from_text(record["prompt"]),
                        output_grid=extract_grid_from_text(record["completion"]),
                        gt_grid=extract_grid_from_text(record["ground_truth"]),
                        subject=record["subject"],
                        prompt=record["prompt"],
                        ground_truth=record["ground_truth"],
                        output=record["completion"],
                        gravity_dir=gravity_dir,
                    )
                )

                # Retired metrics from earlier revisions of the benchmark.
                record.pop("score", None)
                record.pop("normalized_score", None)

            with open(results_file, "w") as handle:
                json.dump(records, handle, indent=2)

            aggregated = aggregate_results([Result.from_dict(r) for r in records])

            with open(
                results_file.replace("_results.json", "_aggregated_results.json"), "w"
            ) as handle:
                json.dump(aggregated, handle, indent=2)

    print("\nAll files updated.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "experiment",
        choices=[
            "main",
            "rotations",
            "rotations-best-model",
            "few-shot",
            "physics-enhanced",
            "physics-neutral",
            "rescore",
        ],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help=(
            "Models to evaluate; defaults to the experiment's own set. Either a "
            "benchmark model name (" + ", ".join(RUNNABLE_MODELS) + ") or any "
            "OpenRouter model id, e.g. anthropic/claude-sonnet-4.5"
        ),
    )
    parser.add_argument(
        "--base-url",
        help="Run against a served environment instead of an in-process one",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Model calls to keep in flight at once. Only the calls run in "
            "parallel; the environment is stepped on the main thread, so the "
            "samples served are unchanged. A reasoning model can take minutes "
            "per sample, where the default of 1 is impractically slow."
        ),
    )
    parser.add_argument(
        "--replay-from",
        metavar="RESULTS_DIR",
        help=(
            "Evaluate the exact samples a stored run was scored on, naming a "
            "directory under results/ (e.g. claude-opus-4-20250514). Without "
            "it a run draws its own samples and its numbers are not "
            "comparable to the published columns."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=SAMPLE_COUNT,
        help=(
            "Samples per subject to evaluate. The published v1 runs used 100; "
            "the regenerated v2 sets contain 300."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default="v1",
        help=(
            "Dataset version to evaluate on: v1 (the published benchmark), "
            "v2 (regenerated with widely separated samples), or v2-20 (the "
            "same on a 20x20 grid)."
        ),
    )
    parser.add_argument("--few-shot-count", type=int, default=1)
    parser.add_argument("--rotations", type=int, default=3)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.experiment == "rescore":
        rescore_stored_results(args.results_root)
    elif args.experiment == "main":
        run_main_experiment(
            models=args.models,
            base_url=args.base_url,
            concurrency=args.concurrency,
            replay_from=args.replay_from,
            dataset_version=args.dataset_version,
            sample_count=args.sample_count,
        )
    elif args.experiment == "rotations":
        run_rotation_comparison_experiment(
            models=args.models,
            rotations=args.rotations,
            base_url=args.base_url,
            concurrency=args.concurrency,
            replay_from=args.replay_from,
        )
    elif args.experiment == "rotations-best-model":
        run_rotation_best_model_experiment(
            models=args.models, rotations=args.rotations, base_url=args.base_url
        )
    elif args.experiment == "few-shot":
        run_few_shot_experiment(
            few_shot_count=args.few_shot_count,
            models=args.models,
            base_url=args.base_url,
        )
    elif args.experiment == "physics-enhanced":
        run_prompt_style_experiment(
            "physics_enhanced", models=args.models, base_url=args.base_url
        )
    elif args.experiment == "physics-neutral":
        run_prompt_style_experiment(
            "physics_neutral", models=args.models, base_url=args.base_url
        )


if __name__ == "__main__":
    main()

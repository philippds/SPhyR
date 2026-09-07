"""Run SPhyR as a multi-step environment: submit, be simulated, revise.

The benchmark asks a model to answer once. This asks whether the verifier's
signal is *actionable*: the completion is simulated, the model is told how the
structure behaved -- whether it stood, how stiff it was for the material spent,
how much material that was -- and asked to try again.

The feedback is physical and never says which cells are wrong, so an
improvement cannot come from being handed the answer. It can only come from the
model using grounding it has no way to compute for itself.

What separates this from the single-step benchmark is what it measures. A
benchmark score says how well a model does the task; the trajectory across
rounds says whether the environment's reward can be optimised against, which is
what an environment has to support to be worth training in.

Usage::

    python -m sphyr.run_feedback --models ollama/qwen3:8b --subject full_easy
"""

import argparse
import json
import os

from tqdm import tqdm

from sphyr.policies import get_policy
from sphyr.run_eval import RESULTS_ROOT, open_session, results_dir_name
from sphyr.tasks import DEFAULT_DATASET_VERSION

from sphyr_env import SPhyRAction

DEFAULT_ROUNDS = 3
DEFAULT_SAMPLES = 50
SAVE_EVERY = 5


def run_episode(session, policy, subject, sample_index, rounds, feedback=True):
    """One episode: an answer, then ``rounds`` further attempts, scored at each.

    With ``feedback`` the attempts are revisions -- each one sees how the last
    was simulated. Without it they are independent redraws of the same prompt,
    which is the control the revision numbers need: taking the best of several
    attempts beats a single attempt whether or not the attempts learn anything,
    simply because the best of several noisy draws is higher than one. Only the
    difference between the two conditions says whether the feedback did work.
    """
    observation = session.reset(
        subject=subject,
        sample_index=sample_index,
        feedback_rounds=rounds if feedback else 0,
    )

    trajectory = []
    for _ in range(rounds + 1):
        action = policy(observation)
        scored = session.step(action)
        metrics = scored.metrics or {}
        trajectory.append(
            {
                "topology_score": metrics.get("topology_score", 0.0),
                "structural_efficiency": metrics.get("structural_efficiency", 0.0),
                "material_efficiency": metrics.get("material_efficiency", 0.0),
                "volume_ratio": metrics.get("volume_ratio"),
                "load_carrying": bool(metrics.get("load_carrying")),
                "valid_output_grid": bool(metrics.get("valid_output_grid")),
                "completion": scored.completion,
            }
        )
        if not feedback:
            # Re-pose the same task rather than carrying the verdict forward.
            observation = session.reset(subject=subject, sample_index=sample_index)
            continue
        if scored.done:
            break
        observation = scored

    return trajectory


def run(model, policy, subject, rounds, samples, dataset_version,
        base_url=None, feedback=True):
    suffix = '_feedback' if feedback else '_resample'
    root = f"{RESULTS_ROOT}/{results_dir_name(model, suffix, dataset_version)}"
    os.makedirs(root, exist_ok=True)
    path = f"{root}/{subject}_results.json"

    episodes = []
    if os.path.exists(path):
        with open(path, "r") as handle:
            episodes = json.load(handle)
    done_indices = {e["sample_index"] for e in episodes}

    print(
        f"{model} on {subject} ({dataset_version}), {rounds} feedback rounds - "
        f"already stored: {len(episodes)}"
    )

    with open_session(
        base_url=base_url, sample_count=samples, dataset_version=dataset_version
    ) as session:
        for sample_index in tqdm(range(samples)):
            if sample_index in done_indices:
                continue
            try:
                trajectory = run_episode(
                    session, policy, subject, sample_index, rounds, feedback
                )
            except Exception as error:
                print(f"Error on sample {sample_index}: {error}")
                continue

            episodes.append(
                {
                    "sample_index": sample_index,
                    "model": model,
                    "dataset_version": dataset_version,
                    "rounds": rounds,
                    "trajectory": trajectory,
                }
            )
            if len(episodes) % SAVE_EVERY == 0:
                with open(path, "w") as handle:
                    json.dump(episodes, handle, indent=2)

    with open(path, "w") as handle:
        json.dump(episodes, handle, indent=2)

    report(episodes, rounds)
    return episodes


def report(episodes, rounds):
    """Mean score at each round, so the trajectory is visible."""
    if not episodes:
        return
    print(f"\n{'round':>6} {'TS':>8} {'LC %':>7} {'valid %':>8} {'n':>5}")
    for r in range(rounds + 1):
        vals = [e["trajectory"][r] for e in episodes if len(e["trajectory"]) > r]
        if not vals:
            continue
        print(
            f"{r:6d} "
            f"{100 * sum(v['topology_score'] for v in vals) / len(vals):8.2f} "
            f"{100 * sum(v['load_carrying'] for v in vals) / len(vals):7.1f} "
            f"{100 * sum(v['valid_output_grid'] for v in vals) / len(vals):8.1f} "
            f"{len(vals):5d}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--subject", default="full_easy")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help=(
            "Redraw the same prompt instead of revising, as the control for "
            "the revision condition."
        ),
    )
    args = parser.parse_args(argv)

    for model in args.models:
        run(
            model=model,
            policy=get_policy(model),
            subject=args.subject,
            rounds=args.rounds,
            samples=args.samples,
            dataset_version=args.dataset_version,
            base_url=args.base_url,
            feedback=not args.no_feedback,
        )


if __name__ == "__main__":
    main()

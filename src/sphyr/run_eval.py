from dataclasses import dataclass
import json
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import os
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_runners import (
    run_claude,
    run_claude_opus,
    run_deepkseek,
    run_gemini,
    run_gemini_1_5,
    run_openai,
    run_openai_3_5_turbo,
    run_openai_4o,
    run_perplexity_sonar,
    run_perplexity_sonar_reasoning,
)
from prompt_templates import (
    PROMPT_TEMPLATE,
    FEW_SHOT_PROMPT_TEMPLATE,
    FEW_SHOT_EXAMPLES,
)

# --------------------------------
# CONFIG
# --------------------------------
load_dotenv()
login(token=os.getenv("HF"))

SUBJECTS = [
    "1_random_cell_easy",
    "5_random_cell_easy",
    "10_random_cell_easy",
    "1_random_row_easy",
    "3_random_row_easy",
    "1_random_column_easy",
    "3_random_column_easy",
    "full_easy",
    "1_random_cell_hard",
    "5_random_cell_hard",
    "10_random_cell_hard",
    "1_random_row_hard",
    "3_random_row_hard",
    "1_random_column_hard",
    "3_random_column_hard",
    "full_hard",
]

RANDOM_SEED = 42
SAMPLE_COUNT = 100
rnd = random.Random(RANDOM_SEED)
MAX_THREADS = 4

MODEL_RUNNERS = {
    "gpt-4.1-2025-04-14": run_openai,
    "gpt-4o-2024-08-06": run_openai_4o,
    "gpt-3.5-turbo-0125": run_openai_3_5_turbo,
    "gemini-1.5-pro": run_gemini_1_5,
    "gemini-2.5-pro-preview-05-06": run_gemini,
    "claude-3-7-sonnet-20250219": run_claude,
    "claude-opus-4-20250514": run_claude_opus,
    "deepseek-reasoner": run_deepkseek,
    "perplexity-sonar": run_perplexity_sonar,
    "perplexity-sonar-reasoning": run_perplexity_sonar_reasoning,
}


# --------------------------------
# DATA CLASSES
# --------------------------------
@dataclass
class Sample:
    subject: str
    raw_input: str
    prompt: str
    ground_truth: str


@dataclass
class Result:
    subject: str
    prompt: str
    input_grid: str
    ground_truth: str
    completion: str
    exact_match: bool
    score: float
    normalized_score: float


# --------------------------------
# UTILS
# --------------------------------
def rotate_grid(grid_str, times=1):
    grid = [row.split() for row in grid_str.strip().split("\n")]
    for _ in range(times % 4):
        grid = [list(row) for row in zip(*grid[::-1])]
    return "\n".join(" ".join(row) for row in grid)


def count_differences(list1, list2) -> int:
    return sum(
        cell1 != cell2
        for row1, row2 in zip(list1, list2)
        for cell1, cell2 in zip(row1, row2)
    )


def calculate_score(output_diff, raw_diff) -> float:
    if output_diff == 0 and raw_diff == 0:
        return 1
    return 1 - (output_diff / raw_diff)


def aggregate_results(results: list[Result]) -> dict:
    return {
        "total_exact_match": sum(r.exact_match for r in results),
        "total_score": sum(r.score for r in results),
        "total_normalized_score": sum(r.normalized_score for r in results),
    }


def create_few_shot_example(dataset, current_sample, few_shot_count, rotation_count):
    few_shot_examples = rnd.sample(dataset, few_shot_count + 1)
    few_shot_examples = [
        ex
        for ex in few_shot_examples
        if ex["input_grid"] != current_sample["input_grid"]
    ][:few_shot_count]

    return "\n\n".join(
        FEW_SHOT_EXAMPLES.format(
            EXAMPLE_GRID=rotate_grid(ex["input_grid"], times=rotation_count),
            EXAMPLE_COMPLETED_GRID=rotate_grid(
                ex["ground_truth"], times=rotation_count
            ),
        )
        for ex in few_shot_examples
    )


def generate_prompts(
    dataset, subject, sample_count=SAMPLE_COUNT, rotation_count=0, few_shot_count=0
):
    samples = []
    for sample in dataset[:sample_count]:
        input_grid = rotate_grid(sample["input_grid"], times=rotation_count)
        fill_instruction = (
            "'V' cells with either '1' (solid) or '0' (empty)"
            if subject.endswith("easy")
            else "'V' cells with a floating point number between 0 and 1"
        )

        if few_shot_count > 0:
            few_shot_examples = create_few_shot_example(
                dataset, sample, few_shot_count, rotation_count
            )
            prompt = FEW_SHOT_PROMPT_TEMPLATE.format(
                FILL_INSTRUCTION=fill_instruction,
                FEW_SHOT_EXAMPLES=few_shot_examples,
                GRID=input_grid,
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                FILL_INSTRUCTION=fill_instruction, GRID=input_grid
            )

        ground_truth = rotate_grid(sample["ground_truth"], times=rotation_count)
        samples.append(Sample(subject, input_grid, prompt, ground_truth))

    return samples


# --------------------------------
# PIPELINE STEPS
# --------------------------------


# 1. Generate Completions
def generate_completions(model, samples, name_suffix):
    root_dir = f"results/{model}{name_suffix}"
    os.makedirs(root_dir, exist_ok=True)
    subject = samples[0].subject
    completions_path = f"{root_dir}/{subject}_completions.json"

    existing = {}
    if os.path.exists(completions_path):
        with open(completions_path, "r") as f:
            for r in json.load(f):
                existing[r["prompt"]] = r

    results = list(existing.values())
    eval_fn = MODEL_RUNNERS[model]

    for sample in tqdm(samples, desc=f"Generating completions for {model} - {subject}"):
        if sample.prompt in existing:
            continue
        try:
            completion = eval_fn(sample.prompt)
            result = {
                "subject": sample.subject,
                "prompt": sample.prompt,
                "input_grid": sample.raw_input,
                "ground_truth": sample.ground_truth,
                "completion": completion,
            }
            results.append(result)
            existing[sample.prompt] = result
            with open(completions_path, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error generating completion: {e}")


# 2. Evaluate Completions
def evaluate_completions(model, subject):
    root_dir = f"results/{model}"
    completions_path = f"{root_dir}/{subject}_completions.json"
    evaluations_path = f"{root_dir}/{subject}_evaluations.json"

    if not os.path.exists(completions_path):
        print(f"No completions found for {model} - {subject}")
        return

    with open(completions_path, "r") as f:
        completions = json.load(f)

    evaluations = []
    for c in completions:
        raw_input_list = [line.split() for line in c["input_grid"].splitlines()]
        output_list = [line.split() for line in c["completion"].splitlines()]
        ground_list = [line.split() for line in c["ground_truth"].splitlines()]

        raw_diff = count_differences(raw_input_list, ground_list)
        out_diff = count_differences(output_list, ground_list)
        score = calculate_score(out_diff, raw_diff) if out_diff else 1
        normalized_score = max(score, 0)
        exact_match = out_diff == 0

        evaluations.append(
            {
                **c,
                "exact_match": exact_match,
                "score": score,
                "normalized_score": normalized_score,
            }
        )

    with open(evaluations_path, "w") as f:
        json.dump(evaluations, f, indent=2)


# 3. Aggregate Evaluations
def aggregate_evaluations(model, subject):
    root_dir = f"results/{model}"
    evaluations_path = f"{root_dir}/{subject}_evaluations.json"
    aggregated_path = f"{root_dir}/{subject}_aggregated.json"

    if not os.path.exists(evaluations_path):
        print(f"No evaluations found for {model} - {subject}")
        return

    with open(evaluations_path, "r") as f:
        evaluations = json.load(f)

    full_results = [
        Result(
            subject=r["subject"],
            prompt=r["prompt"],
            input_grid=r["input_grid"],
            ground_truth=r["ground_truth"],
            completion=r["completion"],
            exact_match=r["exact_match"],
            score=r["score"],
            normalized_score=r["normalized_score"],
        )
        for r in evaluations
    ]

    aggregated = aggregate_results(full_results)
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2)


# --------------------------------
# RUNNERS
# --------------------------------
def run_generate_experiment(samples, models, name_suffix="", skip_inference=False):
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(generate_completions, model, samples, name_suffix): model
            for model in models
        }
        for future in as_completed(futures):
            future.result()


def run_evaluate_experiment(subject, models, name_suffix=""):
    for model in models:
        evaluate_completions(model, subject, name_suffix)


def run_aggregate_experiment(subject, models, name_suffix=""):
    for model in models:
        aggregate_evaluations(model, subject, name_suffix)


# --------------------------------
# EXPERIMENTS
# --------------------------------
def run_main_experiment(skip_inference=False):
    for subject in tqdm(SUBJECTS):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)
        samples = generate_prompts(dataset=dataset_list, subject=subject)

        run_generate_experiment(samples, MODEL_RUNNERS.keys(), "", skip_inference)
        run_evaluate_experiment(subject, MODEL_RUNNERS.keys(), "")
        run_aggregate_experiment(subject, MODEL_RUNNERS.keys(), "")


def run_rotation_comparison_experiment(skip_inference=False):
    selected_subjects = [
        "10_random_cell_easy",
        "3_random_row_easy",
        "3_random_column_easy",
        "full_easy",
    ]
    selected_models = [
        "gpt-4.1-2025-04-14",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "claude-opus-4-20250514",
        "perplexity-sonar",
    ]

    for subject in tqdm(selected_subjects):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)
        samples = generate_prompts(
            dataset=dataset_list, subject=subject, rotation_count=3
        )

        run_generate_experiment(
            samples, selected_models, "_3_rotations", skip_inference
        )
        run_evaluate_experiment(subject, selected_models, "_3_rotations")
        run_aggregate_experiment(subject, selected_models, "_3_rotations")


def run_rotation_best_model_experiment(skip_inference=False):
    selected_subjects = [
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
    ]
    best_model = "claude-opus-4-20250514"

    for subject in tqdm(selected_subjects):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)
        samples = generate_prompts(
            dataset=dataset_list, subject=subject, rotation_count=3
        )

        run_generate_experiment(samples, [best_model], "_3_rotations", skip_inference)
        run_evaluate_experiment(subject, [best_model], "_3_rotations")
        run_aggregate_experiment(subject, [best_model], "_3_rotations")


def run_few_shot_experiment(few_shot_count=1, skip_inference=False):
    best_model = "claude-opus-4-20250514"

    for subject in tqdm(SUBJECTS):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)
        samples = generate_prompts(
            dataset=dataset_list, subject=subject, few_shot_count=few_shot_count
        )

        suffix = f"_few_shot_{few_shot_count}"
        run_generate_experiment(samples, [best_model], suffix, skip_inference)
        run_evaluate_experiment(subject, [best_model], suffix)
        run_aggregate_experiment(subject, [best_model], suffix)


if __name__ == "__main__":
    run_main_experiment(skip_inference=False)
    run_rotation_comparison_experiment(skip_inference=False)
    run_rotation_best_model_experiment(skip_inference=False)
    run_few_shot_experiment(few_shot_count=1, skip_inference=False)
    run_few_shot_experiment(few_shot_count=3, skip_inference=False)

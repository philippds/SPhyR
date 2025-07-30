import json
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
import os
import random
from tqdm import tqdm
from dataclasses import dataclass

from openai import OpenAI
from google import genai
import anthropic

from prompt_templates import (
    PROMPT_TEMPLATE,
    FEW_SHOT_PROMPT_TEMPLATE,
    FEW_SHOT_EXAMPLES,
)

load_dotenv()
login(token=os.getenv("HF"))

subjects = [
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


def rotate_grid(grid_str, times=1):
    grid = [row.split() for row in grid_str.strip().split("\n")]
    for _ in range(times % 4):
        grid = [list(row) for row in zip(*grid[::-1])]
    return "\n".join(" ".join(row) for row in grid)


def create_few_shot_example(dataset, current_sample, few_shot_count, rotation_count):
    few_shot_examples = rnd.sample(dataset, few_shot_count + 1)
    # make sure current sample is not in few_shot_examples
    few_shot_examples = [
        ex
        for ex in few_shot_examples
        if ex["input_grid"] != current_sample["input_grid"]
    ]
    few_shot_examples = few_shot_examples[:few_shot_count]

    concatenated_few_shot_examples = "\n\n".join(
        FEW_SHOT_EXAMPLES.format(
            EXAMPLE_GRID=rotate_grid(ex["input_grid"], times=rotation_count),
            EXAMPLE_COMPLETED_GRID=rotate_grid(
                ex["ground_truth"], times=rotation_count
            ),
        )
        for ex in few_shot_examples
    )

    return concatenated_few_shot_examples


def generate_prompts(
    dataset, subject, sample_count=SAMPLE_COUNT, rotation_count=0, few_shot_count=0
) -> list[Sample]:
    samples = []

    for sample in dataset[:sample_count]:
        input_grid = rotate_grid(sample["input_grid"], times=rotation_count)

        if subject.endswith("easy"):
            fill_instruction = "'V' cells with either '1' (solid) or '0' (empty)"
            if few_shot_count > 0:
                concatenated_few_shot_examples = create_few_shot_example(
                    dataset, sample, few_shot_count, rotation_count
                )
                prompt = FEW_SHOT_PROMPT_TEMPLATE.format(
                    FILL_INSTRUCTION=fill_instruction,
                    FEW_SHOT_EXAMPLES=concatenated_few_shot_examples,
                    GRID=input_grid,
                )
            else:
                prompt = PROMPT_TEMPLATE.format(
                    FILL_INSTRUCTION=fill_instruction,
                    GRID=input_grid,
                )
        else:
            fill_instruction = "'V' cells with a floating point number between 0 and 1, with one decimal place (e.g., 0.0, 0.1, 0.2, ..., 1.0)"
            if few_shot_count > 0:
                concatenated_few_shot_examples = create_few_shot_example(
                    dataset, sample, few_shot_count, rotation_count
                )
                prompt = FEW_SHOT_PROMPT_TEMPLATE.format(
                    FILL_INSTRUCTION=fill_instruction,
                    FEW_SHOT_EXAMPLES=concatenated_few_shot_examples,
                    GRID=input_grid,
                )
            else:
                prompt = PROMPT_TEMPLATE.format(
                    FILL_INSTRUCTION=fill_instruction,
                    GRID=input_grid,
                )

        ground_truth = rotate_grid(sample["ground_truth"], times=rotation_count)

        samples.append(
            Sample(
                subject=subject,
                raw_input=input_grid,
                prompt=prompt,
                ground_truth=ground_truth,
            )
        )

    return samples


def count_differences(list1, list2) -> int:
    count = 0
    for row1, row2 in zip(list1, list2):
        for cell1, cell2 in zip(row1, row2):
            if cell1 != cell2:
                count += 1
    return count


def calculate_score(
    output_ground_truth_difference_count, raw_input_ground_truth_difference_count
) -> float:
    if (
        output_ground_truth_difference_count == 0
        and raw_input_ground_truth_difference_count == 0
    ):
        return 1
    score = 1 - (
        output_ground_truth_difference_count / raw_input_ground_truth_difference_count
    )
    return score


def aggregate_results(results: list[Result]) -> dict:
    total_exact_match = sum(1 for result in results if result.exact_match)
    total_score = sum(result.score for result in results)
    total_normalized_score = sum(result.normalized_score for result in results)

    return {
        "total_exact_match": total_exact_match,
        "total_score": total_score,
        "total_normalized_score": total_normalized_score,
    }


def run_openai(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OAI"))
    response = client.responses.create(model="gpt-4.1-2025-04-14", input=prompt)
    return response.output_text.strip()


def run_openai_4o(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OAI"))
    response = client.responses.create(model="gpt-4o-2024-08-06", input=prompt)
    return response.output_text.strip()


def run_openai_3_5_turbo(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OAI"))
    response = client.responses.create(model="gpt-3.5-turbo-0125", input=prompt)
    return response.output_text.strip()


def run_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEM"))
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06",
        contents=prompt,
    )
    return response.text


def run_gemini_1_5(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEM"))
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=prompt,
    )
    return response.text


def run_claude(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANT"))
    message = client.messages.create(
        max_tokens=1024,
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_claude_opus(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANT"))
    message = client.messages.create(
        max_tokens=1024,
        model="claude-opus-4-20250514",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def run_deepkseek(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("R1"), base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def run_perplexity_sonar(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("PPLX"), base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_perplexity_sonar_reasoning(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("PPLX"), base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar-reasoning",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def evaluate_against_model(model, samples, name_suffix="") -> list:
    if model == "gpt-4.1-2025-04-14":
        eval_fn = run_openai
    elif model == "gpt-4o-2024-08-06":
        eval_fn = run_openai_4o
    elif model == "gpt-3.5-turbo-0125":
        eval_fn = run_openai_3_5_turbo
    elif model == "gemini-1.5-pro":
        eval_fn = run_gemini_1_5
    elif model == "gemini-2.5-pro-preview-05-06":
        eval_fn = run_gemini
    elif model == "claude-3-7-sonnet-20250219":
        eval_fn = run_claude
    elif model == "claude-opus-4-20250514":
        eval_fn = run_claude_opus
    elif model == "deepseek-reasoner":
        eval_fn = run_deepkseek
    elif model == "perplexity-sonar":
        eval_fn = run_perplexity_sonar
    elif model == "perplexity-sonar-reasoning":
        eval_fn = run_perplexity_sonar_reasoning
    else:
        raise ValueError(f"Unknown model: {model}")

    root_dir = f"results/{model}{name_suffix}"

    os.makedirs(root_dir, exist_ok=True)
    subject = samples[0].subject if samples else "unknown"

    # Load existing results if available
    results_path = f"{root_dir}/{subject}_results.json"
    existing_results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing_result_dicts = json.load(f)
            for r in existing_result_dicts:
                existing_results[r["prompt"]] = r

    results = list(existing_results.values())

    print(
        f"Evaluating {model} for subject {subject} - Existing results: {len(existing_results)}"
    )

    if len(results) < len(samples):
        for sample in tqdm(samples):
            if sample.prompt in existing_results:
                print(f"Skipping existing sample for prompt: {sample.prompt[:60]}...")
                continue

            try:
                output_text = eval_fn(sample.prompt)

                raw_input_list = [
                    line.split() for line in sample.raw_input.splitlines()
                ]
                output_text_list = [line.split() for line in output_text.splitlines()]
                ground_truth_list = [
                    line.split() for line in sample.ground_truth.splitlines()
                ]

                raw_input_ground_truth_difference_count = count_differences(
                    raw_input_list, ground_truth_list
                )

                output_ground_truth_difference_count = count_differences(
                    output_text_list, ground_truth_list
                )

                exact_match = True
                score = 1
                normalized_score = 1
                if output_ground_truth_difference_count != 0:
                    exact_match = False
                    score = calculate_score(
                        output_ground_truth_difference_count,
                        raw_input_ground_truth_difference_count,
                    )
                    normalized_score = max(score, 0)

                result = Result(
                    subject=sample.subject,
                    prompt=sample.prompt,
                    input_grid=sample.raw_input,
                    ground_truth=sample.ground_truth,
                    completion=output_text,
                    exact_match=exact_match,
                    score=score,
                    normalized_score=normalized_score,
                )

                result_dict = {
                    "subject": result.subject,
                    "prompt": result.prompt,
                    "input_grid": result.input_grid,
                    "ground_truth": result.ground_truth,
                    "completion": result.completion,
                    "exact_match": result.exact_match,
                    "score": result.score,
                    "normnalized_score": result.normalized_score,
                }

                results.append(result_dict)
                existing_results[sample.prompt] = result_dict

                # Save after each sample
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)

                if len(results) >= 100:
                    print(f"Processed {len(results)} samples for {model}/{subject}.")
                    break

            except Exception as e:
                print(f"Error processing prompt {sample.prompt[:60]}...: {e}")

    # Only aggregate if all samples were processed
    if len(results) >= len(samples):
        print(f"{model} results for {subject}:")

        full_results = [
            Result(
                subject=r["subject"],
                prompt=r["prompt"],
                input_grid=r["input_grid"],
                ground_truth=r["ground_truth"],
                completion=r["completion"],
                exact_match=r["exact_match"],
                score=r["score"],
                normalized_score=r["normnalized_score"],
            )
            for r in results
        ]

        aggregated_results = aggregate_results(full_results)

        print(f"Total exact match: {aggregated_results['total_exact_match']}")
        print(f"Total score: {aggregated_results['total_score']}")
        print(f"Total normalized score: {aggregated_results['total_normalized_score']}")

        with open(f"{root_dir}/{subject}_aggregated_results.json", "w") as f:
            json.dump(aggregated_results, f)
    else:
        print(
            f"Skipping aggregation for {model}/{subject} – only {len(existing_results)} of {len(samples)} samples completed."
        )


def run_main_experiment():
    subjects = [
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

    models = [
        "gpt-4.1-2025-04-14",
        "claude-3-7-sonnet-20250219",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "claude-opus-4-20250514",
        "gpt-4o-2024-08-06",
        "gemini-1.5-pro",
        "gpt-3.5-turbo-0125",
        "perplexity-sonar",
        "perplexity-sonar-reasoning",
    ]

    for subject in tqdm(subjects):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)

        samples = generate_prompts(dataset=dataset_list, subject=subject)

        for model in models:
            evaluate_against_model(model=model, samples=samples)


def run_rotation_comparison_experiment():
    subjects = [
        "10_random_cell_easy",
        "3_random_row_easy",
        "3_random_column_easy",
        "full_easy",
    ]

    models = [
        "gpt-4.1-2025-04-14",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "claude-opus-4-20250514",
        "perplexity-sonar",
    ]

    for subject in tqdm(subjects):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)

        rotations = 3

        samples = generate_prompts(
            dataset=dataset_list, subject=subject, rotation_count=rotations
        )

        for model in models:
            evaluate_against_model(
                model=model, samples=samples, name_suffix=f"_{rotations}_rotations"
            )


def run_rotation_best_model_experiment():
    subjects = [
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

    models = [
        "claude-opus-4-20250514",
    ]

    for subject in tqdm(subjects):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)

        rotations = 3

        samples = generate_prompts(
            dataset=dataset_list, subject=subject, rotation_count=rotations
        )

        for model in models:
            evaluate_against_model(
                model=model, samples=samples, name_suffix=f"_{rotations}_rotations"
            )


def run_few_shot_experiment(few_shot_count=1):
    subjects = [
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

    models = [
        "claude-opus-4-20250514",
    ]

    for subject in tqdm(subjects):
        dataset = load_dataset("philippds/SPhyR", subject)
        dataset_list = list(dataset["test"])
        rnd.shuffle(dataset_list)

        samples = generate_prompts(
            dataset=dataset_list, subject=subject, few_shot_count=few_shot_count
        )

        for model in models:
            evaluate_against_model(
                model=model, samples=samples, name_suffix=f"_few_shot_{few_shot_count}"
            )


if __name__ == "__main__":
    run_main_experiment()
    run_rotation_comparison_experiment()
    run_rotation_best_model_experiment()
    run_few_shot_experiment(1)
    run_few_shot_experiment(3)

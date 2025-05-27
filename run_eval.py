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

from prompt_templates import PROMT_TEMPLATE

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
    ground_truth: str
    completion: str
    exact_match: bool
    score: float
    normalized_score: float


def generate_prompts(dataset, subject, sample_count=SAMPLE_COUNT) -> list[Sample]:
    samples = []

    for sample in dataset[:sample_count]:
        input_grid = sample["input_grid"]

        if subject.endswith("easy"):
            prompt = PROMT_TEMPLATE.format(
                FILL_INSTRUCTION="'V' cells with either '1' (solid) or '0' (empty)",
                GRID=input_grid,
            )
        else:
            prompt = PROMT_TEMPLATE.format(
                FILL_INSTRUCTION="'V' cells with a floating point number between 0 and 1, with one decimal place (e.g., 0.0, 0.1, 0.2, ..., 1.0)",
                GRID=input_grid,
            )

        ground_truth = sample["ground_truth"]

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
    response = client.responses.create(model="gpt-4.1", input=prompt)
    return response.output_text.strip()


def run_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GEM"))
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06",
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


def run_deepkseek(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("R1"), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    return response.choices[0].message.content


def evaluate_against_model(model, samples) -> list:
    if model == "gpt-4.1":
        eval_fn = run_openai
    elif model == "gemini-2.5-pro-preview-05-06":
        eval_fn = run_gemini
    elif model == "claude-3-7-sonnet-20250219":
        eval_fn = run_claude
    elif model == "deepseek-reasoner":
        eval_fn = run_deepkseek
    else:
        raise ValueError(f"Unknown model: {model}")

    os.makedirs(f"results/{model}", exist_ok=True)
    subject = samples[0].subject if samples else "unknown"

    # Load existing results if available
    results_path = f"results/{model}/{subject}_results.json"
    existing_results = {}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing_result_dicts = json.load(f)
            for r in existing_result_dicts:
                existing_results[r["prompt"]] = r

    results = list(existing_results.values())

    for sample in tqdm(samples):
        if sample.prompt in existing_results:
            print(f"Skipping existing sample for prompt: {sample.prompt[:60]}...")
            continue

        try:
            output_text = eval_fn(sample.prompt)

            raw_input_list = [line.split() for line in sample.raw_input.splitlines()]
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
                ground_truth=sample.ground_truth,
                completion=output_text,
                exact_match=exact_match,
                score=score,
                normalized_score=normalized_score,
            )

            result_dict = {
                "subject": result.subject,
                "prompt": result.prompt,
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

        except Exception as e:
            print(f"Error processing prompt {sample.prompt[:60]}...: {e}")

    # Only aggregate if all samples were processed
    if len(existing_results) == len(samples):
        print(f"{model} results for {subject}:")

        full_results = [
            Result(
                subject=r["subject"],
                prompt=r["prompt"],
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

        with open(f"results/{model}/{subject}_aggregated_results.json", "w") as f:
            json.dump(aggregated_results, f)
    else:
        print(
            f"Skipping aggregation for {model}/{subject} – only {len(existing_results)} of {len(samples)} samples completed."
        )


models = [
    "gpt-4.1",
    "claude-3-7-sonnet-20250219",
    "gemini-2.5-pro-preview-05-06",
    "deepseek-reasoner",
]

for subject in tqdm(subjects):
    dataset = load_dataset("philippds/SPhyR", subject)
    dataset_list = list(dataset["test"])
    rnd.shuffle(dataset_list)

    samples = generate_prompts(dataset=dataset_list, subject=subject)

    for model in models:
        evaluate_against_model(model=model, samples=samples)

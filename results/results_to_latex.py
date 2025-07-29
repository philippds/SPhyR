import os
import json
from pathlib import Path

# Base directory and output file
RESULTS_DIR = Path("results")

# Mapping from JSON metric keys to readable labels
metric_map = {
    "total_exact_match": "Exact Match",
    "total_score": "Score",
    "total_normalized_score": "Normalized Score",
}


# Converts a subject string into (task name, difficulty)
def subject_to_task_and_difficulty(subject):
    parts = subject.split("_")
    difficulty = parts[-1].capitalize()
    if parts[0] == "full":
        task = "Full"
    else:
        count = parts[0]
        unit = parts[2].capitalize()
        task = f"{count} Random {unit}"
        if count != "1":
            task += "s"
    return task, difficulty


# Loads all aggregated results
def find_and_parse_results(model_order, subject_map):
    results = {}
    for model_folder in model_order:
        subdir = RESULTS_DIR / model_folder
        if not subdir.is_dir():
            continue
        results[model_folder] = {}

        for file in subdir.glob("*_aggregated_results.json"):
            for subject, (task, difficulty) in subject_map.items():
                if file.name.startswith(subject):
                    with open(file, "r") as f:
                        data = json.load(f)
                    results[model_folder].setdefault(difficulty, {}).setdefault(
                        task, {}
                    )
                    for key, value in data.items():
                        metric = metric_map.get(key, key)
                        results[model_folder][difficulty][task][metric] = value
                    break
    return results


# Generate task rows for Easy/Hard
def get_table_rows(results, tasks_order, metrics_order, models):
    rows = []
    for difficulty, tasks in tasks_order.items():
        rows.append(r"\midrule")
        rows.append(f"Difficulty: {difficulty} &&&&&\\\\")
        rows.append(r"\midrule")
        for task in tasks:
            for metric in metrics_order:
                row = f"{task if metric == metrics_order[0] else '':<30} & {metric:<20}"
                for model in models:
                    raw_value = (
                        results.get(model, {})
                        .get(difficulty, {})
                        .get(task, {})
                        .get(metric, "")
                    )
                    if isinstance(raw_value, (int, float)):
                        value = (
                            f"{raw_value:.2f}"
                            if metric != "Exact Match"
                            else f"{int(raw_value)}"
                        )
                    else:
                        value = raw_value
                    row += f" & {value}"
                row += r" \\"
                rows.append(row)
    return rows


# Compute averages across tasks
def compute_averages(results, tasks_order, metrics_order, models):
    averages = {"Easy": {}, "Hard": {}, "Overall": {}}

    for difficulty in ["Easy", "Hard"]:
        for metric in metrics_order:
            for model in models:
                total = 0
                count = 0
                for task in tasks_order[difficulty]:
                    value = (
                        results.get(model, {})
                        .get(difficulty, {})
                        .get(task, {})
                        .get(metric)
                    )
                    if isinstance(value, (int, float)):
                        total += value
                        count += 1
                avg = total / count if count > 0 else ""
                averages[difficulty].setdefault(metric, {})[model] = avg

    for metric in metrics_order:
        for model in models:
            easy_avg = averages["Easy"][metric].get(model)
            hard_avg = averages["Hard"][metric].get(model)
            if isinstance(easy_avg, (int, float)) and isinstance(
                hard_avg, (int, float)
            ):
                overall = (easy_avg + hard_avg) / 2
            else:
                overall = ""
            averages["Overall"].setdefault(metric, {})[model] = overall

    return averages


# Render average rows
def get_average_rows(averages, label, metrics_order, models, bold=False):
    rows = [r"\midrule"]
    for metric in metrics_order:
        row_label = f"\\textbf{{{label}}}" if bold else label
        metric_label = (
            f"\\textbf{{{metric}}}" if bold and metric == "Exact Match" else metric
        )
        row = f"{row_label:<30} & {metric_label:<20}"
        for model in models:
            value = averages[label].get(metric, {}).get(model, "")
            if isinstance(value, float):
                value = f"{value:.2f}" if metric != "Exact Match" else f"{value:.2f}"
            row += f" & {value}"
        row += r" \\"
        rows.append(row)
    return rows


# Full LaTeX table generation
def generate_latex_table(results, model_order, model_name_map):
    models = [m for m in model_order if m in results]

    tasks_order = {
        "Easy": [
            "1 Random Cell",
            "5 Random Cells",
            "10 Random Cells",
            "1 Random Row",
            "3 Random Rows",
            "1 Random Column",
            "3 Random Columns",
            "Full",
        ],
        "Hard": [
            "1 Random Cell",
            "5 Random Cells",
            "10 Random Cells",
            "1 Random Row",
            "3 Random Rows",
            "1 Random Column",
            "3 Random Columns",
            "Full",
        ],
    }
    metrics_order = ["Exact Match", "Score", "Normalized Score"]
    averages = compute_averages(results, tasks_order, metrics_order, models)

    header = [
        r"\begin{table}[!h]",
        r"  \caption{Accuracy (\%) and Overlap Score (normalized) on 2D tasks, separated by difficulty.}",
        r"  \label{tab:accuracy_2d}",
        r"  \centering",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{",
        f"  \\begin{{tabular}}{{ll{'l'*len(models)}}}",
        r"    \toprule",
        f"    Task & Metric & {' & '.join(model_name_map[m] for m in models)} \\\\",
    ]

    body = []

    # Easy
    body += get_table_rows(
        results, {"Easy": tasks_order["Easy"]}, metrics_order, models
    )
    body += get_average_rows(averages, "Easy", metrics_order, models, bold=True)

    # Hard
    body += get_table_rows(
        results, {"Hard": tasks_order["Hard"]}, metrics_order, models
    )
    body += get_average_rows(averages, "Hard", metrics_order, models)

    # Overall
    body.append(r"\midrule")
    body.append(r"\midrule")
    body += get_average_rows(averages, "Overall", metrics_order, models, bold=True)

    footer = [r"    \bottomrule", r"  \end{tabular}", r"  }", r"\end{table}"]

    return "\n".join(header + body + footer)


# Entrypoint
def results_to_latex_main_body(output_latex_file):
    # Ordered model folder names (must match actual folder names exactly)
    model_order = [
        "gpt-4.1-2025-04-14",
        "claude-opus-4-20250514",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "perplexity-sonar",
    ]

    # Mapping from model folder names to human-friendly display names
    model_name_map = {
        "gpt-4.1-2025-04-14": "GPT 4.1",
        "claude-opus-4-20250514": "Claude Opus 4",
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
        "deepseek-reasoner": "DeepSeek-R1",
        "perplexity-sonar": "Perplexity Sonar",
    }

    # Known subjects used in filenames
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

    # Map subject to (task, difficulty)
    subject_map = {
        subject: subject_to_task_and_difficulty(subject) for subject in subjects
    }

    results = find_and_parse_results(model_order, subject_map)
    latex_code = generate_latex_table(results, model_order, model_name_map)

    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table written to: {output_latex_file}")


def results_to_latex_appendix(output_latex_file):
    # Ordered model folder names (must match actual folder names exactly)
    model_order = [
        "gpt-3.5-turbo-0125",
        "gpt-4.1-2025-04-14",
        "gpt-4o-2024-08-06",
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-20250514",
        "gemini-1.5-pro",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "perplexity-sonar",
        "perplexity-sonar-reasoning",
    ]

    # Mapping from model folder names to human-friendly display names
    model_name_map = {
        "gpt-3.5-turbo-0125": "GPT 3.5 Turbo",
        "gpt-4.1-2025-04-14": "GPT 4.1",
        "gpt-4o-2024-08-06": "GPT 4o",
        "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
        "claude-opus-4-20250514": "Claude Opus 4",
        "gemini-1.5-pro": "Gemini 1.5 Pro",
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
        "deepseek-reasoner": "DeepSeek-R1",
        "perplexity-sonar": "Perplexity Sonar",
        "perplexity-sonar-reasoning": "Perplexity Sonar Reasoning",
    }

    # Known subjects used in filenames
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

    # Map subject to (task, difficulty)
    subject_map = {
        subject: subject_to_task_and_difficulty(subject) for subject in subjects
    }

    results = find_and_parse_results(model_order, subject_map)
    latex_code = generate_latex_table(results, model_order, model_name_map)

    with open(output_latex_file, "w") as f:
        f.write(latex_code)


def rotation_comparison_results_to_latex_main_body(output_latex_file):
    # Ordered model folder names (must match actual folder names exactly)
    model_order = [
        "gpt-4.1-2025-04-14",
        "gpt-4.1-2025-04-14_3_rotations",
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_3_rotations",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro-preview-05-06_3_rotations",
        "deepseek-reasoner",
        "deepseek-reasoner_3_rotations",
        "perplexity-sonar",
        "perplexity-sonar_3_rotations",
    ]

    # Mapping from model folder names to human-friendly display names
    model_name_map = {
        "gpt-4.1-2025-04-14": "GPT 4.1",
        "gpt-4.1-2025-04-14_3_rotations": "GPT 4.1 (3 Rotations)",
        "claude-opus-4-20250514": "Claude Opus 4",
        "claude-opus-4-20250514_3_rotations": "Claude Opus 4 (3 Rotations)",
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
        "gemini-2.5-pro-preview-05-06_3_rotations": "Gemini 2.5 Pro (3 Rotations)",
        "deepseek-reasoner": "DeepSeek-R1",
        "deepseek-reasoner_3_rotations": "DeepSeek-R1 (3 Rotations)",
        "perplexity-sonar": "Perplexity Sonar",
        "perplexity-sonar_3_rotations": "Perplexity Sonar (3 Rotations)",
    }

    # Known subjects used in filenames
    subjects = [
        "10_random_cell_easy",
        "3_random_row_easy",
        "3_random_column_easy",
        "full_easy",
    ]

    # Map subject to (task, difficulty)
    subject_map = {
        subject: subject_to_task_and_difficulty(subject) for subject in subjects
    }

    results = find_and_parse_results(model_order, subject_map)
    latex_code = generate_latex_table(results, model_order, model_name_map)

    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table written to: {output_latex_file}")


def rotation_best_model_results_to_latex(output_latex_file):
    # Ordered model folder names (must match actual folder names exactly)
    model_order = [
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_3_rotations",
    ]

    # Mapping from model folder names to human-friendly display names
    model_name_map = {
        "claude-opus-4-20250514": "Claude Opus 4",
        "claude-opus-4-20250514_3_rotations": "Claude Opus 4 (3 Rotations)",
    }

    # Known subjects used in filenames
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

    # Map subject to (task, difficulty)
    subject_map = {
        subject: subject_to_task_and_difficulty(subject) for subject in subjects
    }

    results = find_and_parse_results(model_order, subject_map)
    latex_code = generate_latex_table(results, model_order, model_name_map)

    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table written to: {output_latex_file}")


if __name__ == "__main__":
    # results_to_latex_main_body(RESULTS_DIR / "results_to_latex_main_body.tex")

    # results_to_latex_appendix(RESULTS_DIR / "results_to_latex_appendix.tex")

    # rotation_comparison_results_to_latex_main_body(
    #     RESULTS_DIR / "rotation_comparison_results_to_latex.tex"
    # )

    rotation_best_model_results_to_latex(
        RESULTS_DIR / "rotation_best_model_results_to_latex.tex"
    )

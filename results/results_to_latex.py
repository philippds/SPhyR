import os
import json
from pathlib import Path

# Base directory and output file
RESULTS_DIR = Path("results")

# Mapping from JSON metric keys to readable labels
METRICS_MAP = {
    "total_exact_match": "Exact Match",
    "total_difference_ratio": "Difference Ratio (%)",
    "total_penalized_difference_ratio": "Penalized Difference Ratio (%)",
    "total_relative_difference_ratio": "Relative Difference Ratio (%)",
    "total_valid_output_grid": "Valid Output Grid (%)",
    "total_load_support_connected": "Load-Support Connectivity (%)",
    "total_load_support_connected_force_directional": "Load-Support Directional Connectivity (%)",
    "total_isolated_clusters_count": "Average Isolated Clusters Count",
    "total_force_path_cost_average_efficiency_ratio": "Force Path Cost Average Efficiency Ratio (%)",
    "total_difficulty_score": "Average Difficulty Score",
    "total_difficulty_weighted_difference_ratio": "Difficulty Weighted Difference Ratio (%)",
    "total_difficulty_weighted_relative_difference_ratio": "Difficulty Weighted Relative Difference Ratio (%)",
}


METRICS_ORDER = [
    "Exact Match",
    "Difference Ratio (%)",
    "Relative Difference Ratio (%)",
    "Penalized Difference Ratio (%)",
    "Average Difficulty Score",
    "Difficulty Weighted Difference Ratio (%)",
    "Difficulty Weighted Relative Difference Ratio (%)",
    "Valid Output Grid (%)",
    "Load-Support Connectivity (%)",
    "Load-Support Directional Connectivity (%)",
    "Average Isolated Clusters Count",
    "Force Path Cost Average Efficiency Ratio (%)",
]


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
                        metric = METRICS_MAP.get(key, key)
                        if (
                            metric == "Average Difficulty Score"
                            or metric == "Average Isolated Clusters Count"
                        ):
                            value = value / 100.0
                        if (
                            metric == "Difficulty Weighted Difference Ratio (%)"
                            or metric
                            == "Difficulty Weighted Relative Difference Ratio (%)"
                        ):
                            value = value / 3
                        results[model_folder][difficulty][task][metric] = value
                    break
    return results


# Generate task rows for Easy/Hard
def get_table_rows(results, tasks_order, models):
    rows = []
    for difficulty, tasks in tasks_order.items():
        rows.append(r"\midrule")
        rows.append(f"Difficulty: {difficulty} &&&&&&\\\\")
        rows.append(r"\midrule")
        for task in tasks:
            for metric in METRICS_ORDER:
                row = f"{task if metric == METRICS_ORDER[0] else '':<30} & {metric:<20}"
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
def compute_averages(results, tasks_order, models):
    averages = {"Easy": {}, "Hard": {}, "Overall": {}}

    for difficulty in ["Easy", "Hard"]:
        for metric in METRICS_ORDER:
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

    for metric in METRICS_ORDER:
        for model in models:
            easy_avg = averages["Easy"][metric].get(model)
            hard_avg = averages["Hard"][metric].get(model)
            if isinstance(easy_avg, (int, float)) and isinstance(
                hard_avg, (int, float)
            ):
                overall = (easy_avg + hard_avg) / 2
            elif isinstance(easy_avg, (int, float)):
                overall = easy_avg
            elif isinstance(hard_avg, (int, float)):
                overall = hard_avg
            else:
                overall = ""
            averages["Overall"].setdefault(metric, {})[model] = overall

    return averages


# Render average rows
def get_average_rows(averages, label, models, bold=False):
    rows = [r"\midrule"]
    for metric in METRICS_ORDER:
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

    averages = compute_averages(results, tasks_order, models)

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
        r"    \midrule",
    ]

    body = []

    for difficulty, tasks in tasks_order.items():
        body.append(r"\midrule")
        body.append(
            f"\\multicolumn{{{2 + len(models)}}}{{l}}{{\\textbf{{Difficulty: {difficulty}}}}}\\\\"
        )
        body.append(r"\midrule")

        for task in tasks:
            for metric in METRICS_ORDER:
                row = []
                row.append(task if metric == METRICS_ORDER[0] else "")
                row.append(metric)

                # collect numeric values
                values = []
                for model in models:
                    val = (
                        results.get(model, {})
                        .get(difficulty, {})
                        .get(task, {})
                        .get(metric, "")
                    )
                    values.append(val if isinstance(val, (int, float)) else None)

                best_val = max(
                    [v for v in values if isinstance(v, (int, float))], default=None
                )

                # render cells
                for model, val in zip(models, values):
                    formatted = ""
                    if isinstance(val, (int, float)):
                        formatted = (
                            f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                        )
                        if val < 0:
                            formatted = f"\\textcolor{{red}}{{{formatted}}}"
                        if val == best_val:
                            formatted = f"\\textbf{{{formatted}}}"
                    row.append(formatted)
                body.append(" & ".join(row) + r" \\")

        # averages per difficulty
        body.append(r"\midrule")
        for metric in METRICS_ORDER:
            row = [r"\textbf{Average}" if metric == METRICS_ORDER[0] else "", metric]
            vals = [averages[difficulty].get(metric, {}).get(m, "") for m in models]
            best_val = max(
                [v for v in vals if isinstance(v, (int, float))], default=None
            )
            for val in vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{val:.2f}" if metric != "Exact Match" else f"{int(val)}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == best_val:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)
            body.append(" & ".join(row) + r" \\")

    # overall averages
    body.append(r"\midrule\midrule")
    for metric in METRICS_ORDER:
        row = [r"\textbf{Overall}" if metric == METRICS_ORDER[0] else "", metric]
        vals = [averages["Overall"].get(metric, {}).get(m, "") for m in models]
        best_val = max([v for v in vals if isinstance(v, (int, float))], default=None)
        for val in vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}" if metric != "Exact Match" else f"{int(val)}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == best_val:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)
        body.append(" & ".join(row) + r" \\")

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


def generate_latex_table_side_by_side(results, model_order, model_name_map):
    models = [m for m in model_order if m in results]

    tasks_order = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Column",
        "3 Random Columns",
        "Full",
    ]

    # Compute averages using your existing helper
    averages = compute_averages(
        results, {"Easy": tasks_order, "Hard": tasks_order}, models
    )

    # --- HEADER ---
    header = [
        r"\begin{table*}[!h]",
        r"  \caption{Accuracy (\%) and Overlap Score (normalized) on 2D tasks, Easy vs Hard side by side.}",
        r"  \label{tab:accuracy_2d_side_by_side}",
        r"  \centering",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{",
        f"  \\begin{{tabular}}{{ll{'l'*len(models)}|{'l'*len(models)}}}",
        r"    \toprule",
        f"    \\multicolumn{{{2 + len(models)}}}{{c}}{{\\textbf{{Easy}}}} & "
        f"\\multicolumn{{{len(models)}}}{{c}}{{\\textbf{{Hard}}}} \\\\",
        f"    Task & Metric & {' & '.join(model_name_map[m] for m in models)} & "
        f"{' & '.join(model_name_map[m] for m in models)} \\\\",
        r"    \midrule",
    ]

    # --- BODY ---
    body = []
    for task in tasks_order:
        for metric in METRICS_ORDER:
            row_parts = []

            # --- Easy values ---
            easy_values = []
            for model in models:
                value = (
                    results.get(model, {}).get("Easy", {}).get(task, {}).get(metric, "")
                )
                easy_values.append(value if isinstance(value, (int, float)) else None)

            # --- Hard values ---
            hard_values = []
            for model in models:
                value = (
                    results.get(model, {}).get("Hard", {}).get(task, {}).get(metric, "")
                )
                hard_values.append(value if isinstance(value, (int, float)) else None)

            # Find bests (ignore NaNs or missing)
            easy_best = (
                max([v for v in easy_values if isinstance(v, (int, float))])
                if any(isinstance(v, (int, float)) for v in easy_values)
                else None
            )
            hard_best = (
                max([v for v in hard_values if isinstance(v, (int, float))])
                if any(isinstance(v, (int, float)) for v in hard_values)
                else None
            )

            # --- Left side (Easy) ---
            row_parts.append(task if metric == METRICS_ORDER[0] else "")
            row_parts.append(metric)
            for model, value in zip(models, easy_values):
                formatted = ""
                if isinstance(value, (int, float)):
                    formatted = (
                        f"{int(value)}" if metric == "Exact Match" else f"{value:.2f}"
                    )
                    if value < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if value == easy_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row_parts.append(formatted)

            # --- Right side (Hard) ---
            for model, value in zip(models, hard_values):
                formatted = ""
                if isinstance(value, (int, float)):
                    formatted = (
                        f"{int(value)}" if metric == "Exact Match" else f"{value:.2f}"
                    )
                    if value < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if value == hard_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row_parts.append(formatted)

            body.append(" & ".join(row_parts) + r" \\")
        body.append(r"\midrule")

    # --- Add average rows (Easy and Hard) ---
    for index, metric in enumerate(METRICS_ORDER):
        if index == 0:
            avg_row = [r"\textbf{Average}", f"{metric}"]
        else:
            avg_row = ["", f"{metric}"]

        # Easy averages
        easy_vals = []
        for model in models:
            val = averages["Easy"].get(metric, {}).get(model, "")
            easy_vals.append(val if isinstance(val, (int, float)) else None)
        easy_best = (
            max([v for v in easy_vals if isinstance(v, (int, float))])
            if any(isinstance(v, (int, float)) for v in easy_vals)
            else None
        )

        for model, val in zip(models, easy_vals):
            formatted = ""
            if isinstance(val, float):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == easy_best:
                    formatted = f"\\textbf{{{formatted}}}"
            avg_row.append(formatted)

        # Hard averages
        hard_vals = []
        for model in models:
            val = averages["Hard"].get(metric, {}).get(model, "")
            hard_vals.append(val if isinstance(val, (int, float)) else None)
        hard_best = (
            max([v for v in hard_vals if isinstance(v, (int, float))])
            if any(isinstance(v, (int, float)) for v in hard_vals)
            else None
        )

        for model, val in zip(models, hard_vals):
            formatted = ""
            if isinstance(val, float):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == hard_best:
                    formatted = f"\\textbf{{{formatted}}}"
            avg_row.append(formatted)

        body.append(" & ".join(avg_row) + r" \\")

    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table*}",
    ]

    return "\n".join(header + body + footer)


def results_to_latex_main_body_side_by_side(output_latex_file):
    model_order = [
        "gpt-4.1-2025-04-14",
        "claude-opus-4-20250514",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "perplexity-sonar",
    ]

    model_name_map = {
        "gpt-4.1-2025-04-14": "GPT 4.1",
        "claude-opus-4-20250514": "Claude Opus 4",
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
        "deepseek-reasoner": "DeepSeek-R1",
        "perplexity-sonar": "Perplexity Sonar",
    }

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

    subject_map = {
        subject: subject_to_task_and_difficulty(subject) for subject in subjects
    }
    results = find_and_parse_results(model_order, subject_map)
    latex_code = generate_latex_table_side_by_side(results, model_order, model_name_map)

    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(f"Side-by-side LaTeX table written to: {output_latex_file}")


def rotation_comparison_results_to_latex_main_body_side_by_side(output_latex_file):
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

    # Known subjects used in filenames (these are only "Easy" tasks)
    subjects = [
        "10_random_cell_easy",
        "3_random_row_easy",
        "3_random_column_easy",
        "full_easy",
        "10_random_cell_hard",
        "3_random_row_hard",
        "3_random_column_hard",
        "full_hard",
    ]

    # Map subject to (task, difficulty)
    subject_map = {
        subject: subject_to_task_and_difficulty(subject) for subject in subjects
    }

    # Parse all results
    results = find_and_parse_results(model_order, subject_map)

    # Generate side-by-side LaTeX
    latex_code = generate_latex_table_side_by_side(results, model_order, model_name_map)

    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(
        f"Side-by-side rotation comparison LaTeX table written to: {output_latex_file}"
    )


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


def rotation_best_model_results_to_latex_side_by_side(output_latex_file):
    # --- Model order and names ---
    model_order = [
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_3_rotations",
    ]

    model_name_map = {
        "claude-opus-4-20250514": "Claude Opus 4",
        "claude-opus-4-20250514_3_rotations": "Claude Opus 4 (3 Rotations)",
    }

    # --- Subjects ---
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

    subject_map = {s: subject_to_task_and_difficulty(s) for s in subjects}

    # --- Parse results ---
    results = find_and_parse_results(model_order, subject_map)

    # --- Table layout ---
    models = [m for m in model_order if m in results]
    tasks_order = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Column",
        "3 Random Columns",
        "Full",
    ]
    averages = compute_averages(
        results, {"Easy": tasks_order, "Hard": tasks_order}, models
    )

    # --- Header ---
    header = [
        r"\begin{table*}[!h]",
        r"  \caption{Claude Opus 4 with and without rotation augmentation --- Easy vs Hard side by side.}",
        r"  \label{tab:rotation_best_model_side_by_side}",
        r"  \centering",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{",
        f"  \\begin{{tabular}}{{ll{'l'*len(models)}|{'l'*len(models)}}}",
        r"    \toprule",
        f"    \\multicolumn{{{2 + len(models)}}}{{c}}{{\\textbf{{Easy}}}} & "
        f"\\multicolumn{{{len(models)}}}{{c}}{{\\textbf{{Hard}}}} \\\\",
        f"    Task & Metric & {' & '.join(model_name_map[m] for m in models)} & "
        f"{' & '.join(model_name_map[m] for m in models)} \\\\",
        r"    \midrule",
    ]

    # --- Body ---
    body = []
    for task in tasks_order:
        for metric in METRICS_ORDER:
            row = []

            # collect easy/hard numeric values
            easy_vals = [
                results.get(m, {}).get("Easy", {}).get(task, {}).get(metric, "")
                for m in models
            ]
            hard_vals = [
                results.get(m, {}).get("Hard", {}).get(task, {}).get(metric, "")
                for m in models
            ]

            easy_best = max(
                [v for v in easy_vals if isinstance(v, (int, float))], default=None
            )
            hard_best = max(
                [v for v in hard_vals if isinstance(v, (int, float))], default=None
            )

            # left half
            row.append(task if metric == METRICS_ORDER[0] else "")
            row.append(metric)
            for val in easy_vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == easy_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)

            # right half
            for val in hard_vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == hard_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)

            body.append(" & ".join(row) + r" \\")
        body.append(r"\midrule")

    # --- Averages ---
    for i, metric in enumerate(METRICS_ORDER):
        row = [r"\textbf{Average}" if i == 0 else "", metric]

        easy_vals = [averages["Easy"].get(metric, {}).get(m, "") for m in models]
        hard_vals = [averages["Hard"].get(metric, {}).get(m, "") for m in models]
        easy_best = max(
            [v for v in easy_vals if isinstance(v, (int, float))], default=None
        )
        hard_best = max(
            [v for v in hard_vals if isinstance(v, (int, float))], default=None
        )

        for val in easy_vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == easy_best:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)

        for val in hard_vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == hard_best:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)

        body.append(" & ".join(row) + r" \\")

    # --- Footer ---
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table*}",
    ]

    latex_code = "\n".join(header + body + footer)

    # --- Write file ---
    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(
        f"Side-by-side rotation best model LaTeX table written to: {output_latex_file}"
    )


def few_shot_comparison_results_to_latex_main_body(output_latex_file):
    # Ordered model folder names (must match actual folder names exactly)
    model_order = [
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_few_shot_1",
        "claude-opus-4-20250514_few_shot_3",
    ]

    # Mapping from model folder names to human-friendly display names
    model_name_map = {
        "claude-opus-4-20250514": "Claude Opus 4",
        "claude-opus-4-20250514_few_shot_1": "Claude Opus 4 (Few Shot 1)",
        "claude-opus-4-20250514_few_shot_3": "Claude Opus 4 (Few Shot 3)",
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


def physics_enhanced_neutral_prompt_comparison_results_to_latex_main_body(
    output_latex_file,
):
    # Ordered model folder names (must match actual folder names exactly)
    model_order = [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro-preview-05-06_physics_enhanced_prompt",
        "gemini-2.5-pro-preview-05-06_physics_neutral_prompt",
    ]

    # Mapping from model folder names to human-friendly display names
    model_name_map = {
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
        "gemini-2.5-pro-preview-05-06_physics_enhanced_prompt": "Gemini 2.5 Pro (Physics Enhanced Prompt)",
        "gemini-2.5-pro-preview-05-06_physics_neutral_prompt": "Gemini 2.5 Pro (Physics Neutral Prompt)",
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


def few_shot_comparison_results_to_latex_main_body_side_by_side(output_latex_file):
    # --- Model folders (must match directory names) ---
    model_order = [
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_few_shot_1",
        "claude-opus-4-20250514_few_shot_3",
    ]

    # --- Human-readable model names ---
    model_name_map = {
        "claude-opus-4-20250514": "Claude Opus 4 (Zero-Shot)",
        "claude-opus-4-20250514_few_shot_1": "Claude Opus 4 (1-Shot)",
        "claude-opus-4-20250514_few_shot_3": "Claude Opus 4 (3-Shot)",
    }

    # --- Subjects for Easy & Hard tasks ---
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

    # --- Map to task and difficulty ---
    subject_map = {s: subject_to_task_and_difficulty(s) for s in subjects}

    # --- Load results ---
    results = find_and_parse_results(model_order, subject_map)
    models = [m for m in model_order if m in results]

    # --- Tasks and averages ---
    tasks_order = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Column",
        "3 Random Columns",
        "Full",
    ]
    averages = compute_averages(
        results, {"Easy": tasks_order, "Hard": tasks_order}, models
    )

    # --- Header ---
    header = [
        r"\begin{table*}[!h]",
        r"  \caption{Claude Opus 4 Few-Shot Comparison --- Easy vs Hard side by side.}",
        r"  \label{tab:few_shot_comparison_side_by_side}",
        r"  \centering",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{",
        f"  \\begin{{tabular}}{{ll{'l'*len(models)}|{'l'*len(models)}}}",
        r"    \toprule",
        f"    \\multicolumn{{{2 + len(models)}}}{{c}}{{\\textbf{{Easy}}}} & "
        f"\\multicolumn{{{len(models)}}}{{c}}{{\\textbf{{Hard}}}} \\\\",
        f"    Task & Metric & {' & '.join(model_name_map[m] for m in models)} & "
        f"{' & '.join(model_name_map[m] for m in models)} \\\\",
        r"    \midrule",
    ]

    # --- Body ---
    body = []
    for task in tasks_order:
        for metric in METRICS_ORDER:
            row = []

            # Get Easy/Hard values for all models
            easy_vals = [
                results.get(m, {}).get("Easy", {}).get(task, {}).get(metric, "")
                for m in models
            ]
            hard_vals = [
                results.get(m, {}).get("Hard", {}).get(task, {}).get(metric, "")
                for m in models
            ]

            easy_best = max(
                [v for v in easy_vals if isinstance(v, (int, float))], default=None
            )
            hard_best = max(
                [v for v in hard_vals if isinstance(v, (int, float))], default=None
            )

            # Left (Easy)
            row.append(task if metric == METRICS_ORDER[0] else "")
            row.append(metric)
            for val in easy_vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == easy_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)

            # Right (Hard)
            for val in hard_vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == hard_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)

            body.append(" & ".join(row) + r" \\")
        body.append(r"\midrule")

    # --- Average rows ---
    for i, metric in enumerate(METRICS_ORDER):
        row = [r"\textbf{Average}" if i == 0 else "", metric]

        easy_vals = [averages["Easy"].get(metric, {}).get(m, "") for m in models]
        hard_vals = [averages["Hard"].get(metric, {}).get(m, "") for m in models]
        easy_best = max(
            [v for v in easy_vals if isinstance(v, (int, float))], default=None
        )
        hard_best = max(
            [v for v in hard_vals if isinstance(v, (int, float))], default=None
        )

        for val in easy_vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == easy_best:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)

        for val in hard_vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == hard_best:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)

        body.append(" & ".join(row) + r" \\")

    # --- Footer ---
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table*}",
    ]

    latex_code = "\n".join(header + body + footer)

    # --- Write to file ---
    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(
        f"Side-by-side few-shot comparison LaTeX table written to: {output_latex_file}"
    )


def physics_enhanced_neutral_prompt_comparison_results_to_latex_main_body_side_by_side(
    output_latex_file,
):
    # --- Model order (must match directory names) ---
    model_order = [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro-preview-05-06_physics_enhanced_prompt",
        "gemini-2.5-pro-preview-05-06_physics_neutral_prompt",
    ]

    # --- Mapping from model folder names to human-readable display names ---
    model_name_map = {
        "gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro (Base)",
        "gemini-2.5-pro-preview-05-06_physics_enhanced_prompt": "Gemini 2.5 Pro (Physics-Enhanced Prompt)",
        "gemini-2.5-pro-preview-05-06_physics_neutral_prompt": "Gemini 2.5 Pro (Physics-Neutral Prompt)",
    }

    # --- Subjects (both difficulties) ---
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

    # --- Map subjects to (task, difficulty) ---
    subject_map = {s: subject_to_task_and_difficulty(s) for s in subjects}

    # --- Parse results ---
    results = find_and_parse_results(model_order, subject_map)
    models = [m for m in model_order if m in results]

    # --- Task order and averages ---
    tasks_order = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Column",
        "3 Random Columns",
        "Full",
    ]
    averages = compute_averages(
        results, {"Easy": tasks_order, "Hard": tasks_order}, models
    )

    # --- Header ---
    header = [
        r"\begin{table*}[!h]",
        r"  \caption{Gemini 2.5 Pro Physics Prompt Comparison --- Easy vs Hard side by side.}",
        r"  \label{tab:physics_prompt_comparison_side_by_side}",
        r"  \centering",
        r"  \small",
        r"  \resizebox{\textwidth}{!}{",
        f"  \\begin{{tabular}}{{ll{'l'*len(models)}|{'l'*len(models)}}}",
        r"    \toprule",
        f"    \\multicolumn{{{2 + len(models)}}}{{c}}{{\\textbf{{Easy}}}} & "
        f"\\multicolumn{{{len(models)}}}{{c}}{{\\textbf{{Hard}}}} \\\\",
        f"    Task & Metric & {' & '.join(model_name_map[m] for m in models)} & "
        f"{' & '.join(model_name_map[m] for m in models)} \\\\",
        r"    \midrule",
    ]

    # --- Body ---
    body = []
    for task in tasks_order:
        for metric in METRICS_ORDER:
            row = []

            # Collect values
            easy_vals = [
                results.get(m, {}).get("Easy", {}).get(task, {}).get(metric, "")
                for m in models
            ]
            hard_vals = [
                results.get(m, {}).get("Hard", {}).get(task, {}).get(metric, "")
                for m in models
            ]

            easy_best = max(
                [v for v in easy_vals if isinstance(v, (int, float))], default=None
            )
            hard_best = max(
                [v for v in hard_vals if isinstance(v, (int, float))], default=None
            )

            # Left side (Easy)
            row.append(task if metric == METRICS_ORDER[0] else "")
            row.append(metric)
            for val in easy_vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == easy_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)

            # Right side (Hard)
            for val in hard_vals:
                formatted = ""
                if isinstance(val, (int, float)):
                    formatted = (
                        f"{int(val)}" if metric == "Exact Match" else f"{val:.2f}"
                    )
                    if val < 0:
                        formatted = f"\\textcolor{{red}}{{{formatted}}}"
                    if val == hard_best:
                        formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)

            body.append(" & ".join(row) + r" \\")
        body.append(r"\midrule")

    # --- Average rows ---
    for i, metric in enumerate(METRICS_ORDER):
        row = [r"\textbf{Average}" if i == 0 else "", metric]

        easy_vals = [averages["Easy"].get(metric, {}).get(m, "") for m in models]
        hard_vals = [averages["Hard"].get(metric, {}).get(m, "") for m in models]
        easy_best = max(
            [v for v in easy_vals if isinstance(v, (int, float))], default=None
        )
        hard_best = max(
            [v for v in hard_vals if isinstance(v, (int, float))], default=None
        )

        for val in easy_vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == easy_best:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)

        for val in hard_vals:
            formatted = ""
            if isinstance(val, (int, float)):
                formatted = f"{val:.2f}"
                if val < 0:
                    formatted = f"\\textcolor{{red}}{{{formatted}}}"
                if val == hard_best:
                    formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)

        body.append(" & ".join(row) + r" \\")

    # --- Footer ---
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table*}",
    ]

    latex_code = "\n".join(header + body + footer)

    # --- Write output ---
    with open(output_latex_file, "w") as f:
        f.write(latex_code)

    print(
        f"Side-by-side physics prompt comparison LaTeX table written to: {output_latex_file}"
    )


if __name__ == "__main__":
    # results_to_latex_main_body(RESULTS_DIR / "results_to_latex_main_body.tex")

    # results_to_latex_appendix(RESULTS_DIR / "results_to_latex_appendix.tex")

    # rotation_comparison_results_to_latex_main_body(
    #     RESULTS_DIR / "rotation_comparison_results_to_latex.tex"
    # )
    # rotation_best_model_results_to_latex(
    #     RESULTS_DIR / "rotation_best_model_results_to_latex.tex"
    # )

    # rotation_best_model_results_to_latex_side_by_side(
    #     RESULTS_DIR / "rotation_best_model_results_to_latex_side_by_side.tex"
    # )

    # few_shot_comparison_results_to_latex_main_body(
    #     RESULTS_DIR / "few_shot_comparison_results_to_latex.tex"
    # )

    # few_shot_comparison_results_to_latex_main_body_side_by_side(
    #     RESULTS_DIR / "few_shot_comparison_results_to_latex_side_by_side.tex"
    # )

    # physics_enhanced_neutral_prompt_comparison_results_to_latex_main_body(
    #     RESULTS_DIR / "physics_enhanced_neutral_prompt_comparison_results_to_latex.tex"
    # )

    physics_enhanced_neutral_prompt_comparison_results_to_latex_main_body_side_by_side(
        RESULTS_DIR
        / "physics_enhanced_neutral_prompt_comparison_results_to_latex_side_by_side.tex"
    )

    # SIDE BY SIDE
    # results_to_latex_main_body_side_by_side(
    #     RESULTS_DIR / "results_to_latex_main_body_side_by_side.tex"
    # )

    # rotation_comparison_results_to_latex_main_body_side_by_side(
    #     RESULTS_DIR / "rotation_comparison_results_to_latex_side_by_side.tex"
    # )

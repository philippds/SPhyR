import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Base colors for models (tab10 for main models)
base_colors = cm.get_cmap("tab10")
model_color_map = {
    "gpt-3.5-turbo-0125": base_colors(4),
    "gpt-4.1-2025-04-14": base_colors(1),
    "gpt-4o-2024-08-06": base_colors(2),
    "claude-3-7-sonnet-20250219": base_colors(3),
    "claude-opus-4-20250514": base_colors(0),
    "gemini-1.5-pro": base_colors(5),
    "gemini-2.5-pro-preview-05-06": base_colors(6),
    "deepseek-reasoner": base_colors(7),
    "perplexity-sonar": base_colors(8),
    "perplexity-sonar-reasoning": base_colors(9),
}

rotation_colors = cm.get_cmap("tab20")  # used for rotation and few-shot variants

ALL_TASKS_BY_DIFFICULTY = {
    "easy": [
        "1_random_cell_easy",
        "5_random_cell_easy",
        "10_random_cell_easy",
        "1_random_row_easy",
        "3_random_row_easy",
        "1_random_column_easy",
        "3_random_column_easy",
        "full_easy",
    ],
    "hard": [
        "1_random_cell_hard",
        "5_random_cell_hard",
        "10_random_cell_hard",
        "1_random_row_hard",
        "3_random_row_hard",
        "1_random_column_hard",
        "3_random_column_hard",
        "full_hard",
    ],
}


def adjust_color_brightness(color, factor):
    """Lighten or darken a given RGBA color."""
    r, g, b, a = color
    return (min(r * factor, 1), min(g * factor, 1), min(b * factor, 1), a)


def get_model_color(model_name):
    base_name = (
        model_name.replace("_3_rotations", "")
        .replace("_few_shot_1", "")
        .replace("_few_shot_3", "")
        .replace("_physics_enhanced_prompt", "")
        .replace("_physics_neutral_prompt", "")
    )
    base_color = model_color_map.get(base_name, "gray")

    if "_3_rotations" in model_name:
        return adjust_color_brightness(base_color, 1.3)  # slightly lighter

    elif "_few_shot_1" in model_name:
        return adjust_color_brightness(base_color, 1.3)  # slightly brighter
    elif "_few_shot_3" in model_name:
        return adjust_color_brightness(base_color, 1.6)  # even brighter
    elif "_physics_enhanced_prompt" in model_name:
        return adjust_color_brightness(base_color, 1.3)
    elif "_physics_neutral_prompt" in model_name:
        return adjust_color_brightness(base_color, 1.6)
    else:
        return base_color


# "total_exact_match": total_exact_match,
# "total_difference_ratio": total_difference_ratio,
# "total_penalized_difference_ratio": total_penalized_difference_ratio,
# "total_relative_difference_ratio": total_relative_difference_ratio,
# "total_valid_output_grid": total_valid_output_grid,
# "total_load_support_connected": total_load_support_connected,
# "total_load_support_connected_force_directional": total_load_support_connected_force_directional,
# "total_isolated_clusters_count": total_isolated_clusters_count,
# "total_force_path_cost_average_efficiency_ratio": total_force_path_cost_average_efficiency_ratio,
# "total_difficulty_score": total_difficulty_score,

# Metric keys and plot configs
metrics = [
    ("total_exact_match", "Exact Match", (0, 100)),
    ("total_difference_ratio", "Difference Ratio (%)", (0, 100)),
    ("total_penalized_difference_ratio", "Penalized Difference Ratio (%)", (0, 100)),
    ("total_relative_difference_ratio", "Relative Difference Ratio (%)", (0, 100)),
    ("total_valid_output_grid", "Valid Output Grid (%)", (0, 100)),
    ("total_load_support_connected", "Load-Support Connectivity (%)", (0, 100)),
    (
        "total_load_support_connected_force_directional",
        "Load-Support Directional Connectivity (%)",
        (0, 100),
    ),
    ("total_isolated_clusters_count", "Average Isolated Clusters Count", None),
    (
        "total_force_path_cost_average_efficiency_ratio",
        "Force Path Cost Average Efficiency Ratio (%)",
        (0, 100),
    ),
    ("total_difficulty_score", "Average Difficulty Score", None),
    (
        "total_difficulty_weighted_difference_ratio",
        "Difficulty Weighted Difference Ratio (%)",
        (0, 100),
    ),
    (
        "total_difficulty_weighted_relative_difference_ratio",
        "Difficulty Weighted Relative Difference Ratio (%)",
        (0, 100),
    ),
]

# Base path
base_dir = os.path.dirname(os.path.abspath(__file__))


def plot_results(
    model_order,
    model_display_names,
    output_dir,
    task_labels,
    tasks_by_difficulty,
    color_map_key="tab10",
    width=0.19,
):
    # Set color palette
    colors = cm.get_cmap(color_map_key)

    # Output folder
    output_dir = os.path.join(base_dir, f"{output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through difficulties and metrics
    for difficulty, task_keys in tasks_by_difficulty.items():
        for metric_key, metric_name, y_range in metrics:

            # Initialize data grid
            metric_data = [[] for _ in model_order]

            # Read data
            for task in task_keys:
                for i, model_folder in enumerate(model_order):
                    file_path = os.path.join(
                        base_dir, model_folder, f"{task}_aggregated_results.json"
                    )
                    if not os.path.exists(file_path):
                        print(f"⚠️ Missing file: {file_path}")
                        value = 0
                    else:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            value = data.get(metric_key, 0)
                            if (
                                metric_key == "total_difficulty_score"
                                or metric_key == "total_isolated_clusters_count"
                            ):
                                value = value / 100.0
                            if (
                                metric_key
                                == "total_difficulty_weighted_difference_ratio"
                                or metric_key
                                == "total_difficulty_weighted_relative_difference_ratio"
                            ):
                                value = value / 3
                    metric_data[i].append(value)

            # Plotting
            x = np.arange(len(task_labels))
            fig, ax = plt.subplots(figsize=(12, 6))

            for i, model_display in enumerate(model_display_names):
                ax.bar(
                    x + i * width,
                    metric_data[i],
                    width,
                    label=model_display,
                    color=get_model_color(model_order[i]),
                )

            ax.set_xlabel("Task")
            ax.set_ylabel(metric_name)
            ax.set_title(
                f"{metric_name} Across Tasks ({difficulty.capitalize()} Difficulty)"
            )
            ax.set_xticks(x + width * (len(model_order) - 1) / 2)
            ax.set_xticklabels(task_labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if y_range:
                ax.set_ylim(y_range)

            plt.tight_layout()

            # Save to PDF
            filename = f"{metric_key.replace('total_', '')}_{difficulty}.pdf"
            pdf_save_path = os.path.join(output_dir, filename)
            plt.savefig(pdf_save_path, dpi=300)
            print(f"✅ Saved: {pdf_save_path}")
            plt.close()


def plot_line_graph_results(
    model_order,
    model_display_names,
    output_dir,
    task_labels,
    tasks_by_difficulty,
    color_map_key="tab10",
    width=0.19,
):
    # Set color palette
    colors = cm.get_cmap(color_map_key)

    # Output folder
    output_dir = os.path.join(base_dir, f"{output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through difficulties and metrics
    for difficulty, task_keys in tasks_by_difficulty.items():
        for metric_key, metric_name, y_range in metrics:

            # Initialize data grid
            metric_data = [[] for _ in model_order]

            # Read data
            for task in task_keys:
                for i, model_folder in enumerate(model_order):
                    file_path = os.path.join(
                        base_dir, model_folder, f"{task}_aggregated_results.json"
                    )
                    if not os.path.exists(file_path):
                        print(f"⚠️ Missing file: {file_path}")
                        value = 0
                    else:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            value = data.get(metric_key, 0)
                            if (
                                metric_key == "total_difficulty_score"
                                or metric_key == "total_isolated_clusters_count"
                            ):
                                value = value / 100.0
                            if (
                                metric_key
                                == "total_difficulty_weighted_difference_ratio"
                                or metric_key
                                == "total_difficulty_weighted_relative_difference_ratio"
                            ):
                                value = value / 3
                    metric_data[i].append(value)

            # Plotting
            x = np.arange(len(task_labels))
            fig, ax = plt.subplots(figsize=(12, 6))

            for i, model_display in enumerate(model_display_names):
                ax.plot(
                    x,
                    metric_data[i],
                    label=model_display,
                    color=get_model_color(model_order[i]),
                    marker="o",  # add markers for clarity
                    linewidth=2,
                )

            ax.set_xlabel("Task")
            ax.set_ylabel(metric_name)
            ax.set_title(
                f"{metric_name} Across Tasks ({difficulty.capitalize()} Difficulty)"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if y_range:
                ax.set_ylim(y_range)

            plt.tight_layout()

            # Save to PDF
            filename = f"{metric_key.replace('total_', '')}_{difficulty}.pdf"
            pdf_save_path = os.path.join(output_dir, filename)
            plt.savefig(pdf_save_path, dpi=300)
            print(f"✅ Saved: {pdf_save_path}")
            plt.close()


def plot_main_body_results():
    # Task keys by difficulty
    tasks_by_difficulty = ALL_TASKS_BY_DIFFICULTY

    # Labels for x-axis
    task_labels = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Col",
        "3 Random Cols",
        "Full",
    ]

    # Folder names for each model
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

    # Clean display names for legends
    model_display_names = [
        "GPT-3.5 Turbo",
        "GPT-4.1",
        "GPT-4o",
        "Claude 3.7 Sonnet",
        "Claude Opus 4",
        "Gemini 1.5 Pro",
        "Gemini 2.5 Pro",
        "DeepSeek-R1",
        "Perplexity Sonar",
        "Perplexity Sonar Reasoning",
    ]

    plot_results(
        model_order,
        model_display_names,
        "plots/main_body_results",
        task_labels,
        tasks_by_difficulty,
        width=0.1,
    )


def plot_rotation_comparison_results():
    # Task keys by difficulty
    tasks_by_difficulty = {
        "easy": [
            "10_random_cell_easy",
            "3_random_row_easy",
            "3_random_column_easy",
            "full_easy",
        ],
    }

    # Labels for x-axis
    task_labels = [
        "10 Random Cells",
        "3 Random Rows",
        "3 Random Cols",
        "Full",
    ]

    # Folder names for each model
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

    # Clean display names for legends
    model_display_names = [
        "GPT-4.1",
        "GPT-4.1 (3 Rotations)",
        "Claude Opus 4",
        "Claude Opus 4 (3 Rotations)",
        "Gemini 2.5 Pro",
        "Gemini 2.5 Pro (3 Rotations)",
        "DeepSeek-R1",
        "DeepSeek-R1 (3 Rotations)",
        "Perplexity Sonar",
        "Perplexity Sonar (3 Rotations)",
    ]

    plot_results(
        model_order,
        model_display_names,
        "plots/rotation_comparison_results",
        task_labels,
        tasks_by_difficulty,
        color_map_key="tab20",
        width=0.09,
    )


def plot_rotation_best_model_results():
    # Task keys by difficulty
    tasks_by_difficulty = ALL_TASKS_BY_DIFFICULTY

    # Labels for x-axis
    task_labels = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Col",
        "3 Random Cols",
        "Full",
    ]

    # Folder names for each model
    model_order = [
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_3_rotations",
    ]

    # Clean display names for legends
    model_display_names = [
        "Claude Opus 4",
        "Claude Opus 4 (3 Rotations)",
    ]

    plot_results(
        model_order,
        model_display_names,
        "plots/rotation_best_model_results",
        task_labels,
        tasks_by_difficulty,
        width=0.4,
    )


def plot_rotation_comparison_delta_results():
    # Task keys by difficulty
    tasks_by_difficulty = {
        "easy": [
            "10_random_cell_easy",
            "3_random_row_easy",
            "3_random_column_easy",
            "full_easy",
        ],
    }

    # Labels for x-axis
    task_labels = [
        "10 Random Cells",
        "3 Random Rows",
        "3 Random Cols",
        "Full",
    ]

    # Folder names for each model
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

    # Clean display names for legends
    model_display_names = [
        "GPT-4.1",
        "GPT-4.1 (3 Rotations)",
        "Claude Opus 4",
        "Claude Opus 4 (3 Rotations)",
        "Gemini 2.5 Pro",
        "Gemini 2.5 Pro (3 Rotations)",
        "DeepSeek-R1",
        "DeepSeek-R1 (3 Rotations)",
        "Perplexity Sonar",
        "Perplexity Sonar (3 Rotations)",
    ]

    plot_line_graph_results(
        model_order,
        model_display_names,
        "plots/rotation_comparison_delta_results",
        task_labels,
        tasks_by_difficulty,
        color_map_key="tab20",
        width=0.09,
    )

    plot_results_rotation_avg(
        model_order,
        model_display_names,
        "plots/rotation_comparison_delta_results_avg",
        task_labels,
        tasks_by_difficulty,
        color_map_key="tab20",
    )


def plot_results_rotation_avg(
    model_order,
    model_display_names,
    output_dir,
    task_labels,
    tasks_by_difficulty,
    color_map_key="tab10",
):
    # Set color palette
    colors = cm.get_cmap(color_map_key)

    # Output folder
    output_dir = os.path.join(base_dir, f"{output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Split into rotation and non-rotation models
    non_rotation_models = [m for m in model_order if "_3_rotations" not in m]
    rotation_models = [m for m in model_order if "_3_rotations" in m]

    # Loop through difficulties and metrics
    for difficulty, task_keys in tasks_by_difficulty.items():
        for metric_key, metric_name, y_range in metrics:

            # Initialize data for averages
            non_rotation_data = []
            rotation_data = []

            # Read data
            for task in task_keys:
                # Non-rotation average
                non_rot_values = []
                for model_folder in non_rotation_models:
                    file_path = os.path.join(
                        base_dir, model_folder, f"{task}_aggregated_results.json"
                    )
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            value = data.get(metric_key, 0)
                            if (
                                metric_key == "total_difficulty_score"
                                or metric_key == "total_isolated_clusters_count"
                            ):
                                value = value / 100.0
                            if (
                                metric_key
                                == "total_difficulty_weighted_difference_ratio"
                                or metric_key
                                == "total_difficulty_weighted_relative_difference_ratio"
                            ):
                                value = value / 3
                            non_rot_values.append(value)
                non_rotation_data.append(
                    np.mean(non_rot_values) if non_rot_values else 0
                )

                # Rotation average
                rot_values = []
                for model_folder in rotation_models:
                    file_path = os.path.join(
                        base_dir, model_folder, f"{task}_aggregated_results.json"
                    )
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            value = data.get(metric_key, 0)
                            if (
                                metric_key == "total_difficulty_score"
                                or metric_key == "total_isolated_clusters_count"
                            ):
                                value = value / 100.0
                            if (
                                metric_key
                                == "total_difficulty_weighted_difference_ratio"
                                or metric_key
                                == "total_difficulty_weighted_relative_difference_ratio"
                            ):
                                value = value / 3
                            rot_values.append(value)
                rotation_data.append(np.mean(rot_values) if rot_values else 0)

            # Plotting
            x = np.arange(len(task_labels))
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                x,
                non_rotation_data,
                label="Average (No Rotation)",
                color=colors(0),
                marker="o",
                linewidth=2,
            )
            ax.plot(
                x,
                rotation_data,
                label="Average (Rotation)",
                color=colors(1),
                marker="o",
                linewidth=2,
            )

            ax.set_xlabel("Task")
            ax.set_ylabel(metric_name)
            ax.set_title(
                f"{metric_name} - Average Across Models ({difficulty.capitalize()} Difficulty)"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(task_labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if y_range:
                ax.set_ylim(y_range)

            plt.tight_layout()

            # Save to PDF
            filename = (
                f"{metric_key.replace('total_', '')}_{difficulty}_rotation_avg.pdf"
            )
            pdf_save_path = os.path.join(output_dir, filename)
            plt.savefig(pdf_save_path, dpi=300)
            print(f"✅ Saved: {pdf_save_path}")
            plt.close()


def plot_few_shot_results():
    # Task keys by difficulty
    tasks_by_difficulty = ALL_TASKS_BY_DIFFICULTY

    # Labels for x-axis
    task_labels = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Col",
        "3 Random Cols",
        "Full",
    ]

    # Folder names for each model
    model_order = [
        "claude-opus-4-20250514",
        "claude-opus-4-20250514_few_shot_1",
        "claude-opus-4-20250514_few_shot_3",
    ]

    # Clean display names for legends
    model_display_names = [
        "Claude Opus 4",
        "Claude Opus 4 (Few Shot 1)",
        "Claude Opus 4 (Few Shot 3)",
    ]

    plot_results(
        model_order,
        model_display_names,
        "plots/few_shot_results",
        task_labels,
        tasks_by_difficulty,
        width=0.3,
    )


def plot_physics_enhanced_neutral_prompt_comparison_results():
    # Task keys by difficulty
    tasks_by_difficulty = ALL_TASKS_BY_DIFFICULTY

    # Labels for x-axis
    task_labels = [
        "1 Random Cell",
        "5 Random Cells",
        "10 Random Cells",
        "1 Random Row",
        "3 Random Rows",
        "1 Random Col",
        "3 Random Cols",
        "Full",
    ]

    # Folder names for each model
    model_order = [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro-preview-05-06_physics_enhanced_prompt",
        "gemini-2.5-pro-preview-05-06_physics_neutral_prompt",
    ]

    # Clean display names for legends
    model_display_names = [
        "Gemini 2.5 Pro",
        "Gemini 2.5 Pro (Physics Enhanced Prompt)",
        "Gemini 2.5 Pro (Physics Neutral Prompt)",
    ]

    plot_results(
        model_order,
        model_display_names,
        "plots/physics_enhanced_neutral_prompt_comparison_results",
        task_labels,
        tasks_by_difficulty,
        color_map_key="tab20",
        width=0.3,
    )


if __name__ == "__main__":
    plot_main_body_results()
    plot_rotation_comparison_results()
    plot_rotation_best_model_results()
    plot_rotation_comparison_delta_results()
    plot_few_shot_results()
    plot_physics_enhanced_neutral_prompt_comparison_results()

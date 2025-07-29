import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Metric keys and plot configs
metrics = [
    ("total_exact_match", "Exact Match", (0, 100)),
    ("total_score", "Score", (0, 100)),
    ("total_normalized_score", "Normalized Score", None),
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
                    metric_data[i].append(value)

            # Plotting
            x = np.arange(len(task_labels))
            fig, ax = plt.subplots(figsize=(12, 8))

            for i, model_display in enumerate(model_display_names):
                ax.bar(
                    x + i * width,
                    metric_data[i],
                    width,
                    label=model_display,
                    color=colors(i),
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

            # Save to PNG
            filename = f"{metric_key.replace('total_', '')}_{difficulty}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300)
            print(f"✅ Saved: {save_path}")

            plt.close()


def plot_main_body_results():
    # Task keys by difficulty
    tasks_by_difficulty = {
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
        "gpt-4.1-2025-04-14",
        "claude-opus-4-20250514",
        "gemini-2.5-pro-preview-05-06",
        "deepseek-reasoner",
        "perplexity-sonar",
    ]

    # Clean display names for legends
    model_display_names = [
        "GPT-4.1",
        "Claude Opus 4",
        "Gemini 2.5 Pro",
        "DeepSeek-R1",
        "Perplexity Sonar",
    ]

    plot_results(
        model_order,
        model_display_names,
        "plots/main_body_results",
        task_labels,
        tasks_by_difficulty,
    )


def plot_appendix_results():
    # Task keys by difficulty
    tasks_by_difficulty = {
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
        "plots/appendix_results",
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
    tasks_by_difficulty = {
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


if __name__ == "__main__":
    plot_main_body_results()
    plot_appendix_results()
    plot_rotation_comparison_results()
    plot_rotation_best_model_results()

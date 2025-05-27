import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Folder names for each model
model_order = [
    "gpt-4.1",
    "claude-3-7-sonnet-20250219",
    "gemini-2.5-pro-preview-05-06",
    "deepseek-reasoner",
]

# Clean display names for legends
model_display_names = ["GPT 4.1", "Claude 3.7", "Gemini 2.5", "DeepSeek-R1"]

# Metric keys and plot configs
metrics = [
    ("total_exact_match", "Exact Match", (0, 100)),
    ("total_score", "Score", (0, 100)),
    ("total_normalized_score", "Normalized Score", None),
]

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

# Set color palette
colors = cm.get_cmap("tab10", len(model_order))

# Base path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Output folder
output_dir = os.path.join(base_dir, "plots")
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
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))

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

        if y_range:
            ax.set_ylim(y_range)

        plt.tight_layout()

        # Save to PNG
        filename = f"{metric_key.replace('total_', '')}_{difficulty}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved: {save_path}")

        plt.close()

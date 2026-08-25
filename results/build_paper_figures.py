"""Compose the paper's multi-panel appendix figures from the generated plots.

``plot_results.py`` writes one PDF per metric and difficulty.  The paper shows
those plots tiled onto pages, four metrics to a page, easy in the left column
and hard in the right.  Those pages used to be assembled by hand, which is both
unreproducible and error-prone: the figures shipped with the submission had a
duplicated panel in place of a missing one in three separate places.

This script rebuilds them from the plot PDFs, so a regenerated result set flows
all the way through to the paper with one command:

    python results/plot_results.py
    python results/build_paper_figures.py --out ../SPhyR-paper/source

Panels stay vector: pages are merged with a transform, never rasterised.
"""

import argparse
import os

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Page geometry, matching the figures the paper already uses.
PAGE_WIDTH = 595.276
PAGE_HEIGHT = 744.094
MARGIN = 12.0
GUTTER = 6.0

# Metric groups, one per output page.  Names are the plot file stems written by
# plot_results.py (the aggregate key without its "total_" prefix).
RECONSTRUCTION = [
    "exact_match",
    "difference_ratio",
    "relative_difference_ratio",
    "penalized_difference_ratio",
]
DIFFICULTY = [
    "difficulty_score",
    "difficulty_weighted_difference_ratio",
    "difficulty_weighted_relative_difference_ratio",
    "valid_output_grid",
]
TOPOLOGY = [
    "load_support_connected",
    "load_support_connected_force_directional",
    "isolated_clusters_count",
    "force_path_cost_average_efficiency_ratio",
]
STRUCTURAL = [
    "topology_score",
    "structural_efficiency",
    "material_efficiency",
    "compliance_efficiency_vs_ground_truth",
    "load_carrying",
]

BOTH = ("easy", "hard")
EASY_ONLY = ("easy",)

# source figure stem -> (plot directory, metric group, difficulties)
FIGURES = {
    "main_eval_results_01": ("main_body_results", RECONSTRUCTION, BOTH),
    "main_eval_results_02": ("main_body_results", DIFFICULTY, BOTH),
    "main_eval_results_03": ("main_body_results", TOPOLOGY, BOTH),
    "main_eval_results_04": ("main_body_results", STRUCTURAL, BOTH),
    "grid_rotation_all_models_selected_tasks_eval_results_01": (
        "rotation_comparison_results", RECONSTRUCTION + DIFFICULTY, EASY_ONLY,
    ),
    "grid_rotation_all_models_selected_tasks_eval_results_02": (
        "rotation_comparison_results", TOPOLOGY + STRUCTURAL, EASY_ONLY,
    ),
    "grid_rotation_claude_4_eval_results_01": (
        "rotation_best_model_results", RECONSTRUCTION, BOTH,
    ),
    "grid_rotation_claude_4_eval_results_02": (
        "rotation_best_model_results", DIFFICULTY, BOTH,
    ),
    "grid_rotation_claude_4_eval_results_03": (
        "rotation_best_model_results", TOPOLOGY, BOTH,
    ),
    "grid_rotation_claude_4_eval_results_04": (
        "rotation_best_model_results", STRUCTURAL, BOTH,
    ),
    "few_shot_results_01": ("few_shot_results", RECONSTRUCTION, BOTH),
    "few_shot_results_02": ("few_shot_results", DIFFICULTY, BOTH),
    "few_shot_results_03": ("few_shot_results", TOPOLOGY, BOTH),
    "few_shot_results_04": ("few_shot_results", STRUCTURAL, BOTH),
    "physics_enhanced_neutral_prompt_comparison_results_01": (
        "physics_enhanced_neutral_prompt_comparison_results", RECONSTRUCTION, BOTH,
    ),
    "physics_enhanced_neutral_prompt_comparison_results_02": (
        "physics_enhanced_neutral_prompt_comparison_results", DIFFICULTY, BOTH,
    ),
    "physics_enhanced_neutral_prompt_comparison_results_03": (
        "physics_enhanced_neutral_prompt_comparison_results", TOPOLOGY, BOTH,
    ),
    "physics_enhanced_neutral_prompt_comparison_results_04": (
        "physics_enhanced_neutral_prompt_comparison_results", STRUCTURAL, BOTH,
    ),
}


def get_panel_paths(plot_dir, metrics, difficulties):
    """Panels for one page, in reading order, skipping any that do not exist."""
    panels = []
    missing = []
    for metric in metrics:
        for difficulty in difficulties:
            path = os.path.join(PLOTS_DIR, plot_dir, f"{metric}_{difficulty}.pdf")
            if os.path.exists(path):
                panels.append(path)
            else:
                missing.append(os.path.relpath(path, BASE_DIR))
    return panels, missing


def compose_page(panel_paths, columns):
    """Tile panels onto one page, sized to the content.

    The page height follows from the panel aspect ratio rather than being fixed,
    so the sheet has no dead space in it; LaTeX scales the result to
    ``\\textwidth`` and only the aspect ratio matters downstream.
    """
    rows = max(1, -(-len(panel_paths) // columns))

    first = PdfReader(panel_paths[0]).pages[0]
    aspect = float(first.mediabox.width) / float(first.mediabox.height)

    cell_width = (PAGE_WIDTH - 2 * MARGIN - (columns - 1) * GUTTER) / columns
    cell_height = cell_width / aspect
    page_height = 2 * MARGIN + rows * cell_height + (rows - 1) * GUTTER

    page = PageObject.create_blank_page(width=PAGE_WIDTH, height=page_height)

    for index, path in enumerate(panel_paths):
        row, column = divmod(index, columns)
        source = PdfReader(path).pages[0]
        width = float(source.mediabox.width)
        height = float(source.mediabox.height)

        scale = min(cell_width / width, cell_height / height)
        offset_x = MARGIN + column * (cell_width + GUTTER)
        # Rows run top to bottom; PDF coordinates run bottom to top.
        offset_y = page_height - MARGIN - (row + 1) * cell_height - row * GUTTER
        offset_x += (cell_width - width * scale) / 2
        offset_y += (cell_height - height * scale) / 2

        page.merge_transformed_page(
            source, Transformation().scale(scale).translate(offset_x, offset_y)
        )

    return page


def build(output_dir, only=None):
    os.makedirs(output_dir, exist_ok=True)
    built, skipped = [], []

    for stem, (plot_dir, metrics, difficulties) in FIGURES.items():
        if only and only not in stem:
            continue

        panels, missing = get_panel_paths(plot_dir, metrics, difficulties)
        if not panels:
            skipped.append((stem, missing))
            continue

        # Always two columns: with both difficulties that puts easy beside hard
        # on each row, and with one difficulty the metrics simply flow across.
        writer = PdfWriter()
        writer.add_page(compose_page(panels, columns=2))

        path = os.path.join(output_dir, f"{stem}.pdf")
        with open(path, "wb") as handle:
            writer.write(handle)

        built.append((stem, len(panels), missing))

    for stem, count, missing in built:
        note = f"  ({len(missing)} panel(s) not generated yet)" if missing else ""
        print(f"  {stem}.pdf: {count} panels{note}")
    for stem, missing in skipped:
        print(f"  {stem}.pdf: SKIPPED, no panels found ({len(missing)} missing)")

    print(f"\n{len(built)} figures written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(BASE_DIR, "plots", "paper_figures"),
        help="directory to write the composed figures into",
    )
    parser.add_argument(
        "--only", help="only rebuild figures whose name contains this string"
    )
    args = parser.parse_args()
    build(os.path.abspath(args.out), only=args.only)


if __name__ == "__main__":
    main()

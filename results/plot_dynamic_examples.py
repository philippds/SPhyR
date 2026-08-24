"""Figure: what cell-by-cell comparison sees versus what the physics sees.

Renders two completions from the stored benchmark results side by side with the
dataset answer and with the structure a topology optimizer produces for the same
hole and the same material budget:

* a completion that string matching marks wrong and that is in fact stiffer than
  the answer it is compared against, on identical material;
* a completion that string matching marks 99% correct and whose single wrong
  cell severs the load path entirely.

Run from the repository root:

    python results/plot_dynamic_examples.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sphyr.metrics.structural import get_reference_optimum, get_structural_metrics
from sphyr.metrics.utils import extract_grid_from_text
from sphyr.physics.boundary_conditions import (
    get_densities_from_grid,
    get_free_cells_mask,
    get_problem_from_grid,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "dynamic_evaluation")

# Sequential blue ramp (steps 100-700) for density, plus the two reserved
# categorical hues for the boundary conditions.  Loads and supports also carry a
# letter, so identity is never colour alone.
DENSITY_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
LOAD_COLOR = "#eb6834"
SUPPORT_COLOR = "#1baf7a"
MASK_COLOR = "#e8e8e4"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
CELL_EDGE = "#d8d8d3"

DENSITY_CMAP = LinearSegmentedColormap.from_list("sphyr_density", DENSITY_RAMP)

# The two records the figure is built from, located by the analysis in the
# repository history; metrics and reference designs are recomputed here so the
# figure always agrees with the current evaluation code.
CASES = [
    {
        "path": "results/gemini-1.5-pro/full_easy_results.json",
        "index": 4,
        "model": "Gemini 1.5 Pro",
        "subject": "full_easy",
        "headline": "Different answer, stiffer structure",
        "note": (
            "14 of 100 cells differ from the reference answer, on identical material.\n"
            "The completion carries the load through a uniform section rather than a "
            "tapered one, which is stiffer for the same volume: it beats both the\n"
            "dataset answer and the optimizer's own design, so its score saturates."
        ),
    },
    {
        "path": "results/perplexity-sonar/1_random_cell_easy_results.json",
        "index": 0,
        "model": "Perplexity Sonar",
        "subject": "1_random_cell_easy",
        "headline": "Almost identical answer, no structure at all",
        "note": (
            "1 of 100 cells differs from the reference answer - the one cell that\n"
            "was masked - and it cuts the only path from the load to the supports."
        ),
    },
]


def load_case(case):
    """Recompute grids, metrics and the reference optimum for one record."""
    with open(case["path"], "r") as handle:
        record = json.load(handle)[case["index"]]

    input_grid = extract_grid_from_text(record["prompt"])
    output_grid = extract_grid_from_text(record["completion"])
    gt_grid = extract_grid_from_text(record["ground_truth"])

    metrics = get_structural_metrics(
        output_grid=output_grid, gt_grid=gt_grid, input_grid=input_grid
    )

    problem = get_problem_from_grid(gt_grid)
    gt_densities, _ = get_densities_from_grid(gt_grid)
    gt_densities[problem.structural_cells_mask()] = 1.0

    free_cells = get_free_cells_mask(input_grid, (problem.nely, problem.nelx))
    optimum = get_reference_optimum(
        problem=problem,
        free_cells=free_cells,
        pinned_densities=gt_densities,
        volume_fraction=metrics.ground_truth_volume_fraction,
        seed_designs=(gt_densities,),
    )

    optimum_grid = densities_to_grid(optimum.densities, gt_grid)

    return {
        **case,
        "input_grid": input_grid,
        "output_grid": output_grid,
        "gt_grid": gt_grid,
        "optimum_grid": optimum_grid,
        "metrics": metrics,
        "optimum": optimum,
        "differing_cells": count_differences(output_grid, gt_grid),
    }


def densities_to_grid(densities, reference_grid):
    """Turn an optimiser density field back into grid tokens."""
    grid = []
    for row in range(len(reference_grid)):
        tokens = []
        for col in range(len(reference_grid[0])):
            token = str(reference_grid[row][col]).strip()
            if token in ("L", "S"):
                tokens.append(token)
            else:
                tokens.append(f"{densities[row, col]:.2f}")
        grid.append(tokens)
    return grid


def count_differences(grid, other):
    return sum(
        1
        for row, other_row in zip(grid, other)
        for cell, other_cell in zip(row, other_row)
        if cell != other_cell
    )


def cell_density(token):
    if token in ("L", "S"):
        return 1.0
    try:
        return min(max(float(token), 0.0), 1.0)
    except ValueError:
        return 0.0


def draw_grid(ax, grid, title, subtitle, highlight=None):
    """Render one 10x10 structure panel."""
    rows, cols = len(grid), len(grid[0])

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(rows):
        for col in range(cols):
            token = str(grid[row][col]).strip()

            if token == "V":
                ax.add_patch(
                    Rectangle(
                        (col, row), 1, 1,
                        facecolor=MASK_COLOR, edgecolor=TEXT_MUTED,
                        hatch="////", linewidth=0.4,
                    )
                )
                continue

            if token == "L":
                face, label = LOAD_COLOR, "L"
            elif token == "S":
                face, label = SUPPORT_COLOR, "S"
            else:
                face, label = DENSITY_CMAP(cell_density(token)), None

            ax.add_patch(
                Rectangle(
                    (col, row), 1, 1,
                    facecolor=face, edgecolor=CELL_EDGE, linewidth=0.4,
                )
            )
            if label:
                ax.text(
                    col + 0.5, row + 0.5, label,
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=TEXT_PRIMARY,
                )

    if highlight is not None:
        for row in range(rows):
            for col in range(cols):
                if not highlight[row][col]:
                    continue
                # A surface ring under a dark outline, so the marker reads on
                # both an empty cell and a solid one.
                ax.add_patch(
                    Rectangle(
                        (col + 0.09, row + 0.09), 0.82, 0.82,
                        facecolor="none", edgecolor=SURFACE, linewidth=2.4,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (col + 0.09, row + 0.09), 0.82, 0.82,
                        facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=1.3,
                    )
                )

    ax.set_title(title, fontsize=9.5, color=TEXT_PRIMARY, pad=7, fontweight="semibold")
    if subtitle:
        ax.text(
            0.5, -0.06, subtitle,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=TEXT_SECONDARY, linespacing=1.45,
        )


def difference_mask(grid, other):
    return [
        [cell != other_cell for cell, other_cell in zip(row, other_row)]
        for row, other_row in zip(grid, other)
    ]


def format_stats(compliance, volume_fraction, load_carrying=True):
    """Compliance is only a meaningful number while a load path exists."""
    stiffness = (
        f"compliance {compliance:.2f}" if load_carrying else "load path severed"
    )
    return f"{stiffness}\nmaterial {volume_fraction * 100:.0f}%"


def build_figure(cases):
    fig, axes = plt.subplots(len(cases), 4, figsize=(12.4, 9.4))
    fig.patch.set_facecolor(SURFACE)
    if len(cases) == 1:
        axes = np.array([axes])

    for row_index, case in enumerate(cases):
        metrics = case["metrics"]
        row = axes[row_index]

        draw_grid(
            row[0],
            case["input_grid"],
            "Prompt",
            pluralize(count_masked(case["input_grid"]), "masked cell"),
        )
        draw_grid(
            row[1],
            case["output_grid"],
            f"{case['model']} completion",
            format_stats(
                metrics.compliance, metrics.volume_fraction, metrics.load_carrying
            ),
            highlight=difference_mask(case["output_grid"], case["gt_grid"]),
        )
        draw_grid(
            row[2],
            case["gt_grid"],
            "Dataset answer",
            format_stats(
                metrics.ground_truth_compliance, metrics.ground_truth_volume_fraction
            ),
        )
        draw_grid(
            row[3],
            case["optimum_grid"],
            "SIMP reference, same budget",
            format_stats(
                case["optimum"].compliance, metrics.ground_truth_volume_fraction
            ),
        )

        row[0].text(
            -0.13, 1.52,
            f"{chr(ord('A') + row_index)}.  {case['headline']}",
            transform=row[0].transAxes, ha="left", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT_PRIMARY,
        )
        row[0].text(
            -0.13, 1.20, case["note"],
            transform=row[0].transAxes, ha="left", va="bottom",
            fontsize=8.5, color=TEXT_SECONDARY, linespacing=1.5,
        )

    verdicts = []
    for case in cases:
        metrics = case["metrics"]
        if metrics.load_carrying:
            stiffness = (
                f"{metrics.ground_truth_compliance / metrics.compliance:.2f}x "
                "the stiffness of the answer"
            )
        else:
            stiffness = "carries no load"
        verdicts.append(
            f"{case['model']} on {case['subject']}:   "
            f"{100 - case['differing_cells']}% of cells match the answer   |   "
            f"{stiffness}   |   "
            f"topology score {metrics.topology_score:.2f}"
        )

    draw_legend(fig)

    fig.text(
        0.5, 0.035, "\n".join(verdicts),
        ha="center", va="bottom", fontsize=8.5,
        color=TEXT_SECONDARY, linespacing=1.7,
    )

    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.76, bottom=0.13, wspace=0.16, hspace=1.05
    )
    return fig


def draw_legend(fig):
    """Key for the cell encoding, including the difference markers."""
    entries = [
        ("Load", LOAD_COLOR, None),
        ("Support", SUPPORT_COLOR, None),
        ("Material", DENSITY_CMAP(1.0), None),
        ("Empty", DENSITY_CMAP(0.0), None),
        ("Masked", MASK_COLOR, "////"),
        ("Differs from answer", None, "ring"),
    ]

    axis = fig.add_axes([0.05, 0.905, 0.93, 0.05])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 1)
    axis.axis("off")

    cursor = 0.0
    for label, color, style in entries:
        if style == "ring":
            axis.add_patch(
                Rectangle(
                    (cursor, 0.28), 2.0, 0.46,
                    facecolor=SURFACE, edgecolor=TEXT_PRIMARY, linewidth=1.3,
                )
            )
        else:
            axis.add_patch(
                Rectangle(
                    (cursor, 0.28), 2.0, 0.46,
                    facecolor=color, edgecolor=CELL_EDGE,
                    hatch=style, linewidth=0.5,
                )
            )
        axis.text(
            cursor + 2.9, 0.5, label,
            ha="left", va="center", fontsize=8.5, color=TEXT_SECONDARY,
        )
        cursor += 3.6 + len(label) * 1.02


def count_masked(grid):
    return sum(1 for row in grid for cell in row if str(cell).strip() == "V")


def pluralize(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cases = []
    for case in CASES:
        if not os.path.exists(case["path"]):
            print(f"Missing results file, skipping: {case['path']}")
            continue
        cases.append(load_case(case))

    if not cases:
        raise SystemExit("No cases could be loaded.")

    figure = build_figure(cases)

    for extension in ("pdf", "png"):
        path = os.path.join(OUTPUT_DIR, f"dynamic_vs_string_matching.{extension}")
        figure.savefig(path, dpi=220, facecolor=SURFACE)
        print(f"Saved: {path}")

    for case in cases:
        metrics = case["metrics"]
        print(
            f"  {case['model']:18s} {case['subject']:20s} "
            f"differing={case['differing_cells']:3d}  "
            f"C={metrics.compliance:12.4f}  C_gt={metrics.ground_truth_compliance:10.4f}  "
            f"C_opt={case['optimum'].compliance:8.4f}  topo={metrics.topology_score:.3f}"
        )


if __name__ == "__main__":
    main()

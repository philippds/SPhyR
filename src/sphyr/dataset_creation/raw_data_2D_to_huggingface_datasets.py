import ast
import copy
import math
import os
import random
import json

raw_data = "src/sphyr/dataset_creation/topology_optimization_data/2D/raw_data"
output_path = "src/sphyr/dataset_creation/huggingface/2D/datasets"
csv_files = [f for f in os.listdir(raw_data) if f.endswith(".csv")]

dimensions = 10

random.seed(42)


def process_line(line):
    input_str = line.strip()
    parts, temp, inside_brackets = [], "", False
    for char in input_str:
        if char == "[":
            inside_brackets = True
        if char == "]":
            inside_brackets = False
        temp += char
        if char == "," and not inside_brackets:
            parts.append(temp.strip(","))
            temp = ""
    if temp:
        parts.append(temp)

    x_loc = int(math.floor(float(parts[1])))
    y_loc = int(math.floor(float(parts[2])))
    is_static = parts[5] == "True"
    value = float(parts[4])
    return x_loc, y_loc, is_static, value


def add_random_cells(grid, n, dimensions):
    grid = copy.deepcopy(grid)
    for _ in range(n):
        x, y = random.randint(0, dimensions - 1), random.randint(0, dimensions - 1)
        grid[y][x] = "V"
    return grid


def add_random_lines(grid, n, dimensions):
    grid = copy.deepcopy(grid)
    ys = random.sample(range(1, dimensions - 1), n)
    for y in ys:
        for x in range(dimensions):
            grid[y][x] = "V"
    return grid


def add_random_columns(grid, n, dimensions):
    grid = copy.deepcopy(grid)
    xs = random.sample(range(dimensions), n)
    for x in xs:
        for y in range(dimensions):
            grid[y][x] = "V"
    return grid


def add_specific_lines(grid, start, end, dimensions):
    grid = copy.deepcopy(grid)
    for y in range(start, end + 1):
        for x in range(dimensions):
            grid[y][x] = "V"
    return grid


variants = {
    "1_random_cell": lambda g, d: add_random_cells(g, 1, d),
    "5_random_cell": lambda g, d: add_random_cells(g, 5, d),
    "10_random_cell": lambda g, d: add_random_cells(g, 10, d),
    "1_random_row": lambda g, d: add_random_lines(g, 1, d),
    "3_random_row": lambda g, d: add_random_lines(g, 3, d),
    "1_random_column": lambda g, d: add_random_columns(g, 1, d),
    "3_random_column": lambda g, d: add_random_columns(g, 3, d),
    "full": lambda g, d: add_specific_lines(g, 1, 8, d),
}

all_datasets = {}

for index, csv_file in enumerate(csv_files):
    with open(os.path.join(raw_data, csv_file), "r") as f:
        lines = f.readlines()

    grid_easy = [[0 for _ in range(dimensions)] for _ in range(dimensions)]
    grid_hard = [[0.0 for _ in range(dimensions)] for _ in range(dimensions)]

    for line in lines:
        x, y, is_static, value = process_line(line)

        if not is_static:
            grid_easy[y][x] = "1" if float(f"{value:.1f}") > 0 else "0"
            grid_hard[y][x] = f"{value:.1f}"
        elif y == 0:
            grid_easy[y][x] = grid_hard[y][x] = "S"
        else:
            grid_easy[y][x] = grid_hard[y][x] = "L"

    # Flip vertically
    grid_easy = grid_easy[::-1]
    grid_hard = grid_hard[::-1]

    for name, modifier in variants.items():
        for mode, grid in [
            ("easy", grid_easy),
            ("hard", grid_hard),
        ]:
            modified_grid = modifier(grid, dimensions)

            key = f"{name}_{mode}"
            if key not in all_datasets:
                all_datasets[key] = []

            all_datasets[key].append(
                {
                    "index": index,
                    "input_grid": modified_grid,
                    "ground_truth": grid,
                }
            )

for name, dataset in all_datasets.items():
    os.makedirs(output_path, exist_ok=True)
    out_path = f"{output_path}/{name}.jsonl"
    with open(out_path, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {out_path} ({len(dataset)} records)")

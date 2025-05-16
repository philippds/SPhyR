import ast
import copy
import math
import os
import random
import json

raw_data = "dataset_creation/raw_topology_optimization_data/raw_data"
output_path = "dataset_creation/huggingface/datasets"


# find all csv files in the directory
csv_files = [f for f in os.listdir(raw_data) if f.endswith(".csv")]


def process_line(line):
    # Step 1: Remove the newline character
    input_str = line.strip()

    # Step 2: Use a smarter way to split while preserving the list structure
    # Split using a custom parser to avoid issues with commas inside brackets
    parts = []
    temp = ""
    inside_brackets = False

    for char in input_str:
        if char == "[":
            inside_brackets = True
        elif char == "]":
            inside_brackets = False
        temp += char
        if char == "," and not inside_brackets:
            parts.append(temp.strip(","))
            temp = ""
    if temp:  # Add the last piece
        parts.append(temp)

    # Step 3: Process each value to cast it to the appropriate type
    processed_values = []
    for value in parts:
        value = value.strip()  # Remove any stray whitespace
        # Handle boolean
        if value == "True":
            processed_values.append(True)
        elif value == "False":
            processed_values.append(False)
        # Handle list (detect if it starts with '[' and ends with ']')
        elif value.startswith("[") and value.endswith("]"):
            processed_values.append(ast.literal_eval(value))  # Safely parse list
        # Handle numeric types
        else:
            try:
                if "." in value:
                    processed_values.append(float(value))  # Convert to float
                else:
                    processed_values.append(int(value))  # Convert to int
            except ValueError:
                processed_values.append(value)  # Keep as string if unable to cast

    return processed_values


# read all csv files

dimensions = 10

random_cell_1_easy_dataset = []
random_cell_1_hard_dataset = []
random_cell_5_easy_dataset = []
random_cell_5_hard_dataset = []
random_cell_10_easy_dataset = []
random_cell_10_hard_dataset = []
random_line_1_easy_dataset = []
random_line_1_hard_dataset = []
random_line_3_easy_dataset = []
random_line_3_hard_dataset = []
random_column_1_easy_dataset = []
random_column_1_hard_dataset = []
random_column_3_easy_dataset = []
random_column_3_hard_dataset = []
random_line_1_to_8_easy_dataset = []
random_line_1_to_8_hard_dataset = []

for index, csv_file in enumerate(csv_files):
    with open(os.path.join(raw_data, csv_file), "r") as f:
        lines = f.readlines()

        grid_str_list = []

        grid_hard_2d = [[0 for _ in range(dimensions)] for _ in range(dimensions)]
        grid_easy_2d = [[0 for _ in range(dimensions)] for _ in range(dimensions)]

        for line in lines:
            processed_line = process_line(line)

            # data = {
            #     "index": index,
            #     "node_index": processed_line[0],
            #     "x": processed_line[1],
            #     "y": processed_line[2],
            #     "z": processed_line[3],
            #     "density": processed_line[4],
            #     "static": processed_line[5],
            #     "neighbour_node_index": processed_line[6],
            # }

            x_index = int(math.floor(processed_line[1]))
            y_index = int(math.floor(processed_line[2]))

            # check if static
            if processed_line[5] == False:
                grid_easy_2d[y_index][x_index] = (
                    1 if float(f"{processed_line[4]:.1f}") > 0 else 0
                )
                grid_hard_2d[y_index][x_index] = float(f"{processed_line[4]:.1f}")
            elif y_index == 0:
                grid_hard_2d[y_index][x_index] = grid_easy_2d[y_index][x_index] = "S"
            else:
                grid_hard_2d[y_index][x_index] = grid_easy_2d[y_index][x_index] = "L"

        # Reverse the outer list (z-index)
        grid_easy_2d = grid_easy_2d[::-1]
        grid_hard_2d = grid_hard_2d[::-1]

        # 1 random cell for easy and hard
        random_cell_1_easy = copy.deepcopy(grid_easy_2d)
        random_cell_1_hard = copy.deepcopy(grid_hard_2d)
        random_index_x = random.randint(0, dimensions - 1)
        random_index_y = random.randint(0, dimensions - 1)
        random_cell_1_easy[random_index_y][random_index_x] = "V"
        random_cell_1_hard[random_index_y][random_index_x] = "V"

        # 5 random cells for easy and hard
        random_cell_5_easy = copy.deepcopy(grid_easy_2d)
        random_cell_5_hard = copy.deepcopy(grid_hard_2d)
        for _ in range(5):
            random_index_x = random.randint(0, dimensions - 1)
            random_index_y = random.randint(0, dimensions - 1)
            random_cell_5_easy[random_index_y][random_index_x] = "V"
            random_cell_5_hard[random_index_y][random_index_x] = "V"

        # 10 random cells for easy and hard
        random_cell_10_easy = copy.deepcopy(grid_easy_2d)
        random_cell_10_hard = copy.deepcopy(grid_hard_2d)
        for _ in range(10):
            random_index_x = random.randint(0, dimensions - 1)
            random_index_y = random.randint(0, dimensions - 1)
            random_cell_10_easy[random_index_y][random_index_x] = "V"
            random_cell_10_hard[random_index_y][random_index_x] = "V"

        # 1 random line for easy and hard
        random_line_1_easy = copy.deepcopy(grid_easy_2d)
        random_line_1_hard = copy.deepcopy(grid_hard_2d)
        random_index_y = random.randint(1, dimensions - 2)
        for i in range(dimensions):
            random_line_1_easy[random_index_y][i] = "V"
            random_line_1_hard[random_index_y][i] = "V"

        # 3 random lines for easy and hard
        random_line_3_easy = copy.deepcopy(grid_easy_2d)
        random_line_3_hard = copy.deepcopy(grid_hard_2d)
        for _ in range(3):
            random_index_y = random.randint(1, dimensions - 2)
            for i in range(dimensions):
                random_line_3_easy[random_index_y][i] = "V"
                random_line_3_hard[random_index_y][i] = "V"

        # 1 random column for easy and hard
        random_column_1_easy = copy.deepcopy(grid_easy_2d)
        random_column_1_hard = copy.deepcopy(grid_hard_2d)
        random_index_x = random.randint(0, dimensions - 1)
        for i in range(dimensions):
            random_column_1_easy[i][random_index_x] = "V"
            random_column_1_hard[i][random_index_x] = "V"

        # 3 random columns for easy and hard
        random_column_3_easy = copy.deepcopy(grid_easy_2d)
        random_column_3_hard = copy.deepcopy(grid_hard_2d)
        for _ in range(3):
            random_index_x = random.randint(0, dimensions - 1)
            for i in range(dimensions):
                random_column_3_easy[i][random_index_x] = "V"
                random_column_3_hard[i][random_index_x] = "V"

        # line index 1 to 8 with V
        random_line_1_to_8_easy = copy.deepcopy(grid_easy_2d)
        random_line_1_to_8_hard = copy.deepcopy(grid_hard_2d)
        for i in range(1, 9):
            for j in range(dimensions):
                random_line_1_to_8_easy[i][j] = "V"
                random_line_1_to_8_hard[i][j] = "V"

        # make string for layer
        ground_truth_easy_grid_2d_str = []
        ground_truth_hard_grid_2d_str = []

        random_cell_1_easy_grid_2d_str = []
        random_cell_1_hard_grid_2d_str = []
        random_cell_5_easy_grid_2d_str = []
        random_cell_5_hard_grid_2d_str = []
        random_cell_10_easy_grid_2d_str = []
        random_cell_10_hard_grid_2d_str = []

        random_line_1_easy_grid_2d_str = []
        random_line_1_hard_grid_2d_str = []
        random_line_3_easy_grid_2d_str = []
        random_line_3_hard_grid_2d_str = []

        random_column_1_easy_grid_2d_str = []
        random_column_1_hard_grid_2d_str = []
        random_column_3_easy_grid_2d_str = []
        random_column_3_hard_grid_2d_str = []

        random_line_1_to_8_easy_grid_2d_str = []
        random_line_1_to_8_hard_grid_2d_str = []

        for layer_index in range(len(grid_easy_2d)):
            ground_truth_easy_grid_2d_str.append(
                " ".join([str(i) for i in grid_easy_2d[layer_index]])
            )
            ground_truth_hard_grid_2d_str.append(
                " ".join([str(i) for i in grid_hard_2d[layer_index]])
            )
            random_cell_1_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_cell_1_easy[layer_index]])
            )
            random_cell_1_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_cell_1_hard[layer_index]])
            )
            random_cell_5_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_cell_5_easy[layer_index]])
            )
            random_cell_5_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_cell_5_hard[layer_index]])
            )
            random_cell_10_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_cell_10_easy[layer_index]])
            )
            random_cell_10_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_cell_10_hard[layer_index]])
            )
            random_line_1_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_line_1_easy[layer_index]])
            )
            random_line_1_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_line_1_hard[layer_index]])
            )
            random_line_3_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_line_3_easy[layer_index]])
            )
            random_line_3_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_line_3_hard[layer_index]])
            )
            random_column_1_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_column_1_easy[layer_index]])
            )
            random_column_1_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_column_1_hard[layer_index]])
            )
            random_column_3_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_column_3_easy[layer_index]])
            )
            random_column_3_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_column_3_hard[layer_index]])
            )
            random_line_1_to_8_easy_grid_2d_str.append(
                " ".join([str(i) for i in random_line_1_to_8_easy[layer_index]])
            )
            random_line_1_to_8_hard_grid_2d_str.append(
                " ".join([str(i) for i in random_line_1_to_8_hard[layer_index]])
            )

        ground_truth_easy_grid_2d_str = "\n".join(ground_truth_easy_grid_2d_str)
        ground_truth_hard_grid_2d_str = "\n".join(ground_truth_hard_grid_2d_str)

        random_cell_1_easy_grid_2d_str = "\n".join(random_cell_1_easy_grid_2d_str)
        random_cell_1_hard_grid_2d_str = "\n".join(random_cell_1_hard_grid_2d_str)
        random_cell_5_easy_grid_2d_str = "\n".join(random_cell_5_easy_grid_2d_str)
        random_cell_5_hard_grid_2d_str = "\n".join(random_cell_5_hard_grid_2d_str)
        random_cell_10_easy_grid_2d_str = "\n".join(random_cell_10_easy_grid_2d_str)
        random_cell_10_hard_grid_2d_str = "\n".join(random_cell_10_hard_grid_2d_str)

        random_line_1_easy_grid_2d_str = "\n".join(random_line_1_easy_grid_2d_str)
        random_line_1_hard_grid_2d_str = "\n".join(random_line_1_hard_grid_2d_str)
        random_line_3_easy_grid_2d_str = "\n".join(random_line_3_easy_grid_2d_str)
        random_line_3_hard_grid_2d_str = "\n".join(random_line_3_hard_grid_2d_str)

        random_column_1_easy_grid_2d_str = "\n".join(random_column_1_easy_grid_2d_str)
        random_column_1_hard_grid_2d_str = "\n".join(random_column_1_hard_grid_2d_str)
        random_column_3_easy_grid_2d_str = "\n".join(random_column_3_easy_grid_2d_str)
        random_column_3_hard_grid_2d_str = "\n".join(random_column_3_hard_grid_2d_str)

        random_line_1_to_8_easy_grid_2d_str = "\n".join(
            random_line_1_to_8_easy_grid_2d_str
        )
        random_line_1_to_8_hard_grid_2d_str = "\n".join(
            random_line_1_to_8_hard_grid_2d_str
        )

        random_cell_1_easy_dataset.append(
            {
                "index": index,
                "1_random_cell_easy": random_cell_1_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_cell_1_hard_dataset.append(
            {
                "index": index,
                "1_random_cell_hard": random_cell_1_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_cell_5_easy_dataset.append(
            {
                "index": index,
                "5_random_cell_easy": random_cell_5_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_cell_5_hard_dataset.append(
            {
                "index": index,
                "5_random_cell_hard": random_cell_5_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_cell_10_easy_dataset.append(
            {
                "index": index,
                "10_random_cell_easy": random_cell_10_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_cell_10_hard_dataset.append(
            {
                "index": index,
                "10_random_cell_hard": random_cell_10_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_line_1_easy_dataset.append(
            {
                "index": index,
                "1_random_row_easy": random_line_1_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_line_1_hard_dataset.append(
            {
                "index": index,
                "1_random_row_hard": random_line_1_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_line_3_easy_dataset.append(
            {
                "index": index,
                "3_random_row_easy": random_line_3_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_line_3_hard_dataset.append(
            {
                "index": index,
                "3_random_row_hard": random_line_3_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_column_1_easy_dataset.append(
            {
                "index": index,
                "1_random_column_easy": random_column_1_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_column_1_hard_dataset.append(
            {
                "index": index,
                "1_random_column_hard": random_column_1_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_column_3_easy_dataset.append(
            {
                "index": index,
                "3_random_column_easy": random_column_3_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_column_3_hard_dataset.append(
            {
                "index": index,
                "3_random_column_hard": random_column_3_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )
        random_line_1_to_8_easy_dataset.append(
            {
                "index": index,
                "full_easy": random_line_1_to_8_easy_grid_2d_str,
                "ground_truth_easy": ground_truth_easy_grid_2d_str,
            }
        )
        random_line_1_to_8_hard_dataset.append(
            {
                "index": index,
                "full_hard": random_line_1_to_8_hard_grid_2d_str,
                "ground_truth_hard": ground_truth_hard_grid_2d_str,
            }
        )


datasets = [
    random_cell_1_easy_dataset,
    random_cell_1_hard_dataset,
    random_cell_5_easy_dataset,
    random_cell_5_hard_dataset,
    random_cell_10_easy_dataset,
    random_cell_10_hard_dataset,
    random_line_1_easy_dataset,
    random_line_1_hard_dataset,
    random_line_3_easy_dataset,
    random_line_3_hard_dataset,
    random_column_1_easy_dataset,
    random_column_1_hard_dataset,
    random_column_3_easy_dataset,
    random_column_3_hard_dataset,
    random_line_1_to_8_easy_dataset,
    random_line_1_to_8_hard_dataset,
]

for dataset in datasets:
    subject = list(dataset[0].keys())[1]
    with open(f"{output_path}/{subject}.json", "w") as f:
        json.dump(dataset, f, indent=4)

from decimal import Decimal


def get_exact_match(output_grid, gt_grid):
    return count_differences(output_grid, gt_grid) == 0


def count_differences(reference_grid, gt_grid) -> int:
    count = 0
    for row1, row2 in zip(reference_grid, gt_grid):
        for cell1, cell2 in zip(row1, row2):
            if cell1 != cell2:
                count += 1
    return count


def count_relative_differences(reference_grid, gt_grid):
    count = 0.0
    for row1, row2 in zip(reference_grid, gt_grid):
        for cell1, cell2 in zip(row1, row2):
            if cell1 == cell2:
                continue
            if cell1 in ["L", "S", "V"] or cell2 in ["L", "S", "V"]:
                count += 1.0
                continue
            try:
                a_float = Decimal(cell1)
                b_float = Decimal(cell2)
                count += float(abs(a_float - b_float))
            except ValueError:
                count += 1.0
                continue

    return count


def count_differences_with_penalty(reference_grid, gt_grid, penalty=3) -> float:
    count = 0.0
    for row1, row2 in zip(reference_grid, gt_grid):
        for cell1, cell2 in zip(row1, row2):
            if cell1 == cell2:
                continue
            if cell1 in ["L", "S", "V"] or cell2 in ["L", "S", "V"]:
                count += penalty
                continue
            try:
                a_float = Decimal(cell1)
                b_float = Decimal(cell2)
                count += float(abs(a_float - b_float))
            except ValueError:
                count += 1.0
                continue

    return count


def calculate_score(output_gt_difference_count, input_gt_difference_count) -> float:
    if input_gt_difference_count == 0:
        # Input already perfect — only a perfect output keeps score = 1
        return 1.0 if output_gt_difference_count == 0 else 0.0

    score = 1 - (output_gt_difference_count / input_gt_difference_count)
    return score


def grid_value_sum(grid) -> float:
    total = 0.0
    for row in grid:
        for cell in row:
            if cell in ["L", "S", "V"]:
                continue
            try:
                total += float(Decimal(cell))
            except ValueError:
                continue
    return total


def get_difference_ratio(output_grid, gt_grid) -> float:
    gt_grid_value_sum = grid_value_sum(gt_grid)

    output_gt_difference_count = count_differences(
        reference_grid=output_grid, gt_grid=gt_grid
    )

    if output_gt_difference_count == 0:
        return 1.0

    difference_ratio = 1 - (output_gt_difference_count / gt_grid_value_sum)
    return difference_ratio


def get_relative_difference_ratio(output_grid, gt_grid) -> float:
    gt_grid_value_sum = grid_value_sum(gt_grid)
    output_gt_difference_count = count_relative_differences(
        reference_grid=output_grid, gt_grid=gt_grid
    )

    if output_gt_difference_count == 0:
        return 1.0

    relative_difference_ratio = 1 - (output_gt_difference_count / gt_grid_value_sum)
    return relative_difference_ratio


def get_penalized_difference_ratio(output_grid, gt_grid) -> float:
    gt_grid_value_sum = grid_value_sum(gt_grid)
    output_gt_difference_count = count_differences_with_penalty(
        reference_grid=output_grid, gt_grid=gt_grid
    )

    if output_gt_difference_count == 0:
        return 1.0

    relative_difference_ratio = 1 - (output_gt_difference_count / gt_grid_value_sum)
    return relative_difference_ratio

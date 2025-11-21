def is_solid(value):
    """Return True if the cell can transmit force (solid or partially solid)."""
    if value in ("L", "S"):
        return True
    try:
        return float(value) > 0.0
    except ValueError:
        return False


def get_neighbors(r, c, nrows, ncols):
    # left, right, down, up
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # diagonals
    dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            yield nr, nc


def get_gravity_from_folder(folder_name: str):
    """
    Determines gravity direction (force flow) based on rotation info in folder name.
    Returns a (dr, dc) tuple:
      (1, 0)  = down (default)
      (0, 1)  = right
      (-1, 0) = up
      (0, -1) = left
    """
    if "3_rotations" in folder_name:
        return (0, 1)  # left → right
    elif "2_rotations" in folder_name:
        return (-1, 0)  # bottom → top
    elif "1_rotations" in folder_name:
        return (0, -1)  # right → left
    else:
        return (1, 0)  # top → bottom (default)


def get_grid_shape_and_value_validity(output_grid, gt_grid):

    value_validity = True
    for row in output_grid:
        for cell in row:
            if cell in ["L", "S"]:
                continue
            try:
                val = float(cell)
                if val < 0.0 or val > 1.0:
                    value_validity = False
            except ValueError:
                value_validity = False

    shape_validity = len(output_grid) == len(gt_grid) and all(
        len(output_grid[r]) == len(gt_grid[r]) for r in range(len(output_grid))
    )

    return value_validity and shape_validity

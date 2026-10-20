import re


SCORING_POLICY = "masked-only-v1"


def complete_masked_grid(input_grid, output_grid):
    """Insert submitted masked values into the input, preserving visible cells.

    Shape/value validation belongs to the caller. A misshapen answer must never
    become valid through reconstruction. Neither supplied grid is modified.
    """
    if len(input_grid) != len(output_grid) or any(
        len(given) != len(predicted)
        for given, predicted in zip(input_grid, output_grid)
    ):
        raise ValueError("Input and completion must have the same shape")
    return [
        [predicted if given == "V" else given for given, predicted in zip(a, b)]
        for a, b in zip(input_grid, output_grid)
    ]


def visible_cell_changes(input_grid, output_grid):
    """Count copying errors independently of the reconstructed design's score.

    Numeric spellings such as 0 and 0.0 describe the same material state.
    Load/support symbols must match exactly. Call only for a valid grid shape.
    """
    numeric = boundary = 0
    for given_row, predicted_row in zip(input_grid, output_grid):
        for given, predicted in zip(given_row, predicted_row):
            if given == "V" or given == predicted:
                continue
            if given in ("L", "S"):
                boundary += 1
                continue
            try:
                equal = float(given) == float(predicted)
            except (TypeError, ValueError):
                equal = False
            numeric += not equal
    return {
        "visible_cell_changes": numeric + boundary,
        "visible_numeric_cell_changes": numeric,
        "visible_boundary_cell_changes": boundary,
        "visible_cells_preserved": numeric + boundary == 0,
    }


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


GRID_BLOCK_RE = re.compile(
    r"(?:^|\n)(?:[0-9LVSlvs\.]+(?:\s+[0-9LVSlvs\.]+)+\s*\n?)+", re.MULTILINE
)


def extract_grid_from_text(text):
    """Parse the last grid block out of a prompt or a model completion.

    Models often wrap the grid in prose or code fences, so the last block of
    whitespace-separated grid tokens is taken as the answer.
    """
    matches = GRID_BLOCK_RE.findall(text)
    if not matches:
        return []
    return [line.split() for line in matches[-1].strip().splitlines()]


# One entry of a sparse answer: ``row,col`` or ``row,col,value``, optionally
# bulleted or parenthesised.  Commas keep an entry from ever reading as a grid
# row, so a prompt that carries both a grid and a sparse answer parses
# unambiguously.
SPARSE_ENTRY_RE = re.compile(
    r"^[-*\s]*\(?\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([+-]?\d*\.?\d+))?\s*\)?[\s,;.]*$"
)
SPARSE_NONE_RE = re.compile(r"^\W*none\W*$", re.IGNORECASE)


def parse_sparse_answer(text, input_grid, density=False):
    """Rebuild a full grid from an answer that lists only the solid masked cells.

    The sparse answer format exists to take serialisation out of the score:
    the agent never writes the grid, only the masked cells that should carry
    material, so it cannot miscount a row or stop a grid short.  Every masked
    cell it does not list is empty and every visible cell keeps its value.

    An entry naming a cell that is not masked, or lies outside the grid, is
    dropped and counted rather than voiding the answer -- the point of the
    format is that bookkeeping slips cost the cells they touch, not the whole
    sample.  A density entry without a value is dropped the same way.

    Returns ``(grid, info)``.  The grid is empty when the reply contains no
    entry at all and does not say ``NONE``, which scores as an unreadable
    answer, like a grid reply with no grid in it.
    """
    empty = "0.0" if density else "0"
    grid = [[empty if cell == "V" else cell for cell in row] for row in input_grid]
    accepted, rejected, said_none = 0, 0, False

    segments = (
        segment.strip()
        for line in text.replace("`", "").splitlines()
        for segment in line.split(";")
    )
    for segment in segments:
        if not segment:
            continue
        if SPARSE_NONE_RE.match(segment):
            said_none = True
            continue
        match = SPARSE_ENTRY_RE.match(segment)
        if not match:
            continue

        r, c, value = int(match.group(1)), int(match.group(2)), match.group(3)
        masked = 0 <= r < len(input_grid) and 0 <= c < len(input_grid[r]) and (
            input_grid[r][c] == "V"
        )
        if not masked or (density and value is None):
            rejected += 1
            continue

        grid[r][c] = value if value is not None else "1"
        accepted += 1

    info = {"sparse_entries": accepted, "sparse_rejected_entries": rejected}
    if not accepted and not said_none:
        return [], info
    return grid, info


def parse_completion(text, input_grid, answer_format="grid", density=False):
    """Turn an agent's reply into the grid it describes, in either answer format.

    Returns ``(grid, info)``; ``info`` carries format-specific bookkeeping and
    is empty for grid answers.
    """
    if answer_format == "sparse":
        return parse_sparse_answer(text, input_grid, density=density)
    return extract_grid_from_text(text), {}

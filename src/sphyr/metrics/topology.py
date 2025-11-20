from collections import deque
from sphyr.metrics.utils import get_neighbors, is_solid


def is_load_supported(grid):
    """
    Returns True if any 'L' is connected to any 'S' via solid material
    ('L', 'S', or numeric value > 0.0).
    """
    nrows, ncols = len(grid), len(grid[0])

    loads = [(r, c) for r in range(nrows) for c in range(ncols) if grid[r][c] == "L"]
    supports = {(r, c) for r in range(nrows) for c in range(ncols) if grid[r][c] == "S"}

    if not loads or not supports:
        return False  # no load or support present

    visited = [[False] * ncols for _ in range(nrows)]

    q = deque(loads)
    for r, c in loads:
        visited[r][c] = True

    while q:
        r, c = q.popleft()
        if (r, c) in supports:
            return True  # found a path to support!

        for nr, nc in get_neighbors(r, c, nrows, ncols):
            if not visited[nr][nc] and is_solid(grid[nr][nc]):
                visited[nr][nc] = True
                q.append((nr, nc))

    return False


def is_load_supported_force_directional(grid, gravity_dir=(1, 0)):
    """
    Checks if any 'L' is connected to any 'S' via solid material ('1', 'L', 'S'),
    allowing force propagation aligned with gravity_dir.
    gravity_dir = (dr, dc) e.g. (1,0)=down, (0,1)=right, (-1,0)=up, (0,-1)=left
    """
    nrows, ncols = len(grid), len(grid[0])
    loads = [(r, c) for r in range(nrows) for c in range(ncols) if grid[r][c] == "L"]

    for lr, lc in loads:
        stack = [(lr, lc)]
        visited = {(lr, lc)}

        while stack:
            r, c = stack.pop()
            if grid[r][c] == "S":
                return True

            drg, dcg = gravity_dir
            # main + diagonal + lateral around gravity direction
            directions = [
                (dr, dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr != 0 or dc != 0) and (dr * drg >= 0 and dc * dcg >= 0)
            ]
            if drg == 1 and dcg == 0:
                directions = [(dr, dc) for dr, dc in directions if dr >= 0]

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    if (nr, nc) not in visited and is_solid(grid[nr][nc]):
                        visited.add((nr, nc))
                        stack.append((nr, nc))
    return False


def get_force_directed_neighbors(r, c, nrows, ncols):
    """Allowed downward and lateral directions for force propagation."""
    dirs = [(1, 0), (0, -1), (0, 1)]  # down, left, right
    dirs += [(1, -1), (1, 1)]  # down-left, down-right
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            yield nr, nc


def get_isolated_clusters_count(grid):
    """
    Finds clusters of solid cells (non-zero) not connected to L or S.
    Uses 4-connectivity by default for clearer isolation.
    """
    nrows, ncols = len(grid), len(grid[0])
    visited = [[False] * ncols for _ in range(nrows)]
    islands = []

    for r in range(nrows):
        for c in range(ncols):
            val = grid[r][c]
            if visited[r][c] or not is_solid(val) or val in ("L", "S"):
                continue

            cluster = []
            connected_to_structure = False
            q = deque([(r, c)])

            while q:
                cr, cc = q.popleft()
                if visited[cr][cc]:
                    continue
                visited[cr][cc] = True
                cluster.append((cr, cc))

                if grid[cr][cc] in ("L", "S"):
                    connected_to_structure = True

                for nr, nc in get_neighbors(cr, cc, nrows, ncols):
                    if not visited[nr][nc] and is_solid(grid[nr][nc]):
                        q.append((nr, nc))

            if not connected_to_structure:
                islands.append(cluster)

    return len(islands)


def get_difficulty_score(input_grid, gt_grid):
    """
    Computes a Difficulty-Weighted Completion Score (DWCS).
    Each originally masked cell ('V' in input) is assigned a difficulty weight based
    on its ground-truth neighborhood configuration using 8-neighbor connectivity.

    Difficulty levels:
        1.0 → easy  (neighbors uniform, same as GT)
        2.0 → hard  (neighbors uniform, opposite to GT)
        3.0 → boundary/ambiguous (mixed neighbors)
    """

    nrows, ncols = len(input_grid), len(input_grid[0])
    difficulty = 0.0
    void_count = 0

    for r in range(nrows):
        for c in range(ncols):
            if input_grid[r][c] != "V":
                continue  # only evaluate masked cells
            void_count += 1
            gt_val = gt_grid[r][c]

            # collect numeric neighbor values from ground truth
            neighbor_vals = []
            for nr, nc in get_neighbors(r, c, nrows, ncols):
                v = gt_grid[nr][nc]
                if v not in ("L", "S", "V"):
                    try:
                        neighbor_vals.append(float(v))
                    except ValueError:
                        continue

            if not neighbor_vals:
                continue  # no usable neighbors

            all_zeros = all(v == 0.0 for v in neighbor_vals)
            all_ones = all(v == 1.0 for v in neighbor_vals)

            # determine GT value numerically
            try:
                gt_num = float(gt_val)
            except ValueError:
                gt_num = 1.0 if gt_val in ("L", "S") else 0.0

            # assign difficulty weight
            if all_zeros or all_ones:
                if (all_ones and gt_num == 1.0) or (all_zeros and gt_num == 0.0):
                    w = 1.0  # easy
                else:
                    w = 2.0  # hard
            else:
                w = 3.0  # ambiguous boundary

            difficulty += w

    if void_count == 0 or difficulty == 0.0:
        return 0.0

    return difficulty / void_count

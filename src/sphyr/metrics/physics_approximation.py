import heapq
import math

from sphyr.metrics.utils import get_gravity_from_folder, is_solid


def generate_direction_costs(gravity_dir=(1, 0)):
    base_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    direction_costs = []
    for dr, dc in base_dirs:
        dot = dr * gravity_dir[0] + dc * gravity_dir[1]
        mag1 = math.sqrt(dr**2 + dc**2)
        mag2 = math.sqrt(gravity_dir[0] ** 2 + gravity_dir[1] ** 2)
        cos_theta = dot / (mag1 * mag2 + 1e-9)
        angle = math.degrees(math.acos(max(-1, min(1, cos_theta))))
        if angle < 15:
            cost = 1.0
        elif angle < 45:
            cost = 1.2
        elif angle < 100:
            cost = 1.5
        else:
            cost = 3.0
        direction_costs.append((dr, dc, cost))
    return direction_costs


def get_total_force_path_cost(grid, max_cost=1000.0, gravity_dir=(1, 0)):
    """
    Computes a physically meaningful directional load path cost from each 'L' to any 'S',
    aligned with gravity_dir.

    Improvements:
      - Allows sideways (neutral) movement but blocks upward (against gravity).
      - Unsupported loads get a finite penalty (max_cost) instead of inf.
      - Includes depth penalty for longer vertical travel.
      - Returns per-load costs and average cost.
    """

    DIRECTION_COSTS = generate_direction_costs(gravity_dir)
    nrows, ncols = len(grid), len(grid[0])

    loads = [(r, c) for r in range(nrows) for c in range(ncols) if grid[r][c] == "L"]
    supports = {(r, c) for r in range(nrows) for c in range(ncols) if grid[r][c] == "S"}

    if not loads or not supports:
        return 0.0

    results = {}
    total_cost = 0.0
    valid_count = 0

    for lr, lc in loads:
        pq = [(0.0, lr, lc)]
        dist = [[float("inf")] * ncols for _ in range(nrows)]
        dist[lr][lc] = 0.0
        found = False

        while pq:
            cost, r, c = heapq.heappop(pq)
            if cost > dist[r][c]:
                continue

            # Stop if we reach a support
            if (r, c) in supports:
                results[(lr, lc)] = cost
                total_cost += cost
                valid_count += 1
                found = True
                break

            for dr, dc, w in DIRECTION_COSTS:
                # Allow sideways moves, forbid strong uphill movement
                dot = dr * gravity_dir[0] + dc * gravity_dir[1]
                if dot < -0.5:  # e.g., if gravity is (1,0), forbid dr < 0 (upward)
                    continue

                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols and is_solid(grid[nr][nc]):
                    depth_penalty = 1.0 + 0.05 * abs(nr - lr)
                    new_cost = cost + w * depth_penalty
                    if new_cost < dist[nr][nc] and new_cost < max_cost:
                        dist[nr][nc] = new_cost
                        heapq.heappush(pq, (new_cost, nr, nc))

        if not found:
            # Penalize unsupported loads with a large but finite cost
            results[(lr, lc)] = max_cost
            total_cost += max_cost
            valid_count += 1

    avg_cost = total_cost / valid_count if valid_count > 0 else 0.0
    return avg_cost


def get_total_force_path_cost_average_efficiency_ratio(
    output_grid, gt_grid, gravity_dir
) -> float:
    output_total_force_path_cost = get_total_force_path_cost(
        output_grid, gravity_dir=gravity_dir
    )
    gt_total_force_path_cost = get_total_force_path_cost(
        gt_grid, gravity_dir=gravity_dir
    )
    total_force_path_cost_average_efficiency_ratio = 0
    if output_total_force_path_cost > 0:
        total_force_path_cost_average_efficiency_ratio = max(
            0.0,
            min(
                1.0,
                gt_total_force_path_cost / output_total_force_path_cost,
            ),
        )

    return total_force_path_cost_average_efficiency_ratio

"""Bidirectional RRT-Connect planning helpers for the X3Plus joint space."""

import time

import numpy as np
from pydrake.trajectories import PiecewisePolynomial

from x3plus_drake_planner.planning_utils.iris_helper import solve_ik_in_region


class _Tree:
    """Array-backed search tree with parent indices."""

    def __init__(self, q_root, capacity=512):
        q_root = np.asarray(q_root, dtype=float)
        self.q = np.empty((capacity, len(q_root)))
        self.parent = np.full(capacity, -1, dtype=int)
        self.q[0] = q_root
        self.size = 1

    def nearest(self, q):
        distances = np.linalg.norm(self.q[:self.size] - q, axis=1)
        return int(np.argmin(distances))

    def add(self, q, parent):
        if self.size == len(self.q):
            self.q = np.vstack((self.q, np.empty_like(self.q)))
            self.parent = np.concatenate(
                (self.parent, np.full(len(self.parent), -1, dtype=int))
            )
        self.q[self.size] = q
        self.parent[self.size] = parent
        self.size += 1
        return self.size - 1

    def path_to_root(self, index):
        path = []
        while index != -1:
            path.append(self.q[index].copy())
            index = self.parent[index]
        return path[::-1]


def _extend(tree, q_target, step, checker):
    """Extend a tree once toward a target, unless the edge is obstructed."""
    nearest_index = tree.nearest(q_target)
    q_near = tree.q[nearest_index]
    difference = q_target - q_near
    distance = np.linalg.norm(difference)
    if distance < 1e-12:
        return "Reached", nearest_index

    if distance <= step:
        q_new, status = q_target.copy(), "Reached"
    else:
        q_new = q_near + difference * (step / distance)
        status = "Advanced"

    if not checker.CheckConfigCollisionFree(q_new):
        return "Trapped", nearest_index
    if not checker.CheckEdgeCollisionFree(q_near, q_new):
        return "Trapped", nearest_index
    return status, tree.add(q_new, nearest_index)


def _connect(tree, q_target, step, checker, max_steps=1000):
    """Greedily extend a tree toward a target until reached or trapped."""
    status, index = "Advanced", None
    for _ in range(max_steps):
        status, index = _extend(tree, q_target, step, checker)
        if status != "Advanced":
            break
    return status, index


def rrt_connect(
    checker,
    q_start,
    q_goal,
    q_low,
    q_high,
    step=0.25,
    max_iters=5000,
    seed=0,
):
    """Plan a joint-space path with bidirectional RRT-Connect.

    Returns ``(path, iterations)`` where path includes both endpoints, or
    ``(None, max_iters)`` when no path is found.
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a finite positive value")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    if not (
        q_start.shape == q_goal.shape == q_low.shape == q_high.shape
        and q_start.ndim == 1
    ):
        raise ValueError("Start, goal, and joint limit vectors must have matching shapes")
    if not (
        np.all(np.isfinite(q_start))
        and np.all(np.isfinite(q_goal))
        and np.all(np.isfinite(q_low))
        and np.all(np.isfinite(q_high))
        and np.all(q_low <= q_high)
    ):
        raise ValueError("Configurations and limits must be finite and ordered")
    if np.any(q_start < q_low) or np.any(q_start > q_high):
        raise ValueError("q_start is outside the supplied joint limits")
    if np.any(q_goal < q_low) or np.any(q_goal > q_high):
        raise ValueError("q_goal is outside the supplied joint limits")
    if not checker.CheckConfigCollisionFree(q_start):
        raise ValueError("q_start is in collision")
    if not checker.CheckConfigCollisionFree(q_goal):
        raise ValueError("q_goal is in collision")

    rng = np.random.default_rng(seed)
    tree_a, tree_b = _Tree(q_start), _Tree(q_goal)
    a_is_start = True

    for iteration in range(1, max_iters + 1):
        q_random = rng.uniform(q_low, q_high)
        status_a, index_a = _extend(tree_a, q_random, step, checker)

        if status_a != "Trapped":
            status_b, index_b = _connect(
                tree_b, tree_a.q[index_a], step, checker
            )
            if status_b == "Reached":
                path_a = tree_a.path_to_root(index_a)
                path_b = tree_b.path_to_root(index_b)
                if a_is_start:
                    path = path_a + path_b[::-1][1:]
                else:
                    path = path_b + path_a[::-1][1:]
                return path, iteration

        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start

    return None, max_iters


def shortcut_path(path, checker, rng, iters=100):
    """Remove intermediate waypoints when a direct collision-free edge exists."""
    path = [np.asarray(point, dtype=float).copy() for point in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        first, second = sorted(rng.choice(len(path), 2, replace=False))
        if second - first < 2:
            continue
        if checker.CheckEdgeCollisionFree(path[first], path[second]):
            path = path[:first + 1] + path[second:]
    return path


def path_length(path):
    """Return the joint-space path length in radians."""
    return float(
        sum(
            np.linalg.norm(second - first)
            for first, second in zip(path[:-1], path[1:])
        )
    )


def resample_path(path, spacing):
    """Insert waypoints so consecutive joint-space points are within spacing."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a finite positive value")
    if not path:
        raise ValueError("path must contain at least one waypoint")

    result = [np.asarray(path[0], dtype=float).copy()]
    for first, second in zip(path[:-1], path[1:]):
        distance = np.linalg.norm(second - first)
        count = max(1, int(np.ceil(distance / spacing)))
        for index in range(1, count + 1):
            result.append(first + (second - first) * index / count)
    return result


def path_to_trajectory(path, v_max, vel_scale=0.5, min_dt=0.02):
    """Create a cubic trajectory from joint waypoints and velocity limits."""
    if len(path) < 2:
        raise ValueError("path must contain at least two waypoints")
    if not np.isfinite(vel_scale) or vel_scale <= 0:
        raise ValueError("vel_scale must be a finite positive value")
    if not np.isfinite(min_dt) or min_dt <= 0:
        raise ValueError("min_dt must be a finite positive value")

    if len(path) == 2:
        path = [path[0], 0.5 * (path[0] + path[1]), path[1]]

    waypoints = np.column_stack(path)
    max_velocity = np.asarray(v_max, dtype=float)
    if max_velocity.shape != (waypoints.shape[0],):
        raise ValueError("v_max must have one value per joint")
    max_velocity = np.where(
        np.isfinite(max_velocity) & (max_velocity > 0),
        max_velocity,
        1.0,
    )

    durations = [
        max(
            min_dt,
            float(
                np.max(
                    np.abs(second - first) / (max_velocity * vel_scale)
                )
            ),
        )
        for first, second in zip(path[:-1], path[1:])
    ]
    breaks = np.concatenate(([0.0], np.cumsum(durations)))
    zeros = np.zeros(waypoints.shape[0])
    return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        breaks, waypoints, zeros, zeros
    )


def colliding_times(trajectory, checker, n=400):
    """Return sampled times where a trajectory configuration is in collision."""
    if n < 2:
        raise ValueError("n must be at least 2")
    times = np.linspace(trajectory.start_time(), trajectory.end_time(), n)
    return [
        time_value
        for time_value in times
        if not checker.CheckConfigCollisionFree(
            trajectory.value(time_value).ravel()
        )
    ]


def find_goal_config(
    plant,
    plant_context,
    gripper_frame,
    world_frame,
    domain,
    checker,
    target_xyz,
    q0,
    q_low,
    q_high,
    tol=np.array([0.01, 0.005, 0.01]),
    n_random=30,
    seed=0,
):
    """Try IK from q0 and random starts, returning the found solution nearest q0."""
    if n_random < 0:
        raise ValueError("n_random must be non-negative")
    rng = np.random.default_rng(seed)
    q0 = np.asarray(q0, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    if not (q0.shape == q_low.shape == q_high.shape):
        raise ValueError("q0 and joint limit vectors must have matching shapes")

    starts = [q0] + [
        rng.uniform(q_low, q_high) for _ in range(n_random)
    ]
    solutions = []
    for q_init in starts:
        solution = solve_ik_in_region(
            plant,
            plant_context,
            gripper_frame,
            world_frame,
            domain,
            q_init,
            target_xyz,
            tol,
            checker,
        )
        if solution is not None:
            solutions.append(np.asarray(solution, dtype=float))

    if not solutions:
        return None
    return min(solutions, key=lambda solution: np.linalg.norm(solution - q0))


def plan_rrt_connect(
    checker,
    plant,
    q_start,
    q_goal,
    q_low,
    q_high,
    step=0.25,
    max_iters=5000,
    shortcut_iters=100,
    resample_spacings=(0.1, 0.025),
    vel_scale=0.5,
    seed=0,
):
    """Plan, shortcut, and time-parameterize a collision-checked RRT path.

    Returns ``(trajectory, info, shortcut_path, raw_path)``. Trajectory
    collision checking is sampled; each RRT and shortcut edge is checked by
    the supplied Drake collision checker.
    """
    if not resample_spacings:
        raise ValueError("resample_spacings must contain at least one spacing")

    start_time = time.perf_counter()
    raw_path, iterations = rrt_connect(
        checker,
        q_start,
        q_goal,
        q_low,
        q_high,
        step=step,
        max_iters=max_iters,
        seed=seed,
    )
    if raw_path is None:
        raise RuntimeError(
            f"RRT-Connect found no path in {max_iters} iterations"
        )
    rrt_time = time.perf_counter() - start_time

    shortcut_start = time.perf_counter()
    shortcut_rng = np.random.default_rng(seed)
    path = shortcut_path(
        raw_path, checker, shortcut_rng, iters=shortcut_iters
    )
    shortcut_time = time.perf_counter() - shortcut_start

    velocity_lower = np.asarray(plant.GetVelocityLowerLimits())
    velocity_upper = np.asarray(plant.GetVelocityUpperLimits())
    max_velocity = np.minimum(-velocity_lower, velocity_upper)
    max_velocity = np.where(
        np.isfinite(max_velocity) & (max_velocity > 0),
        max_velocity,
        1.0,
    )

    trajectory_start = time.perf_counter()
    trajectory = None
    used_spacing = None
    for spacing in resample_spacings:
        candidate = path_to_trajectory(
            resample_path(path, spacing),
            max_velocity,
            vel_scale=vel_scale,
        )
        if not colliding_times(candidate, checker):
            trajectory = candidate
            used_spacing = spacing
            break

    if trajectory is None:
        print(
            "WARNING: sampled spline trajectories collided; using "
            "first-order hold (velocity discontinuous)."
        )
        dense_path = resample_path(path, resample_spacings[-1])
        durations = [
            max(
                0.02,
                float(
                    np.max(
                        np.abs(second - first)
                        / (max_velocity * vel_scale)
                    )
                ),
            )
            for first, second in zip(dense_path[:-1], dense_path[1:])
        ]
        breaks = np.concatenate(([0.0], np.cumsum(durations)))
        trajectory = PiecewisePolynomial.FirstOrderHold(
            breaks, np.column_stack(dense_path)
        )
        if colliding_times(trajectory, checker):
            raise RuntimeError(
                "The fallback trajectory also failed sampled collision checking"
            )
    trajectory_time = time.perf_counter() - trajectory_start

    info = {
        "iterations": iterations,
        "n_waypoints_raw": len(raw_path),
        "n_waypoints_short": len(path),
        "length_raw": path_length(raw_path),
        "length_short": path_length(path),
        "duration": trajectory.end_time(),
        "spacing": used_spacing,
        "t_rrt": rrt_time,
        "t_shortcut": shortcut_time,
        "t_traj": trajectory_time,
        "t_total": time.perf_counter() - start_time,
    }
    return trajectory, info, path, raw_path

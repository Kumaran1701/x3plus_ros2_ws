import numpy as np

from iris_helper import grow_iris_region, solve_ik_in_region


# ------------------------------------------- HELPERS ----------------------------------------------

def task_pos(plant, context, frame, q):
    """End-effector world position for configuration q."""
    plant.SetPositions(context, q)
    return plant.CalcRelativeTransform(context, plant.world_frame(), frame).translation()


def sample_informed_candidates(p_start, p_goal, n, rng, stretch=1.3):
    """
    Sample n task-space points inside a prolate spheroid whose two foci are
    p_start and p_goal (the Informed RRT* sampling set, Gammell et al.).

    stretch = c_max / c_min: 1.0 collapses to the segment, larger = fatter.
    Returned points are sorted by distance to p_goal (nearest first).
    """
    c_min = np.linalg.norm(p_goal - p_start)
    if c_min < 1e-9:
        return np.repeat(p_goal[None, :], n, axis=0)

    c_max = stretch * c_min
    a = 0.5 * c_max                                  # semi-transverse axis
    b = 0.5 * np.sqrt(c_max**2 - c_min**2)           # semi-conjugate axes
    center = 0.5 * (p_start + p_goal)

    # Orthonormal basis with first axis along start -> goal
    d = (p_goal - p_start) / c_min
    helper = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e2 = np.cross(d, helper)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(d, e2)
    R = np.column_stack([d, e2, e3])

    # Uniform samples in the unit ball, then scale and rotate
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x *= rng.random((n, 1)) ** (1.0 / 3.0)
    pts = center + (x * np.array([a, b, b])) @ R.T

    order = np.argsort(np.linalg.norm(pts - p_goal, axis=1))
    return pts[order]


def bisect_toward_target(plant, context, gripper_frame, world_frame, region,
                         q_start, target_xyz, tol, checker, n_iter=6):
    """
    Bisection on s in [0, 1] along p(s) = p_start + s * (target - p_start).
    s = 0 is q_start (known feasible, inside region); s = 1 is the target
    (assumed already infeasible in this region).

    Each probe is one region-constrained IK solve, warm-started from the
    best config so far. Returns (q_best, s_best).

    Note: feasibility along the segment is not guaranteed monotone, so this
    finds *a* boundary of feasibility, not provably the farthest point.
    Cost is bounded: exactly n_iter IK solves.
    """
    p_start = task_pos(plant, context, gripper_frame, q_start)
    q_best, lo, hi = q_start, 0.0, 1.0

    for _ in range(n_iter):
        s = 0.5 * (lo + hi)
        p = p_start + s * (target_xyz - p_start)
        q = solve_ik_in_region(plant, context, gripper_frame, world_frame,
                               region, q_best, p, tol, checker)
        if q is None:
            hi = s
        else:
            lo, q_best = s, q

    return q_best, lo


def spheroid_fallback(plant, context, gripper_frame, world_frame, region,
                      q_start, target_xyz, tol, checker, rng,
                      d_current, min_progress, n_candidates=15, stretch=1.3):
    """
    Try spheroid-sampled task-space candidates (nearest-to-target first) with
    region-constrained IK. Returns the first feasible q, or None.
    Candidates that wouldn't improve on d_current by min_progress are skipped.
    """
    p_start = task_pos(plant, context, gripper_frame, q_start)
    cands = sample_informed_candidates(p_start, target_xyz, 4 * n_candidates, rng, stretch)
    cands = [p for p in cands
             if np.linalg.norm(p - target_xyz) < d_current - min_progress][:n_candidates]

    for p in cands:
        q = solve_ik_in_region(plant, context, gripper_frame, world_frame,
                               region, q_start, p, tol, checker)
        if q is not None:
            return q
    return None


def regions_overlap(r1, r2):
    return not r1.Intersection(r2).IsEmpty()


# ------------------------------------------- MAIN CHAIN -------------------------------------------

def grow_region_chain_informed(
    checker,
    plant,
    context,
    gripper_frame,
    world_frame,
    q0,
    target_xyz,
    domain,
    iris_zo_options,
    final_tol=np.array([0.01, 0.01, 0.01]),
    max_hops=6,
    n_bisect=6,
    min_progress=0.01,      # metres of end-effector progress required per hop
    n_fallback=15,
    stretch=1.3,
    seed=0,
):
    """
    Returns: region_chain, q_chain, q_goal   (same as grow_region_chain)
    """
    rng = np.random.default_rng(seed)
    target_xyz = np.asarray(target_xyz, dtype=float)

    home_region = grow_iris_region(checker, q0, domain, iris_zo_options)
    region_chain = [home_region]
    q_chain = [q0]

    # --- Stage 1: direct IK to the goal (only joint limits / domain constrain it) ---
    q_goal = solve_ik_in_region(plant, context, gripper_frame, world_frame,
                                domain, q0, target_xyz, final_tol, checker)
    goal_region = None

    if q_goal is not None:
        goal_region = grow_iris_region(checker, q_goal, domain, iris_zo_options)

        if regions_overlap(home_region, goal_region):
            print("Direct IK succeeded and goal region overlaps home region.")
            return [home_region, goal_region], [q0, q_goal], q_goal

        print("Direct IK succeeded but goal region does not overlap home - bridging.")
    else:
        print("Direct IK to target failed - growing chain.")

    # --- Stage 2: chain regions toward the target ---
    for hop in range(max_hops):
        current_region = region_chain[-1]
        q_current = q_chain[-1]

        d_current = np.linalg.norm(task_pos(plant, context, gripper_frame, q_current) - target_xyz)

        # Without a known goal region, test whether the target is reachable from here
        if goal_region is None:
            q_goal = solve_ik_in_region(plant, context, gripper_frame, world_frame,
                                        current_region, q_current, target_xyz,
                                        final_tol, checker)
            if q_goal is not None:
                goal_region = grow_iris_region(checker, q_goal, domain, iris_zo_options)
                region_chain.append(goal_region)
                q_chain.append(q_goal)
                print(f"Reached target at hop {hop}. Chain length: {len(region_chain)}.")
                return region_chain, q_chain, q_goal

        # Bisection along the straight line to the target
        q_next, s = bisect_toward_target(plant, context, gripper_frame, world_frame,
                                         current_region, q_current, target_xyz,
                                         final_tol, checker, n_iter=n_bisect)
        d_next = np.linalg.norm(task_pos(plant, context, gripper_frame, q_next) - target_xyz)
        print(f"Hop {hop}: bisection s={s:.3f}, {d_current:.4f} -> {d_next:.4f} m from target.")

        # Straight line blocked: fall back to spheroid candidates
        if d_current - d_next < min_progress:
            print(f"Hop {hop}: bisection stalled, trying spheroid candidates.")
            q_alt = spheroid_fallback(plant, context, gripper_frame, world_frame,
                                      current_region, q_current, target_xyz, final_tol,
                                      checker, rng, d_current, min_progress,
                                      n_candidates=n_fallback, stretch=stretch)
            if q_alt is None:
                raise RuntimeError(f"Hop {hop}: no progress from bisection or spheroid "
                                   f"fallback - region may be boxed in.")
            q_next = q_alt

        # One IrisZo call per hop, at the best point found
        next_region = grow_iris_region(checker, q_next, domain, iris_zo_options)
        region_chain.append(next_region)
        q_chain.append(q_next)

        # With a known goal region, finish as soon as the chain touches it
        if goal_region is not None and regions_overlap(next_region, goal_region):
            region_chain.append(goal_region)
            q_chain.append(q_goal)
            print(f"Chain reached goal region at hop {hop}. Chain length: {len(region_chain)}.")
            return region_chain, q_chain, q_goal

    raise RuntimeError(f"Exceeded max_hops={max_hops} without reaching the target.")
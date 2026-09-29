import time
import numpy as np

from pydrake.all import (
    Context,
    HPolyhedron,
    MathematicalProgram,
    MultibodyPlant,
    Solve,
)

from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import Hyperellipsoid
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.planning import IrisZo


# ------------------------------ PART 1: SEQUENTIAL IRIS REGION GENERATION ----------------------------

# -------------------- 1(A) GROW A VERIFIED IRIS REGUIN AROUND A SEED CONFIGURATION -------------------

def grow_iris_region(
    checker,
    center_q,
    domain,
    iris_zo_options,
    radius=1e-3,           # Initial radius of the seed hypersphere.
    max_retries=3,         # Maximum number of IrisZo attempts.
    containment_tol=1e-9,  # Maximum allowed seed containment violation.
):
    """
    Grow an IrisZo region around center_q and explicitly verify that
    the seed point is contained in the resulting region.

    IrisZo can internally encounter an "initial seed point not contained
    in region" condition without raising a Python exception. Therefore,
    the returned region is checked explicitly before being accepted.

    If the seed is not contained, the starting ellipsoid radius is
    increased and IrisZo is run again.
    """

    for attempt in range(max_retries):

        this_radius = radius * (2 ** attempt)

        ellipsoid = Hyperellipsoid.MakeHypersphere(radius=this_radius, center=center_q)

        region = IrisZo(
            checker=checker,
            starting_ellipsoid=ellipsoid,
            domain=domain,
            options=iris_zo_options,
        )

        violation = np.max(region.A() @ center_q - region.b())

        if violation <= containment_tol:
            return region

        print(
            f"  grow_iris_region: seed not contained (violation={violation:.6f}), "
            f"retry {attempt + 1}/{max_retries} with radius={this_radius:.4g}"
        )

    raise RuntimeError(
        f"IrisZo could not produce a region containing its own seed after {max_retries} attempts. "
        f"(center={center_q} point may be too close to obstacle geometry"
        f"check actual clearance via checker.CalcRobotClearance before using it as a seed."
    )


# ------ 1(B) SAMPLE CONFIGURATION INSIDE IRIS REGION AND FIND CLOSEST CONFIGURATION TO TARGET --------

def sample_closest_in_region(
    region: HPolyhedron,
    plant: MultibodyPlant,
    context: Context,
    gripper_frame,
    target_xyz,                 # Desired end-effector position in world coordinates.
    generator: RandomGenerator, 
    seed_point,
    n_samples=200,
):
    """
    Sample configurations inside the current IRIS region using Drake's
    native hit-and-run sampler and return the sample whose gripper
    position is closest to target_xyz.

    Returns:
        best_q: Configuration with the shortest distance to target_xyz.
        best_dist: Corresponding Cartesian distance in metres.
    """

    best_q = None
    best_dist = np.inf

    prev = seed_point

    for _ in range(n_samples):

        prev = region.UniformSample(generator, prev)
        plant.SetPositions(context, prev)
        X = plant.CalcRelativeTransform(context, plant.world_frame(), gripper_frame)

        dist = np.linalg.norm(X.translation() - target_xyz)

        if dist < best_dist:
            best_dist = dist
            best_q = prev.copy()

    return best_q, best_dist

 
# ------------ 1(C) SOLVE IK TO TARGET WHILE HARDCONSTRAINING SOLUTION TO IRIS REGION ---------------

def solve_ik_in_region(
    plant: MultibodyPlant,
    context: Context,
    gripper_frame,
    world_frame,
    region: HPolyhedron,
    q_init,                 # Initial IK configuration guess.
    target_xyz,
    tol,                    # Cartesian position tolerance.
    checker,
):
    """
    Solve IK for target_xyz while hard-constraining the solution to
    remain inside the supplied IRIS region.

    The hard region constraint guarantees that a successful IK solution
    belongs to the current region, providing connectivity between
    consecutive regions.

    Returns:
        np.ndarray: Collision-free IK solution if successful.
        None: If IK fails or the resulting configuration is in collision.
    """

    ik = InverseKinematics(plant, context)

    ik.AddMinimumDistanceLowerBoundConstraint(0.001, influence_distance_offset=0.01)

    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        frameA=world_frame,
        p_AQ_lower=target_xyz - tol,
        p_AQ_upper=target_xyz + tol,
    )

    q = ik.q()

    prog = ik.get_mutable_prog()

    # Hard-constrain the solution to the IRIS region:
    # A @ q <= b
    A = region.A()
    b = region.b()

    prog.AddLinearConstraint(A, -np.inf * np.ones_like(b), b, q)

    prog.SetInitialGuess(q, q_init)

    result = Solve(prog)

    if not result.is_success():
        return None

    q_sol = result.GetSolution(q)

    if not checker.CheckConfigCollisionFree(q_sol):
        return None

    return q_sol


# ------------------- 1(D) SEQUENTIALLY GROW CONNECTED IRIS REGION TOWARDS TARGET -------------------

def grow_region_chain(
    checker,
    plant: MultibodyPlant,
    context: Context,
    gripper_frame,
    world_frame,
    q0,
    target_xyz,
    domain,
    iris_zo_options,
    final_tol=np.array([0.01, 0.01, 0.01]),
    max_hops=6,
    n_samples_per_hop=200,
    seed=0,
):
    """
    Sequentially grow an IRIS region chain toward the target.

    At each hop:

    1. Sample configurations inside the current region.
    2. Select the sampled configuration whose end-effector position
       is closest to the target.
    3. Attempt IK to the actual target while constraining the solution
       to the current region.
    4. If IK succeeds, grow a final region around the solution.
    5. If IK fails, grow a new IRIS region around the closest sample
       and continue.

    The hard region constraint in solve_ik_in_region() provides
    connectivity between consecutive regions.

    Returns:
        region_chain: List of HPolyhedron IRIS regions.
        q_chain: List of configurations associated with the regions.
        q_goal: Final IK configuration reaching the target.

    Raises:
        RuntimeError: If sampling fails or the target cannot be reached
                      within max_hops.
    """

    generator = RandomGenerator(seed)

    region_chain = [grow_iris_region(checker, q0, domain, iris_zo_options)]

    q_chain = [q0]

    for hop in range(max_hops):

        current_region = region_chain[-1]
        q_current = q_chain[-1]

        # --- 1. Find the sampled configuration closest to target ---
        
        q_next, dist = sample_closest_in_region(
            current_region,
            plant,
            context,
            gripper_frame,
            target_xyz,
            generator,
            seed_point=q_current,
            n_samples=n_samples_per_hop,
        )

        if q_next is None:
            raise RuntimeError(
                f"Hop {hop}: sampling failed inside region - "
                f"region may be degenerate."
            )

        print(
            f"Hop {hop}: closest sampled point is "
            f"{dist:.4f} m from target."
        )
        
        # --- 2. Try IK to actual target inside current region ---

        q_goal = solve_ik_in_region(
            plant,
            context,
            gripper_frame,
            world_frame,
            current_region,
            q_next,
            target_xyz,
            final_tol,
            checker,
        )

        # --- 3. Check for Target reached ---

        if q_goal is not None:

            goal_region = grow_iris_region(checker, q_goal, domain, iris_zo_options)

            # Defensive connectivity check.
            overlap = not current_region.Intersection(goal_region).IsEmpty()

            if not overlap:
                print(
                    f"WARNING: expected guaranteed overlap failed "
                    f"at hop {hop} - falling back to a bridging region."
                )

                q_mid = 0.5 * (q_next + q_goal)

                if not checker.CheckConfigCollisionFree(q_mid):
                    raise RuntimeError(
                        "Bridging midpoint is in collision - a plain "
                        "average of two valid configs is not guaranteed "
                        "collision-free. Need a local search near q_mid "
                        "instead of using it directly."
                    )

                bridge_region = grow_iris_region(checker, q_mid, domain, iris_zo_options)
                region_chain.append(bridge_region)

            region_chain.append(goal_region)
            q_chain.append(q_goal)

            print(
                f"Reached target at hop {hop}. "
                f"Chain length: {len(region_chain)} regions."
            )

            return region_chain, q_chain, q_goal

        # --- 4. Target not reachable yet => grow next region ---

        next_region = grow_iris_region(checker, q_next, domain, iris_zo_options)
        region_chain.append(next_region)
        q_chain.append(q_next)

    raise RuntimeError(
        f"Exceeded max_hops={max_hops} without reaching "
        f"a valid IK to the target."
    )


# ---------------------------------------------- PART 2 ----------------------------------------------
# ---------------------------- VISUALIZE COMPUTED IRIS REGION IN MESHCAT -----------------------------

def animate_iris(
    root_diagram,
    root_context,
    plant: MultibodyPlant,
    region: HPolyhedron,
    speed: float,
    meshcat,
    running_as_notebook=False,
):
    """
    A simple hit-and-run-style idea for visualizing the IRIS regions:
    1. Start at the center. Pick a random direction and run to the boundary.
    2. Pick a new random direction; project it onto the current boundary, and run along
       it. Repeat
    """

    plant_context = plant.GetMyContextFromRoot(root_context)

    q = region.ChebyshevCenter()

    plant.SetPositions(plant_context, q)

    root_diagram.ForcedPublish(root_context)

    print("Press the 'Stop Animation' button in Meshcat to continue.")
    meshcat.AddButton("Stop Animation", "Escape")

    rng = np.random.default_rng()
    nq = plant.num_positions()
    prog = MathematicalProgram()
    qvar = prog.NewContinuousVariables(nq, "q")
    prog.AddLinearConstraint(region.A(), 0 * region.b() - np.inf, region.b(), qvar)
    cost = prog.AddLinearCost(np.ones((nq, 1)), qvar)

    while meshcat.GetButtonClicks("Stop Animation") < 1:
        direction = rng.standard_normal(nq)
        cost.evaluator().UpdateCoefficients(direction)

        result = Solve(prog)
        assert result.is_success()

        q_next = result.GetSolution(qvar)

        for t in np.append(np.arange(0, 1, 0.05), 1):
            qs = t * q_next + (1 - t) * q
            plant.SetPositions(plant_context, qs)
            root_diagram.ForcedPublish(root_context)
            if running_as_notebook:
                time.sleep(0.05)

        q = q_next

        if not running_as_notebook:
            break

    meshcat.DeleteButton("Stop Animation")
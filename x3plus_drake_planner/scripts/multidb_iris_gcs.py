import time
import numpy as np

import open3d as o3d
from pathlib import Path

from pydrake.all import (
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    Rgba,
    RigidTransform,
    StartMeshcat,
    Sphere
)

from pydrake.planning import GcsTrajectoryOptimization, IrisZoOptions
from manipulation.meshcat_utils import PublishPositionTrajectory

from x3plus_drake_planner.geometry_utils.multidb import (
    prepare_complete_collision_geometry,
)
from x3plus_drake_planner.pointcloud_utils.preprocess import load_point_cloud

from x3plus_drake_planner.planning_utils.drake_utils import (
    build_x3plus,
    visualize_target_points,
    create_collision_checker,
    export_trajectory
)

from x3plus_drake_planner.planning_utils.iris_helper import (
    grow_region_chain,
    animate_iris,
)
from x3plus_drake_planner.planning_utils.iris_informed import (
    grow_region_chain_informed,
)


#  ---------------------- COMPUTE PRE-TASK TARGET POINT FROM THE SELECTED PLANE -----------------------

def compute_target_point(significant_planes, plant, plant_context, q0):
    """
    Compute a task-space target from the selected planar surface.

    Returns:
        centroid: plane centroid,
        target_offset_point: Pre-task position offset from the target surface.
    """

    target_cloud = np.asarray(significant_planes[2][0].points)

    centroid = np.mean(target_cloud, axis=0)

    _,_, vh = np.linalg.svd(target_cloud - centroid)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    gripper_frame = plant.GetFrameByName("grip_joint_parent")
    plant.SetPositions(plant_context, q0)

    X_WG_q0 = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
                                          plant.GetFrameByName("grip_joint_parent"))

    if np.dot(normal, X_WG_q0.translation() - centroid) < 0:
        normal = -normal

    offset_distance = -0.08
    target_point = centroid + offset_distance * normal
    vertical_offset = np.array([-0.1, 0.18, 0.0])
    target_offset_point = target_point + vertical_offset
    
    return centroid, target_offset_point


# --------------------- BUILD AND SOLVE GCS TRAJECTORY FROM THE IRIS REGION CHAIN ---------------------

def solve_gcs_trajectory(region_chain, q_chain, q_goal, q0, plant, robot_diagram, context):
    """
    Construct and solve a Grapth-of-Convex-Sets trajectory through the IRIS region chain.
    
    Returns:
        traj: Drake piecewise-polynomial trajectory.
    """
    # Initialize GCS with active trajectory dimensions 
    gcs = GcsTrajectoryOptimization(len(q0)) 

    # Store all IRIS region as order-2 Bezier regions
    region_nodes = []

    for region in region_chain:
        node = gcs.AddRegions([region], order=2)
        region_nodes.append(node)

    # Start and Goal points (order-0) regions
    source_node = gcs.AddRegions([Point(q_chain[0])], order=0)
    target_node = gcs.AddRegions([Point(q_goal)], order=0)

    # Connect: source_node, region_nodes and target_node
    gcs.AddEdges(source_node, region_nodes[0]) # start -> first IRIS region

    for i in range(len(region_nodes) - 1):
        gcs.AddEdges(region_nodes[i], region_nodes[i + 1]) # IRIS region -> IRIS region

    gcs.AddEdges(region_nodes[-1], target_node) # last IRIS region -> goal

    # Trajectory costs
    weight = 5.0 
    num_positions = plant.num_positions()
    weight_matrix = (weight * np.eye(num_positions))

    gcs.AddPathEnergyCost(weight_matrix)
    gcs.AddTimeCost()
    gcs.AddVelocityBounds(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())

    # GCS solver options
    options = GraphOfConvexSetsOptions()

    options.preprocessing = True
    options.max_rounded_paths = 10
    options.rounding_seed = 1235

    print("\nSolving GCS Trajectory Path")

    traj, result = gcs.SolvePath(source_node, target_node, options)

    if not result.is_success():
        print("FAILURE: No feasible GCS path found.")
        print(f"Result: {result.get_solution_result()}")
        raise RuntimeError("GCS trajectory optimization failed.")

    print("SUCCESS: GCS path found.")

    # Visualize trajectory in Meshcat
    visualizer = robot_diagram.GetSubsystemByName("meshcat_visualizer(illustration)")
    PublishPositionTrajectory(traj, context, plant, visualizer)
    visualizer.ForcedPublish(visualizer.GetMyContextFromRoot(context))

    return traj


# ----------------------------------------------- MAIN ------------------------------------------------

def main():

    # ---------- Part 1: Load perception data -----------------------------------------------

    SRC_DIR = Path(__file__).resolve().parents[2]
    PERCEPTION_DIR = SRC_DIR / "x3plus_perception_utils"
    
    captures_dir = (
        PERCEPTION_DIR
        / "x3plus_perception_utils"
        / "rgbd_pose_dataset"
        / "chrissek_tsdf"
    )

    npz_path = captures_dir / "00010.npz"
    pointcloud = load_point_cloud(npz_path, max_depth=0.6, voxel_size=0.005)

    # ---------- Part 2: Generate collision geometry ----------------------------------------

    print("\n---------- Generating collision geometry ----------")

    obstacle_verts_list, significant_planes = prepare_complete_collision_geometry(pointcloud)
    print(f"\nGenerated {len(obstacle_verts_list)} collision obstacles")

    # ---------- Part 3: Build Drake X3Plus scene -------------------------------------------

    meshcat = StartMeshcat()

    robot_diagram, plant = build_x3plus(obstacle_verts_list, meshcat)
    context = robot_diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # ---------- Part 4: Set initial arm configuration --------------------------------------

    arm_joints = [
        plant.GetJointByName("arm_joint1"),
        plant.GetJointByName("arm_joint2"),
        plant.GetJointByName("arm_joint3"),
        plant.GetJointByName("arm_joint4"),
        plant.GetJointByName("arm_joint5"),
    ]

    q0 = np.array([0.0, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0.0])

    plant.SetPositions(plant_context, q0)
    robot_diagram.ForcedPublish(context)

    print("Inital Arm Configuration (q0): ", q0)

    # ---------- Part 5: Select and visualize target point -----------------------------------

    centroid, target_offset_point = compute_target_point(significant_planes,
                                                         plant,
                                                         plant_context,
                                                         q0)
    
    visualize_target_points(meshcat, centroid, target_offset_point)

    # ---------- Part 6: Intialize Drake collision checker ------------------------------------

    checker = create_collision_checker(robot_diagram, plant)
    print("q0 collision-free check: ", checker.CheckConfigCollisionFree(q0))

    # ---------- Part 7: Set up IRIS parameters -----------------------------------------------

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("grip_joint_parent")
    joint1_index = plant.GetJointByName("arm_joint1").position_start()

    padding = np.deg2rad(35) # manually restricting the region

    min_limits = plant.GetPositionLowerLimits().copy()
    max_limits = plant.GetPositionUpperLimits().copy()

    # Restricting the arm's base joint to force the region growing forward.
    min_limits[joint1_index] = max(min_limits[joint1_index], -padding)
    max_limits[joint1_index] = min(max_limits[joint1_index], padding)

    domain = HPolyhedron.MakeBox(min_limits, max_limits)

    # IRIS Options
    iris_zo_options = IrisZoOptions() 

    iris_zo_options.sampled_iris_options.epsilon = 0.01
    iris_zo_options.sampled_iris_options.delta = 0.01

    # ---------- Part 8: Sequential grow IRIS toward 'target_offset_point' --------------------

    target_offset_point = [0.67860311, 0.05494937, 0.38887407]
    print(f"\nSequentially growing IRIS towards {target_offset_point}")

    t0 = time.perf_counter()

    region_chain, q_chain, q_goal = grow_region_chain_informed(checker,
                                                      plant,
                                                      plant_context,
                                                      gripper_frame,
                                                      world_frame,
                                                      q0,
                                                      target_offset_point,
                                                      domain,
                                                      iris_zo_options)

    print(f"IRIS region generation took: {time.perf_counter() - t0:.4f} s")
    print(f"Generated {len(region_chain)} IRIS regions.")
    print("q_goal: ", q_goal)
    print("q_goal collision-free check: ", checker.CheckConfigCollisionFree(q_goal))

    # ---------- part 9: GCS trajectory optimization ------------------------------------------

    traj = solve_gcs_trajectory(region_chain,
                                q_chain,
                                q_goal,
                                q0,
                                plant,
                                robot_diagram,
                                context)

    print("Trajectory execution time: ", traj.end_time())

    # ---------- Part 10: Sample and export GCS trajectory ------------------------------------

    export_trajectory(traj, dt=0.02)

    print("\nSequence completed.")


if __name__ == "__main__":
    main()
    

 
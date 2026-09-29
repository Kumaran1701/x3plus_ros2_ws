import time
import numpy as np
import cv2
import open3d as o3d

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

from tsdf_coacd import prepare_tsdf_coacd_collision_geometry

from drake_utilis import (
    build_x3plus,
    visualize_target_points,
    create_collision_checker,
)

from iris_helper import grow_region_chain, animate_iris


# ---------------------------- LOAD NPZ AND CREATE WORLD-FRAME POINTCLOUD -----------------------------

def load_point_cloud(npz_path, max_depth=0.6, voxel_size=0.005):

    data = np.load(npz_path)

    depth = data["depth"].astype(np.float32) / 1000.0 # mm to m
    rgb = data["rgb"]
    pose = data["pose"]

    fx, fy, cx, cy, _, _ = data["depth_intrinsics"]

    H, W = depth.shape # depth cam resolution

    rgb_resized = cv2.resize(rgb, (W, H)) # resize rgb cam to depth cam resolution 
    rgb_flat = rgb_resized.reshape(-1, 3)

    # Generate pointcloud in depth cam frame
    u, v = np.meshgrid(np.arange(W), np.arange(H)) 
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    valid = Z.reshape(-1) > 0
    close = Z.reshape(-1) < 0.6 # keep points within 0.6 m
    valid = valid & close

    points_cam = points_cam[valid]
    colors = rgb_flat[valid] / 255.0

    # Transform to world frame
    ones = np.ones((points_cam.shape[0], 1)) 
    points_h = np.hstack((points_cam, ones))
    points_world = (pose @ points_h.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)

    print(f"\nLoaded downsampled pointcloud with {len(pcd.points)} points")

    return pcd


#  ---------------------- COMPUTE PRE-TASK TARGET POINT FROM THE SELECTED PLANE -----------------------

def compute_target_point(significant_planes, plant, plant_context, q0):

    target_cloud = np.asarray(significant_planes[2][0].points)

    centroid = np.mean(target_cloud, axis=0)

    _,_, vh = np.linalg.svd(target_cloud - centroid)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    gripper_frame = plant.GetFrameByName("grip_joint_parent")
    plant.SetPosition(plant_context, q0)

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


# -------------------------- EXPORT GCS TRAJECTORY TO A FIXED-RATE NPZ FILE ---------------------------

def export_trajectory(traj, filename="trajectory_export.npz", dt=0.02):

    times = np.arange(0.0, traj.end_time(), dt)
    q_samples = np.array([traj.value(t).ravel() for t in times])

    np.savez(filename, times=times, q=q_samples)
    print(f"Saved trajectory to {filename}")


# ----------------------------------------------- MAIN ------------------------------------------------

def main():

    # ---------- Part 1: Load perception data ---------------------------------------------------------

    #npz_path = "sep3_claude/00000.npz"
    #pointcloud = load_point_cloud(npz_path, max_depth=0.6, voxel_size=0.005)

    # ---------- Part 2: Generate collision geometry --------------------------------------------------

    print("\n---------- Generating collision geometry ----------")

    obstacle_verts_list = prepare_tsdf_coacd_collision_geometry(captures_dir="/home/kumaran/Mobile_Manipulator/x3plus_drake/sep3_claude")

    print(f"\nGenerated {len(obstacle_verts_list)} collision obstacles")

    # ---------- Part 3: Build Drake X3Plus scene -----------------------------------------------------

    meshcat = StartMeshcat()

    robot_diagram, plant = build_x3plus(obstacle_verts_list, meshcat)
    context = robot_diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # ---------- Part 4: Set initial arm configuration ------------------------------------------------

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
'''
    # ---------- Part 5: Select and visualize target point ---------------------------------------------

    centroid, target_offset_point = compute_target_point(significant_planes,
                                                         plant,
                                                         plant_context,
                                                         q0)
    
    visualize_target_points(meshcat, centroid, target_offset_point)

    # ---------- Part 6: Intialize Drake collision checker --------------------------------------------- 

    checker = create_collision_checker(robot_diagram, plant)

    print("q0 collision-free check: ", checker.CheckConfigCollisionFree(q0))

    # ---------- Part 7: Set up IRIS parameters ---------------------------------------------------------

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

    # ---------- Part 8: Sequential grow IRIS toward 'target_offset_point' ------------------------------

    print(f"\nSequentially growing IRIS towards {target_offset_point}")

    t0 = time.perf_counter()

    region_chain, q_chain, q_goal = grow_region_chain(checker,
                                                      plant,
                                                      plant_context,
                                                      gripper_frame,
                                                      world_frame,
                                                      q0,
                                                      target_offset_point,
                                                      domain,
                                                      iris_zo_options)

    print(f"IRIS region generation took: {time.perf_counter - t0:.4f} s")
    print(f"Generated {len(region_chain)} IRIS regions.")
    print("q_goal: ", q_goal)
    print("q_goal collision-free check: ", checker.CheckConfigCollisionFree(q_goal))

    # ---------- part 9: GCS trajectory optimization ----------------------------------------------------

    traj = solve_gcs_trajectory(region_chain,
                                q_chain,
                                q_goal,
                                q0,
                                plant,
                                robot_diagram,
                                context)

    print("Trajectory execution time: ", traj.end_time())

    # ---------- Part 10: Sample and export GCS trajectory ----------------------------------------------

    export_trajectory(traj, filename="trajectory_export.npz", dt=0.02)

    print("\nSequence completed.")
'''

if __name__ == "__main__":
    main()
    

 
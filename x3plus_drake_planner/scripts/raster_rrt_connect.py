import time
from pathlib import Path

import numpy as np
from pydrake.all import HPolyhedron, RigidTransform, Sphere, StartMeshcat
from manipulation.meshcat_utils import PublishPositionTrajectory

from x3plus_drake_planner.geometry_utils.plane_rasterization import (
    process_complete_workspace_rasterization,
)
from x3plus_drake_planner.geometry_utils.plane_segmentation import segment_planes
from x3plus_drake_planner.planning_utils.drake_utils import (
    build_x3plus,
    create_collision_checker,
    export_trajectory,
)
from x3plus_drake_planner.planning_utils.rrt_connect_planner import (
    find_goal_config,
    plan_rrt_connect,
)
from x3plus_drake_planner.pointcloud_utils.preprocess import load_fused_point_cloud


def main():

    workspace_dir = Path(__file__).resolve().parents[2]
    perception_dir = workspace_dir / "x3plus_perception_utils"
    captures_dir = (
        perception_dir
        / "x3plus_perception_utils"
        / "rgbd_pose_dataset"
        / "chrissek_tsdf"
    )

    point_cloud = load_fused_point_cloud(captures_dir, start=7, end=7, max_depth=0.6, voxel_size=0.005)

    # ---------- Part 2: Generate collision geometry ----------------------------------------
    
    print("\n---------- Segmenting planes for raster collision geometry ----------")
    significant_planes, outliers = segment_planes(point_cloud, 
                                                  distance_threshold=0.005,
                                                  ransac_n=3,
                                                  num_iterations=1000,
                                                  min_points=1000)
    hull_meshes, noise = process_complete_workspace_rasterization(significant_planes,
                                                                  outliers=outliers)

    obstacle_vertices = [np.asarray(mesh.vertices) for mesh in hull_meshes if len(mesh.vertices) >= 4]

    if not obstacle_vertices:
        raise RuntimeError("Rasterization produced no usable collision hulls")
    
    print(f"Generated {len(obstacle_vertices)} collision hulls")
    print(f"Total noise points not included in collision geometry: {len(noise.points)}")

    # ---------- Part 3: Build Drake X3Plus scene -------------------------------------------

    meshcat = StartMeshcat()

    robot_diagram, plant = build_x3plus(obstacle_vertices, meshcat)
    context = robot_diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # ---------- Part 4: Set initial arm configuration --------------------------------------

    q0 = np.array([0.0, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0.0])

    plant.SetPositions(plant_context, q0)
    robot_diagram.ForcedPublish(context)

    print("Initial arm configuration:", q0)

    # ---------- Part 5: Visualize target point -----------------------------------

    target_offset_point = np.array([0.67860311, 0.05494937, 0.38887407], dtype=float)
    
    meshcat.SetObject(path="target_offset_point",
                      shape=Sphere(0.01))
    meshcat.SetTransform("target_offset_point",
                         RigidTransform(target_offset_point))
    
    print(f"Planning to manually defined target: {target_offset_point}")

    # ---------- Part 6: Intialize Drake collision checker ------------------------------------

    checker = create_collision_checker(robot_diagram, plant)
    print("q0 collision-free check: ", checker.CheckConfigCollisionFree(q0))

    # ---------- Part 7: Initialize Required Frames and Joint Limits --------------------------

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("grip_joint_parent")
    joint1_index = plant.GetJointByName("arm_joint1").position_start()

    q_low = plant.GetPositionLowerLimits().copy()
    q_high = plant.GetPositionUpperLimits().copy()

    base_joint_padding = np.deg2rad(35)

    q_low[joint1_index] = max(q_low[joint1_index], -base_joint_padding)
    q_high[joint1_index] = min(q_high[joint1_index], base_joint_padding)
    
    domain = HPolyhedron.MakeBox(q_low, q_high)

    # ---------- Part 8: Find Collision-free Goal Configuration -------------------------------

    print("\n---------- Finding collision-free IK goal ----------")
    q_goal = find_goal_config(plant,
                              plant_context,
                              gripper_frame,
                              world_frame,
                              domain,
                              checker,
                              target_offset_point,
                              q0,
                              q_low,
                              q_high,
                              tol=np.array([0.01, 0.005, 0.01]),
                              n_random=30,
                              seed=0)
    
    if q_goal is None:
        raise RuntimeError("No collision-free IK solution found for the manual target")
    print("RRT-Connect goal configuration:", q_goal)


    # ---------- Part 9: RRTConnect: q0 <--> q_goal --------------------------------------------

    print("\n---------- Planning with RRT-Connect ----------")
    start_time = time.perf_counter()
    trajectory, info, path, raw_path = plan_rrt_connect(checker,
                                                        plant,
                                                        q0,
                                                        q_goal,
                                                        q_low,
                                                        q_high,
                                                        step=0.25,
                                                        max_iters=5000,
                                                        shortcut_iters=100,
                                                        seed=0)
    
    print(f"RRT planning pipeline completed in {time.perf_counter() - start_time:.3f}s")
    print(f"RRT-Connect: {info['t_rrt']:.3f}s / {info['iterations']} iterations; "
          f"shortcut: {info['t_shortcut']:.3f}s; "
          f"trajectory: {info['t_traj']:.3f}s"
    )
    print(f"Waypoints {len(raw_path)} -> {len(path)}; "
          f"joint-space length {info['length_raw']:.2f} -> "
          f"{info['length_short']:.2f} rad; "
          f"duration {info['duration']:.2f}s"
    )

    visualizer = robot_diagram.GetSubsystemByName("meshcat_visualizer(illustration)")
    PublishPositionTrajectory(trajectory, context, plant, visualizer)
    visualizer.ForcedPublish(visualizer.GetMyContextFromRoot(context))

    trajectory_path = (
        workspace_dir
        / "x3plus_motion_planner"
        / "trajectory"
        / "trajectory_export.npz"
    )
    export_trajectory(trajectory, filename=trajectory_path, dt=0.02)
    print("\nRaster RRT-Connect planning complete.")


if __name__ == "__main__":
    main()

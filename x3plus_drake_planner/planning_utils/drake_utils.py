
"""Drake scene construction, collision checking, and trajectory utilities."""

import time
import numpy as np
from pathlib import Path

from pydrake.all import (
    AddDefaultVisualization,
    Rgba,
    RigidTransform,
    Sphere,
)
from pydrake.geometry import (
    CollisionFilterDeclaration, 
    GeometrySet, 
    Convex, 
    ProximityProperties
)
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker

from manipulation.utils import ConfigureParser
from pydrake.geometry import ProximityProperties

import tempfile


# ----------------------------------------------- Helper -----------------------------------------------
# ------------------------- Convert 3D Obstacle Vertices to Drake Convex Shape -------------------------

def make_convex_shape(verts):
    """
    Convert obstacle vertices into a Drake Convex collision shape.
    """
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False, mode="w") as f:
        path = f.name
        for pt in verts:
            f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
        # qhull-free: Convex() takes the point cloud and hulls it internally
    return Convex(path)


# -------------------------- EXPORT TRAJECTORY TO A FIXED-RATE NPZ FILE ---------------------------

def export_trajectory(traj, filename=None, dt=0.02):
    """
    Sample and export a fixed rate Drake trajectory to a NPZ file.
    """

    if filename is None:
        filename = (
            Path(__file__).resolve().parents[2]
            / "x3plus_motion_planner"
            / "trajectory"
            / "trajectory_export.npz"
        )
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    end_time = traj.end_time()
    times = np.arange(0.0, end_time, dt)
    if times.size == 0 or times[-1] < end_time:
        times = np.append(times, end_time)
    q_samples = np.array([traj.value(t).ravel() for t in times])

    np.savez(filename, times=times, q=q_samples)
    print(f"Saved trajectory to {filename}")


# ----------------------------------------------- PART 1 ------------------------------------------------
# -------------- BUILD X3PLUS + GRIPPER WITH COLLISION GEOMETRY AND SELF-COLLISION FILTERS --------------

def build_x3plus(obstacle_verts_list, meshcat):
    """
    Build the Drake X3Plus + Gripper + perceived obstacle geometry into the Plant.
    
    Adds:
        - X3Plus robot with gripper.
        - Perceived convex obstacles from point cloud.
        - Virtual joint actuators for the arm joints (for simulation).
        - Self-collision filters for robot links.
        - Meshcat visualization.

    Returns:
        robot_diagram: Drake RobotDiagram containing the complete scene.
        plane: MultibodyPlant associated with the RobotDiagram.
    """

    SRC_DIR = Path(__file__).resolve().parents[2]
    DESCRIPTION_DIR = SRC_DIR / "x3plus_description"

    # ---------- 1. Robot Diagram and Model Parsing ----------
    
    rdb = RobotDiagramBuilder(time_step=0.001)
    builder = rdb.builder()
    plant = rdb.plant()
    scene_graph = rdb.scene_graph()

    t_parse = time.perf_counter()
    parser = rdb.parser()  # Get the unified parser attached to the RobotDiagramBuilder
    parser.package_map().Add("meshes", str(DESCRIPTION_DIR / "meshes"))
    x3plus = parser.AddModels(str(DESCRIPTION_DIR / "urdf" / "drake_x3plus_fixed.urdf"))[0]
    gripper = parser.AddModels(str(DESCRIPTION_DIR / "urdf" / "drake_x3plus_gripper.urdf"))[0]
    ConfigureParser(parser)
    print(f"Model parsing time (URDFs): {time.perf_counter() - t_parse:.4f} seconds")
    
    # ---------- 2. Weld Frames ----------

    # Weld robot to world 
    t_weld = time.perf_counter()
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base_footprint", x3plus),
                     RigidTransform([0.40, 0.1, 0.0]))
    # Weld gripper to robot
    plant.WeldFrames(plant.GetFrameByName('arm_link5', x3plus),
                     plant.GetFrameByName('gripper_base', gripper),
                     RigidTransform.Identity())
    
    print(f"Welding frames time: {time.perf_counter() - t_weld:.4f} seconds")

    # ---------- 3. Perceived Convex Obstacle Geometry ----------

    for idx, verts in enumerate(obstacle_verts_list):
        shape_collision = make_convex_shape(verts)
        shape_visual = make_convex_shape(verts)
    
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), shape_collision,
                                        f"obstacle_{idx}_collision", ProximityProperties())
        plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(), shape_visual,
                                     f"obstacle_{idx}_visual", np.array([0.9, 0.1, 0.1, 0.5]))
        
    # ---------- 4. Add Joint Actuators (Virtual) ----------------------------
    # --- For Joint Space Control or Operational Space Control----------------

    # Note: X3Plus is only Position-Controlled so the output of JSC or OSC
    #       can only be tested in simualtion and not on the hardware.

    t_finalize = time.perf_counter()
    arm_joints = [plant.GetJointByName("arm_joint1"),
                  plant.GetJointByName("arm_joint2"),
                  plant.GetJointByName("arm_joint3"),
                  plant.GetJointByName("arm_joint4"),
                  plant.GetJointByName("arm_joint5")]
    
    for joint in arm_joints:
        plant.AddJointActuator(joint.name(), joint, effort_limit=100.0)
    
    # Finalize Plant - No more changes to the MultiBodyPlant is possible after this.
    plant.Finalize()
    print(f"Plant finalization time: {time.perf_counter() - t_finalize:.4f} seconds")
    
    # ---------- 5. Filter Out Self-Collision of Robot Links ----------

    t_filter1 = time.perf_counter()

    inspector = scene_graph.model_inspector()
    filter_manager = scene_graph.collision_filter_manager()

    base_frame  = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("base_link").index())
    link1_frame = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("arm_link1").index())
    link2_frame = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("arm_link2").index())
    link3_frame = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("arm_link3").index())
    link4_frame = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("arm_link4").index()) 
    link5_frame = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("arm_link5").index()) 
    mono_frame  = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("mono_link").index())
    llink1_frame = plant.GetBodyFrameIdOrThrow(plant.GetBodyByName("llink1").index())
    
    base_geoms  = GeometrySet(inspector.GetGeometries(base_frame))
    link1_geoms = GeometrySet(inspector.GetGeometries(link1_frame))
    link2_geoms = GeometrySet(inspector.GetGeometries(link2_frame))
    link3_geoms = GeometrySet(inspector.GetGeometries(link3_frame))
    link4_geoms = GeometrySet(inspector.GetGeometries(link4_frame)) 
    link5_geoms = GeometrySet(inspector.GetGeometries(link5_frame)) 
    mono_geoms  = GeometrySet(inspector.GetGeometries(mono_frame))
    llink1_geoms = GeometrySet(inspector.GetGeometries(llink1_frame)) 

    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(base_geoms, link1_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link1_geoms, link2_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link2_geoms, link3_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link3_geoms, link4_geoms)) 
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link4_geoms, link5_geoms)) 
    
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(base_geoms, link2_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(base_geoms, link3_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link1_geoms, link3_geoms)) 
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link2_geoms, link4_geoms)) 
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link3_geoms, link5_geoms)) 

    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(base_geoms, link4_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(base_geoms, mono_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link5_geoms, mono_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(base_geoms, link5_geoms))
    filter_manager.Apply(CollisionFilterDeclaration().ExcludeBetween(link4_geoms, llink1_geoms))
    
    print(f"Arm/Base collision filtering time: {time.perf_counter() - t_filter1:.4f} seconds")

    # ---------- 6. Visualization Setup ----------------

    t_viz = time.perf_counter()
    AddDefaultVisualization(builder, meshcat)
    print(f"Meshcat visualization setup time: {time.perf_counter() - t_viz:.4f} seconds")

    # ---------- 7. Finish RobotDiagram Build ----------
    t_build = time.perf_counter()
    robot_diagram = rdb.Build()
    print(f"RobotDiagram build time: {time.perf_counter() - t_build:.4f} seconds")

    return robot_diagram, plant


# ----------------------------------------------- PART 2 ------------------------------------------------
# ------- VISUAL MARKERS FOR GRIPPER's POSITION AT TARGET AND PRE-TARGET LOCATIONS IN TASK-SPACE --------

def visualize_target_points(meshcat, centroid, target_offset_point_shifted):
    """
    Visualize target and pre-target positions in Meshcat.
    """
    # Target location 
    meshcat.SetObject(path="target_centroid",
                      shape=Sphere(0.01),
                      rgba=Rgba(0.1, 0.9, 0.1, 1.0))

    meshcat.SetTransform("target_centroid",
                         RigidTransform(centroid))

    # Pre-Target or Offset location from target 
    meshcat.SetObject(path="target_offset_point",
                      shape=Sphere(0.01),
                      rgba=Rgba(0.1, 0.1, 0.9, 1.0))

    meshcat.SetTransform("target_offset_point",
                         RigidTransform(target_offset_point_shifted))

    print(f"Target centroid: {centroid}")
    print(f"Target offset point: {target_offset_point_shifted}")


# ----------------------------------------------- PART 3 ------------------------------------------------
# --------------- DRAKE CLASS FOR COLLISION AND DISTANCE QUERIES b/w ROBOT AND ENVIRONMENT ---------------

def create_collision_checker(robot_diagram, plant):
    """
    Create a SceneGraphCollisionChecker for robot-environment collision and distance queries.
    Returns:
        checker: Instance of Drake's SceneGraphCollisionChecker.
    """

    x3plus_model_instance = (plant.GetModelInstanceByName("x3plus"))

    gripper_model_instance = (plant.GetModelInstanceByName("x3plus_gripper"))

    checker = SceneGraphCollisionChecker(model=robot_diagram,
                                         robot_model_instances=[x3plus_model_instance,
                                                                gripper_model_instance],
                                         edge_step_size=0.01)

    return checker
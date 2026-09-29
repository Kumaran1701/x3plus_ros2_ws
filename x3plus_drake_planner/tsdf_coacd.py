import multiprocessing as mp
from pathlib import Path
import trimesh

import coacd    
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


def load_capture(npz_path, 
                 depth_key="depth", 
                 depth_intrinsics_key="depth_intrinsics",
                 pose_key="pose",
                 depth_scale=1000.0,
                 use_color=False
                 ):
    
    data = np.load(npz_path, allow_pickle=True)

    depth_raw = data[depth_key].astype(np.float32)
    depth_m = depth_raw / depth_scale

    fx, fy, cx, cy, width, height = data[depth_intrinsics_key]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width),
                                                  int(height),
                                                  float(fx),
                                                  float(fy),
                                                  float(cx),
                                                  float(cy)
                                                  )

    rgb = None

    if use_color and "rgb" in data:
        rgb_bgr = data["rgb"]
        rgb = np.ascontiguousarray(rgb_bgr[..., ::-1])

        if rgb.shape[:2] != depth_m.shape:
            raise ValueError(f"RGB shape {rgb.shape[:2]} does not match depth shape {depth_m.shape}.")

    X_World_Cam = data[pose_key].astype(np.float64)

    return depth_m, rgb, intrinsic, X_World_Cam
    

def run_tsdf_fusion(captures_dir,
                    voxel_length=0.004,
                    sdf_trunc=None,
                    depth_trunc=2.0,
                    use_color=False,
                    ):
     
    captures_dir = Path(captures_dir)
    npz_files = sorted(captures_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} captures in {captures_dir}.")

    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {captures_dir}."
        )

    if sdf_trunc is None:
        sdf_trunc = voxel_length * 5.0

    if use_color:
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    else:
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.NoColor

    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length,
                                                          sdf_trunc=sdf_trunc,
                                                          color_type=color_type)

    integrated = 0

    for index, npz_path in enumerate(npz_files):
        try:
            depth_m, rgb, intrinsic, X_World_Cam = load_capture(npz_path, use_color=use_color) # X_World_Cam: Pose of Cam wrt World
        except Exception as exc:
            print(f"[{index}] skipping {npz_path.name}: {exc}")
            continue

        depth_image = o3d.geometry.Image(depth_m)

        if use_color and rgb is not None:
            color_image = o3d.geometry.Image(rgb)
        else:
            color_image = o3d.geometry.Image(np.full((*depth_m.shape, 3), 128, dtype=np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image,
                                                                   depth_image,
                                                                   depth_scale=1.0,
                                                                   depth_trunc=depth_trunc,
                                                                   convert_rgb_to_intensity=False,
                                                                   )
        # TSDF expects world -> cam
        X_Cam_World = np.linalg.inv(X_World_Cam) # X_Cam_World: Pose of World wrt Cam

        volume.integrate(rgbd, intrinsic, X_Cam_World)
        integrated += 1

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"Fused mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
    
    return mesh
    

def connected_component_split(mesh, 
                              min_bbox_diag=0.02
                              ):

    trimesh_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                   faces=np.asarray(mesh.triangles),
                                   process=False,
                                   )

    components = trimesh_mesh.split(only_watertight=False)

    clusters = []

    for component in components:
        bbox_diag = np.linalg.norm(component.bounds[-1] - component.bounds[0])

        if bbox_diag >= min_bbox_diag:
            clusters.append(component.vertices)

    print(f"Connected components: {len(components)} total, "
          f"{len(clusters)} kept (bbox diagonal >= {min_bbox_diag} m)")

    return clusters


def dbscan_split(mesh_vertices_list,
                 eps=0.01,
                 min_samples=20,
                 min_bbox_diag=0.02,
                 ):
    
    if mesh_vertices_list:
        points = np.vstack(mesh_vertices_list)
    else:
        points = np.empty((0, 3))

    print(f"Falling back to DBSCAN, eps={eps}, min_samples={min_samples}")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    cluster_labels = sorted(set(labels) - {-1})

    print(f"DBSCAN found {len(cluster_labels)} clusters ({np.sum(labels == -1)} noise points).")

    clusters = []

    for label in cluster_labels:
        points_in_cluster = points[labels == label]

        bbox_diag = np.linalg.norm(points_in_cluster.max(axis=0) - points_in_cluster.min(axis=0))

        if bbox_diag >= min_bbox_diag:
            clusters.append(points_in_cluster)

    return clusters


def segment_mesh(mesh,
                 min_bbox_diag=0.02,
                 min_expected_components=2,
                 dbscan_eps=0.01,
                 dbscan_min_samples=20,
                 ):
    
    clusters = connected_component_split(mesh, min_bbox_diag=min_bbox_diag)

    if len(clusters) < min_expected_components:
        print(f"Only {len(clusters)} compoents. Trying DBScan fallback.")

        if clusters:
            input_points = clusters
        else:
            input_points = [np.asarray(mesh.vertices)]

        clusters = dbscan_split(mesh_vertices_list=input_points,
                                eps=dbscan_eps,
                                min_samples=dbscan_min_samples,
                                min_bbox_diag=min_bbox_diag,
                                )

    if not clusters:
        raise RuntimeError("No usable clusters after mesh segmentation.")

    return clusters


def estimate_alpha(point_cloud,
                   neighbor_factor=3.0,
                   ):
    
    distances = point_cloud.compute_nearest_neighbor_distance()

    if len(distances) == 0:
        return None

    return neighbor_factor * float(np.mean(distances))


def compute_hull_volume(points):

    try:
        return ConvexHull(points).volume
    except Exception:
        return None


def compute_alpha_mesh(point_cloud,
                       alpha,
                       ):
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        if len(mesh.triangles) >= 4:
            return mesh

    except Exception:
        pass

    return None


def alpha_mesh_volume(mesh):

    if mesh is None:
        return None

    try:
        if mesh.is_watertight():
            return mesh.get_volume()
    except Exception:
        pass

    return None


# ============================================================
# ------- COACD -------
# ============================================================

def process_cluster(points,
                    alpha_neighbor_factor=3.0,
                    concavity_volume_ratio=1.15,
                    coacd_threshold=0.10,
                    max_convex_hulls=16,
                    resolution=20000,
                    mcts_nodes=20,
                    mcts_iterations=60,
                    mcts_max_depth=3,
                    ):
    """
    Process one segmented point cluster.

    Near-convex clusters are represented directly by their convex hull. Concave clusters are decomposed using CoACD.

    Returns:
        List of Nx3 vertex arrays.
    """

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    hull_volume = compute_hull_volume(points)

    if hull_volume is None or hull_volume <= 0:

        hull_mesh, _ = point_cloud.compute_convex_hull()

        return [np.asarray(hull_mesh.vertices)]

    alpha = estimate_alpha(point_cloud, neighbor_factor=alpha_neighbor_factor)

    if alpha is not None:
        alpha_mesh = compute_alpha_mesh(point_cloud, alpha)
    else:
        alpha_mesh = None

    alpha_volume = alpha_mesh_volume(alpha_mesh)

    is_concave = True

    if alpha_volume is not None and alpha_volume > 0:
        is_concave = (hull_volume / alpha_volume > concavity_volume_ratio)

    # Near-convex cluster: no CoACD necessary.
    if not is_concave:

        hull_mesh, _ = point_cloud.compute_convex_hull()

        return [np.asarray(hull_mesh.vertices)]

    # Use alpha-shape mesh as input to CoACD when available.
    if alpha_mesh is not None:
        vertices = np.asarray(alpha_mesh.vertices)
        faces = np.asarray(alpha_mesh.triangles)

    else:
        hull_mesh, _ = point_cloud.compute_convex_hull()

        vertices = np.asarray(hull_mesh.vertices)
        faces = np.asarray(hull_mesh.triangles)

    try:
        coacd_mesh = coacd.Mesh(vertices.astype(np.float64), 
                                faces.astype(np.int32)
                                )

        convex_parts = coacd.run_coacd(coacd_mesh,
                                       threshold=coacd_threshold,
                                       max_convex_hull=max_convex_hulls,
                                       preprocess_mode="auto",
                                       resolution=resolution,
                                       mcts_nodes=mcts_nodes,
                                       mcts_iterations=mcts_iterations,
                                       mcts_max_depth=mcts_max_depth,
                                       )

        return [np.asarray(vertices) for vertices, _ in convex_parts]

    except Exception:

        hull_mesh, _ = point_cloud.compute_convex_hull()
        
        return [np.asarray(hull_mesh.vertices)]


def run_coacd_on_clusters(clusters, 
                          n_workers=None,
                          ):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Processing {len(clusters)} clusters across {n_workers} worker(s).")

    # CoACD/Open3D require spawn here.
    context = mp.get_context("spawn")

    with context.Pool(n_workers) as pool:
        results = pool.map(process_cluster, clusters)

    obstacle_verts_list = []

    skipped = 0
    decomposed = 0

    for parts in results:

        if len(parts) == 1:
            skipped += 1
        else:
            decomposed += 1

        obstacle_verts_list.extend(parts)

    print(f"{skipped} cluster(s) kept as convex hulls.")
    print(f"{decomposed} cluster(s) decomposed by CoACD.")

    print(f"Total obstacle parts: {len(obstacle_verts_list)}")

    return obstacle_verts_list


def prepare_tsdf_coacd_collision_geometry(captures_dir, voxel_length=0.004, depth_trunc=2.0):
    """
    Build collision geometry directly from the captured depth data.

    Returns:
        obstacle_verts_list: list of Nx3 vertex arrays representing convex obstacle parts.
    """

    mesh = run_tsdf_fusion(captures_dir=captures_dir, voxel_length=voxel_length, depth_trunc=depth_trunc)

    clusters = segment_mesh(mesh)

    obstacle_verts_list = run_coacd_on_clusters(clusters)

    return obstacle_verts_list
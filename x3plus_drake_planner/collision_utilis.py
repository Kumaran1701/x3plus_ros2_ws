import time
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# --------------------------------------------- PART 1 ------------------------------------------------
# ----------------------------------- RANSAC PLANE SEGMENTATION ---------------------------------------

def segment_planes(point_cloud, distance_threshold=0.005, ransac_n=3, num_iterations=1000, min_points=1000):

    significant_planes = []
    remaining_cloud = point_cloud
    plane_index = 1

    while True:
        plane_model, inliers = remaining_cloud.segment_plane(distance_threshold=distance_threshold,
                                                             ransac_n=ransac_n,
                                                             num_iterations=num_iterations)
        if len(inliers) < min_points:
            break

        print(f"Plane {plane_index} equation {plane_model}")
        plane_index += 1

        inlier_point_cloud = remaining_cloud.select_by_index(inliers)
        inlier_point_cloud.paint_uniform_color(np.random.rand(3))

        significant_planes.append((inlier_point_cloud, plane_model))

        remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)

    return significant_planes, remaining_cloud


# --------------------------------------------- PART 2 ------------------------------------------------
# -------------------------------- DBSCAN ON SEGMENTED PLANES (RED) -----------------------------------

def cluster_plane_patches(significant_planes, camera_normal=np.array([1.0, 0.0, 0.0])):

    all_plane_sub_hulls = []
    all_noise_pcds = []
    plane_dbscan_noise_pcds = []

    print(
        f"\nProcessing {len(significant_planes)} distinct "
        f"plane point clouds with Adaptive DBSCAN"
    )

    for plane_idx, (plane_pcd, plane_model) in enumerate(significant_planes):

        print(f"\n--- Analyzing Plane {plane_idx + 1}/{len(significant_planes)}"
              f"({len(plane_pcd.points)} points) ---")

        t0 = time.perf_counter()

        plane_normal = np.array(plane_model[:3])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        alignment = np.abs(np.dot(plane_normal, camera_normal))

        if alignment < 0.2:
            print(
                f"-> Alignment: {alignment:.3f} (PARALLEL to camera path)."
                f"Applying loose clustering parameters."
            )
            current_eps = 0.01
            current_min_points = 4
        else:
            print(
                f"-> Alignment: {alignment:.3f} (FACING camera/Typical orientation). "
                f"Applying strict parameters."
            )
            current_eps = 0.007
            current_min_points = 6

        # ---------------- Stage 1: Clustering on planes -------------------

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            plane_labels = np.array(plane_pcd.cluster_dbscan(eps=current_eps,
                                                             min_points=current_min_points,
                                                             print_progress=True)
                                                             )
        print(f"DBSCAN time: {time.perf_counter() - t0:.3f} s")

        # --------------- Save Noise from plane clustering ------------------

        noise_indices = np.where(plane_labels == -1)[0]

        noise_pcd = plane_pcd.select_by_index(noise_indices)

        # For visualization of noise from plane clusters
        all_noise_pcds.append(noise_pcd)

        # For further clustering of all outliers
        plane_dbscan_noise_pcds.append(noise_pcd)

        noise_pcd.paint_uniform_color([0.6, 0.6, 0.6]) # grey

        plane_max_label = plane_labels.max()

        num_plane_clusters = (plane_max_label + 1 if plane_max_label >= 0 else 0)

        print(
            f"Plane {plane_idx + 1} broken into "
            f"{num_plane_clusters} spatially distinct sub-clusters"
        )

        if num_plane_clusters == 0:
            continue

        # Color plane clusters
        cmap_values = (plane_labels + (plane_idx * 5)) / ((plane_max_label + (plane_idx * 5))
            if (plane_max_label + (plane_idx * 5)) > 0
            else 1
        )

        colors = plt.get_cmap("tab20")(cmap_values)
        colors[plane_labels < 0] = 0
        plane_pcd.colors = (o3d.utility.Vector3dVector(colors[:, :3]))

        # ---------- Convex Hull for each plane sub-cluster (RED) ----------

        t0 = time.perf_counter()

        for cluster_idx in range(num_plane_clusters):

            cluster_indices = np.where(plane_labels == cluster_idx)[0].tolist()

            sub_cluster_pcd = (plane_pcd.select_by_index(cluster_indices))

            if len(sub_cluster_pcd.points) >= 3:
                try:
                    hull_ls_plane, _ = (sub_cluster_pcd.compute_convex_hull())
                    hull_ls_plane.compute_vertex_normals()
                    hull_ls_plane.paint_uniform_color((1, 0, 0))

                    all_plane_sub_hulls.append(hull_ls_plane)

                except Exception:
                    continue

        print(
            f"Convex hull time: "
            f"{time.perf_counter() - t0:.3f} s"
        )

    print(
        f"Part 1 complete! "
        f"Extracted {len(all_plane_sub_hulls)} plane patches."
    )

    return (
        all_plane_sub_hulls,
        all_noise_pcds,
        plane_dbscan_noise_pcds,
    )


# --------------------------------------------- PART 3 ------------------------------------------------
# ----------------------------- DBSCAN ON OUTLIERS POINTS (GREEN & BLUE) ------------------------------

def cluster_outlier_patches(outliers, plane_dbscan_noise_pcds=None):
    convex_hulls_list = []

    print(f"\nCombining original RANSAC outliers with plane DBSCAN noise")

    combined_outliers = o3d.geometry.PointCloud()

    # Original RANSAC outliers
    combined_outliers += outliers

    # Plane DBSCAN noise
    if plane_dbscan_noise_pcds is not None:
        for noise_pcd in plane_dbscan_noise_pcds:
            combined_outliers += noise_pcd

    print(f"Original RANSAC outliers: {len(outliers.points)} points")
    print(f"Combined outlier cloud: {len(combined_outliers.points)} points")

    # ---------------- Stage 2: Dense Clustering on Outliers -------------------

    print(f"\nRunning Primary DBSCAN on {len(combined_outliers.points)} non-planar/outlier points")

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        outlier_labels = np.array(combined_outliers.cluster_dbscan(eps=0.007,
                                                                   min_points=7,
                                                                   print_progress=True))

    outlier_max_label = outlier_labels.max()

    num_outlier_clusters = (outlier_max_label + 1
                            if outlier_max_label >= 0
                            else 0
                            )

    print(f"Primary outlier cloud has {num_outlier_clusters} dense structural clusters")

    # --------------- Convex Hull of Stage 2 Clusters (GREEN) -----------------

    print(f"Computing convex hulls for {num_outlier_clusters} dense structural clusters")

    for cluster_idx in range(num_outlier_clusters):

        cluster_indices = np.where(outlier_labels == cluster_idx)[0].tolist()
        cluster_pcd = (combined_outliers.select_by_index(cluster_indices))

        if len(cluster_pcd.points) >= 4:
            try:
                hull_ls_outlier, _ = (cluster_pcd.compute_convex_hull())
                hull_ls_outlier.compute_vertex_normals()
                hull_ls_outlier.paint_uniform_color((0, 1, 0))

                convex_hulls_list.append(hull_ls_outlier)

            except Exception:
                continue

    #  --------------------- Noise from Stage 2 Clusters ------------------------

    outlier_noise_indices = np.where(outlier_labels == -1)[0].tolist()

    outlier_noise_pcd = (combined_outliers.select_by_index(outlier_noise_indices))

    print(f"\nRunning Secondary DBSCAN on {len(outlier_noise_pcd.points)} outlier noise points")

    # -------------- Stage 3: Relaxed Clustering on Stage 2 Noise ----------------

    final_noise_pcd = o3d.geometry.PointCloud()

    if len(outlier_noise_pcd.points) > 0:

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):

            secondary_labels = np.array(outlier_noise_pcd.cluster_dbscan(eps=0.01,
                                                                         min_points=3,
                                                                         print_progress=True))

        secondary_max_label = (secondary_labels.max())
        num_secondary_clusters = (secondary_max_label + 1
                                  if secondary_max_label >= 0
                                  else 0
                                  )

        print(f"Secondary outlier pass extracted {num_secondary_clusters} sparse structural clusters")

        # --------------- Convex Hull of Stage 3 Clusters (BLUE) ------------------

        for cluster_idx in range(num_secondary_clusters):

            cluster_indices = np.where(secondary_labels == cluster_idx)[0].tolist()
            sub_cluster_pcd = (outlier_noise_pcd.select_by_index(cluster_indices))

            if len(sub_cluster_pcd.points) >= 4:
                try:
                    hull_ls_secondary, _ = (sub_cluster_pcd.compute_convex_hull())
                    hull_ls_secondary.compute_vertex_normals()
                    hull_ls_secondary.paint_uniform_color((0, 0, 1))

                    convex_hulls_list.append(hull_ls_secondary)

                except Exception:
                    continue

        # ------------------------- Final Noise (GREY) -----------------------------

        true_noise_indices = np.where(secondary_labels == -1)[0].tolist()

        final_noise_pcd = (outlier_noise_pcd.select_by_index(true_noise_indices))
        final_noise_pcd.paint_uniform_color([0.3, 0.3, 0.3])

    else:
        print("No noise points found in primary outlier pass to re-cluster.")

    print(f"Successfully generated total of {len(convex_hulls_list)} outlier convex hulls!")

    return convex_hulls_list, final_noise_pcd
    

# --------------------------------------------- PART 4 ------------------------------------------------
# ------------------------------ CONVERT CONVEX HULL TO DRAKE VERTICES --------------------------------

def complete_hulls_to_obstacle_vertices(all_plane_sub_hulls, convex_hulls_list):

    all_hull_meshes = all_plane_sub_hulls + convex_hulls_list

    obstacle_verts_list = []

    for mesh in all_hull_meshes:

        verts = np.asarray(mesh.vertices)

        if verts.shape[0] >= 4:
            obstacle_verts_list.append(verts)

    print("Total obstacles built from hulls:", len(obstacle_verts_list))

    return obstacle_verts_list


# --------------------------------------------- PART 5 ------------------------------------------------
# -------------------------------------- CONVENIENCE WRAPPER ------------------------------------------

def prepare_complete_collision_geometry(point_cloud, camera_normal=np.array([1.0, 0.0, 0.0])):
    """
    Run the complete point-cloud to Drake-obstacle preparation pipeline.

    Returns:
        obstacle_verts_list
    """
    
    # --- RANSAC plane segmentation ------------------

    significant_planes, outliers = segment_planes(point_cloud)

    # --- Multi-Stage DBSCAN -------------------------
    # --- Stage 1: DBSCAN on segmented planes --------

    all_plane_sub_hulls, plane_noise_pcds, plane_dbscan_noise_pcds = cluster_plane_patches(
                                                                        significant_planes,
                                                                        camera_normal=camera_normal)
    # --- Stage 2 & Stage 3 DBSCAN --------------------
    # --- Stage 2: DBSCAN on Outliers  ----------------
    # --- Stage 3: DCSCAN on Stage 2's Noise ----------

    convex_hulls_list, final_noise_pcd = cluster_outlier_patches(outliers,
                                                                 plane_dbscan_noise_pcds)

    # --- Convex Hull to DRAKE Vertices ---------------

    obstacle_verts_list = complete_hulls_to_obstacle_vertices(all_plane_sub_hulls, convex_hulls_list)

    return obstacle_verts_list, significant_planes


# ------------------------------------------ OLD DEVELOPMENT ------------------------------------------
# ----------------------- OBSTACLES REPRESENTATION WITH ONLY PLANAR SURFACES ONLY ---------------------

def plane_hulls_to_obstacle_vertices(plane_hulls):
    obstacle_verts_list = []

    for hull in plane_hulls:
        if len(hull.vertices) >= 4:
            obstacle_verts_list.append(np.asarray(hull.vertices))

    return obstacle_verts_list


def prepare_plane_only_geometry(point_cloud, camera_normal=np.array([1.0, 0.0, 0.0])):

    significant_planes, _ = segment_planes(point_cloud)
    all_plane_sub_hulls, _, _ = cluster_plane_patches(significant_planes, camera_normal=camera_normal)

    obstacle_verts_list = plane_hulls_to_obstacle_vertices(all_plane_sub_hulls)

    return obstacle_verts_list
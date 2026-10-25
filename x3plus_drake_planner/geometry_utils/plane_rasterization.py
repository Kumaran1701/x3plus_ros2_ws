import copy
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _components(mask):
    """Return the 8-connected components of a binary mask as uint8 masks."""
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    return [
        (labels == label).astype(np.uint8) * 255
        for label in range(1, count)
    ]


def _solidity(mask):
    """Return filled-outline solidity and filled area, ignoring interior holes."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 1.0, 0

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    area = cv2.countNonZero(filled)

    hull_mask = np.zeros_like(mask)
    contour_points = np.vstack(contours)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(contour_points), 255)
    hull_area = cv2.countNonZero(hull_mask)
    return area / max(hull_area, 1), area


def _deepest_defect(mask, min_depth_px):
    """Return the deepest convexity-defect point, if it exceeds the threshold."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    contour = max(contours, key=len)
    if len(contour) < 4:
        return None

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return None

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return None
    if defects is None:
        return None

    defects = np.asarray(defects).reshape(-1, 4)
    deepest = defects[int(np.argmax(defects[:, 3]))]
    if deepest[3] / 256.0 < min_depth_px:
        return None

    x, y = contour[deepest[2], 0]
    return int(x), int(y)


def split_nonconvex(
    mask,
    min_solidity=0.92,
    min_defect_px=3.0,
    min_piece_px=20,
    max_depth=6,
    _depth=0,
):
    """Recursively split a binary mask into nearly convex, non-overlapping masks.

    Cuts are horizontal or vertical in the 2D raster coordinates. The returned
    masks collectively cover the input pixels.
    """
    solidity, area = _solidity(mask)
    if (
        solidity >= min_solidity
        or _depth >= max_depth
        or area < 2 * min_piece_px
    ):
        return [mask]

    notch = _deepest_defect(mask, min_defect_px)
    if notch is None:
        return [mask]
    x0, y0 = notch

    best_score = solidity
    best_parts = None
    for axis, position in (("v", x0), ("v", x0 + 1), ("h", y0), ("h", y0 + 1)):
        first, second = mask.copy(), mask.copy()
        if axis == "v":
            first[:, position:] = 0
            second[:, :position] = 0
        else:
            first[position:, :] = 0
            second[:position, :] = 0

        parts = _components(first) + _components(second)
        if not parts or any(cv2.countNonZero(part) < min_piece_px for part in parts):
            continue

        stats = [_solidity(part) for part in parts]
        total_area = sum(part_area for _, part_area in stats)
        score = sum(
            part_solidity * part_area
            for part_solidity, part_area in stats
        ) / total_area
        if score > best_score + 1e-3:
            best_score = score
            best_parts = parts

    if best_parts is None:
        return [mask]

    result = []
    for part in best_parts:
        result.extend(
            split_nonconvex(
                part,
                min_solidity=min_solidity,
                min_defect_px=min_defect_px,
                min_piece_px=min_piece_px,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        )
    return result


def refine_labels(label_image, **kwargs):
    """Split each nonzero label into near-convex regions and return labels/count."""
    refined = np.zeros_like(label_image)
    next_label = 1

    for label in np.unique(label_image):
        if label == 0:
            continue
        mask = (label_image == label).astype(np.uint8) * 255
        for part in split_nonconvex(mask, **kwargs):
            refined[part > 0] = next_label
            next_label += 1

    return refined, next_label - 1


def _plane_basis(normal):
    """Build orthonormal in-plane axes for projecting a plane to 2D."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])

    axis_u = np.cross(normal, reference)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def _convex_hull_for_patch(sub_cloud, plane_normal):
    """Build a convex mesh, extruding planar points if the direct hull fails."""
    try:
        hull, _ = sub_cloud.compute_convex_hull()
        return hull
    except RuntimeError:
        print("Direct convex hull failed; retrying with a 5 mm plane-normal extrusion.")
        points = np.asarray(sub_cloud.points)
        extrusion_offset = plane_normal * 0.005
        volumetric_points = np.vstack(
            (points + extrusion_offset, points - extrusion_offset)
        )

        volumetric_cloud = o3d.geometry.PointCloud()
        volumetric_cloud.points = o3d.utility.Vector3dVector(volumetric_points)
        try:
            hull, _ = volumetric_cloud.compute_convex_hull()
            return hull
        except RuntimeError:
            print("Extruded convex hull failed; using an axis-aligned box fallback.")
            bounds = sub_cloud.get_axis_aligned_bounding_box()
            minimum = bounds.get_min_bound()
            maximum = bounds.get_max_bound()
            center = (minimum + maximum) / 2.0
            extent = np.maximum(maximum - minimum, 0.01)

            try:
                hull = o3d.geometry.TriangleMesh.create_box(
                    width=float(extent[0]),
                    height=float(extent[1]),
                    depth=float(extent[2]),
                )
                hull.translate(center - extent / 2.0)
                return hull
            except RuntimeError as box_error:
                raise RuntimeError(
                    "Could not construct a convex hull for a plane patch"
                ) from box_error


def process_complete_workspace_rasterization(
    significant_planes,
    outliers=None,
    voxel_size=0.005,
    visualize=True,
):
    """Build plane and outlier hulls, returning hulls and DBSCAN noise separately.

    Returns:
        all_workspace_hulls: plane-patch and primary-DBSCAN outlier hull meshes.
        noise: unclustered RANSAC outlier points as an Open3D point cloud.
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be a finite positive value")

    print(
        "\n========================================================================="
    )
    print(
        "Starting Multi-Plane Image-Rasterization Pipeline for "
        f"{len(significant_planes)} Planes"
    )
    print(
        "========================================================================="
    )
    global_start = time.perf_counter()

    all_workspace_hulls = []
    all_processed_clouds = []
    noise = o3d.geometry.PointCloud()
    total_hulls_generated = 0

    for plane_idx, (plane_cloud, plane_model) in enumerate(significant_planes):
        point_count = len(plane_cloud.points)
        print(
            f"\n--- Processing Plane {plane_idx + 1}/{len(significant_planes)} "
            f"({point_count} points) ---"
        )
        plane_start = time.perf_counter()

        points_3d = np.asarray(plane_cloud.points)
        if len(points_3d) < 4:
            print(f"-> Plane {plane_idx + 1} has insufficient points. Skipping.")
            continue

        plane_normal = np.asarray(plane_model[:3], dtype=float)
        normal_length = np.linalg.norm(plane_normal)
        if not np.isfinite(normal_length) or normal_length == 0:
            raise ValueError(f"Plane {plane_idx + 1} has an invalid normal")
        plane_normal /= normal_length

        axis_u, axis_v = _plane_basis(plane_normal)
        points_2d = np.column_stack((points_3d @ axis_u, points_3d @ axis_v))

        min_coords = np.min(points_2d, axis=0)
        pixel_coords = ((points_2d - min_coords) / voxel_size).astype(np.int32)
        image_width, image_height = np.max(pixel_coords, axis=0) + 2
        binary_image = np.zeros(
            (image_height, image_width), dtype=np.uint8
        )
        binary_image[pixel_coords[:, 1], pixel_coords[:, 0]] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        clean_count, clean_labels = cv2.connectedComponents(clean_image)

        residual_image = cv2.absdiff(binary_image, clean_image)
        _, residual_labels = cv2.connectedComponents(residual_image)
        residual_labels = np.where(
            residual_labels > 0,
            residual_labels + (clean_count - 1),
            0,
        )
        final_labels = np.where(clean_labels > 0, clean_labels, residual_labels)
        final_labels, component_count = refine_labels(final_labels)
        assigned_labels = final_labels[pixel_coords[:, 1], pixel_coords[:, 0]]

        color_indices = (assigned_labels + plane_idx * 7) % 20
        colors = plt.get_cmap("tab20")(color_indices / 20.0)[:, :3]
        visual_cloud = copy.deepcopy(plane_cloud)
        visual_cloud.colors = o3d.utility.Vector3dVector(colors)
        all_processed_clouds.append(visual_cloud)

        plane_hulls_count = 0
        processed_point_count = 0
        for component_id in range(1, component_count + 1):
            point_indices = np.flatnonzero(assigned_labels == component_id)
            processed_point_count += len(point_indices)
            if len(point_indices) < 4:
                continue

            sub_cloud = plane_cloud.select_by_index(point_indices.tolist())
            hull = _convex_hull_for_patch(sub_cloud, plane_normal)
            hull.compute_vertex_normals()
            hull.paint_uniform_color((1, 0, 0))
            all_workspace_hulls.append(hull)
            plane_hulls_count += 1
            total_hulls_generated += 1

        print(
            f"-> Plane {plane_idx + 1} processing complete in "
            f"{time.perf_counter() - plane_start:.3f}s. "
            f"Generated {plane_hulls_count} clean red hulls."
        )
        print(f"Processed {processed_point_count} / {point_count} points")

    if outliers is not None and len(outliers.points) > 0:
        print(
            "\n========================================================================="
        )
        print(
            f"Running Primary DBSCAN on {len(outliers.points)} RANSAC outlier points"
        )
        print(
            "========================================================================="
        )
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ):
            outlier_labels = np.asarray(
                outliers.cluster_dbscan(
                    eps=0.007,
                    min_points=7,
                    print_progress=True,
                )
            )

        cluster_count = int(outlier_labels.max()) + 1
        print(f"Primary DBSCAN found {cluster_count} outlier clusters")

        outlier_hulls_generated = 0
        for cluster_id in range(cluster_count):
            point_indices = np.flatnonzero(outlier_labels == cluster_id)
            if len(point_indices) < 4:
                continue

            cluster_cloud = outliers.select_by_index(point_indices.tolist())
            try:
                hull, _ = cluster_cloud.compute_convex_hull()
            except RuntimeError:
                print(
                    f"Outlier cluster {cluster_id} convex hull failed; "
                    "using an axis-aligned box fallback."
                )
                bounds = cluster_cloud.get_axis_aligned_bounding_box()
                minimum = bounds.get_min_bound()
                maximum = bounds.get_max_bound()
                center = (minimum + maximum) / 2.0
                extent = np.maximum(maximum - minimum, 0.01)
                hull = o3d.geometry.TriangleMesh.create_box(
                    width=float(extent[0]),
                    height=float(extent[1]),
                    depth=float(extent[2]),
                )
                hull.translate(center - extent / 2.0)

            hull.compute_vertex_normals()
            hull.paint_uniform_color((0, 1, 0))
            all_workspace_hulls.append(hull)
            outlier_hulls_generated += 1
            total_hulls_generated += 1

        noise_indices = np.flatnonzero(outlier_labels == -1)
        if len(noise_indices) > 0:
            noise = outliers.select_by_index(noise_indices.tolist())
            noise.paint_uniform_color((0.3, 0.3, 0.3))

        print(f"Generated {outlier_hulls_generated} green outlier hulls")
        print(f"Remaining DBSCAN noise points: {len(noise.points)}")

    print(
        "\n========================================================================="
    )
    print(
        "ALL PLANES PROCESSED. Global Pipeline Execution Time: "
        f"{time.perf_counter() - global_start:.3f}s"
    )
    print(
        "Total Plane and Outlier Collision Hulls Generated: "
        f"{total_hulls_generated}"
    )
    print(f"DBSCAN noise points returned separately: {len(noise.points)}")
    print(
        "========================================================================="
    )

    if visualize:
        print("Launching Unified 3D Global Space View...")
        o3d.visualization.draw_geometries(
            all_workspace_hulls + all_processed_clouds + [noise],
            window_name="Unified Global Space View: Rasterized Planes + Outliers",
            width=1400,
            height=850,
            mesh_show_back_face=True,
        )

    return all_workspace_hulls, noise

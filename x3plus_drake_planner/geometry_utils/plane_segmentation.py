import numpy as np


# ----------------------------------- RANSAC PLANE SEGMENTATION ---------------------------------------

def segment_planes(point_cloud, distance_threshold=0.005, ransac_n=3, num_iterations=1000, min_points=1000):
    """
    Iteratively segment dominantplanes from the point cloud using RANSAC. 

    Returns: 
        significant_planes: a list of plane equation and its corresponding points
        remaining_cloud: the outliers after removing all significant planes.
    """
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



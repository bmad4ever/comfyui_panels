from shapely.geometry import Polygon
import numpy as np
import math
import cv2


def grow_4sided_polygon(
        polygon: Polygon,
        directions: list[str],
        local: bool,
        distance: float
) -> Polygon:
    """
    Adjust polygon edges based on cardinal directions.

    Parameters:
    -----------
    polygon : Polygon
        Shapely polygon with exterior coordinates
    directions : List[str]
        List of cardinal directions ('N', 'S', 'E', 'W')
    local : bool
        If True, move along edge normal; if False, move along cardinal direction
    distance : float
        Distance to move the vertices

    Returns:
    --------
    Polygon
        New polygon with adjusted edges
    """

    # Cardinal direction vectors
    CARDINAL_VECTORS = {
        'N': np.array([0, -1]),
        'S': np.array([0, 1]),
        'E': np.array([1, 0]),
        'W': np.array([-1, 0])
    }

    # Extract coordinates (excluding the duplicate last point)
    coords = np.array(polygon.exterior.coords[:-1], dtype=np.float32)
    n_points = len(coords)

    # Cluster vertices into 4 groups (corners) using OpenCV kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, cluster_centers = cv2.kmeans(
        coords,
        4,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    # Flatten labels array
    labels = labels.flatten()

    # Sort clusters by position (to maintain polygon order)
    cluster_order = _order_clusters_spatially(cluster_centers)

    # Create mapping from old to new cluster labels
    label_mapping = {old: new for new, old in enumerate(cluster_order)}
    labels = np.array([label_mapping[label] for label in labels])

    # Identify edges (consecutive cluster pairs)
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        # Find indices belonging to current and next cluster
        curr_indices = np.where(labels == i)[0]
        next_indices = np.where(labels == next_i)[0]

        # Edge vertices are the ones at the boundary between clusters
        edge_indices = _find_edge_vertices(coords, curr_indices, next_indices, labels)

        if len(edge_indices) >= 2:
            edges.append({
                'indices': edge_indices,
                'cluster_pair': (i, next_i)
            })

    # Calculate edge normals and match to directions
    edge_directions = []
    for edge in edges:
        normal = _calculate_edge_normal(coords, edge['indices'])
        edge_directions.append({
            'edge': edge,
            'normal': normal
        })

    # Create a copy of coordinates for modification
    new_coords = coords.copy()

    # Process each requested direction
    for direction in directions:
        cardinal_vec = CARDINAL_VECTORS[direction.upper()]

        # Find the edge that best matches this direction
        best_edge = None
        best_similarity = -1

        for edge_info in edge_directions:
            # Calculate similarity (dot product)
            similarity = np.dot(edge_info['normal'], cardinal_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_edge = edge_info

        if best_edge is not None:
            # Determine movement vector
            if local:
                move_vector = best_edge['normal'] * distance
            else:
                move_vector = cardinal_vec * distance

            # Move the vertices of this edge
            for idx in best_edge['edge']['indices']:
                new_coords[idx] += move_vector

    # Create new polygon (close the loop by adding first point at end)
    new_polygon_coords = np.vstack([new_coords, new_coords[0:1]])

    return Polygon(new_polygon_coords)


def _order_clusters_spatially(centers: np.ndarray) -> list[int]:
    """Order cluster centers spatially (e.g., clockwise from top-left)."""
    # Calculate angle from centroid
    centroid = centers.mean(axis=0)
    angles = []

    for i, center in enumerate(centers):
        vec = center - centroid
        angle = math.atan2(vec[1], vec[0])
        angles.append((angle, i))

    # Sort by angle to get spatial order
    angles.sort(reverse=True)  # Counter-clockwise from right
    return [idx for _, idx in angles]


def _find_edge_vertices(
        coords: np.ndarray,
        curr_indices: np.ndarray,
        next_indices: np.ndarray,
        all_labels: np.ndarray
) -> list[int]:
    """Find vertices that form the edge between two clusters."""
    n = len(coords)
    edge_vertices = []

    # Find transition points in the sequence
    for i in range(n):
        curr_label = all_labels[i]
        next_label = all_labels[(i + 1) % n]

        # Check if this is a transition between the two clusters
        if (curr_label == all_labels[curr_indices[0]] and
                next_label == all_labels[next_indices[0]]):
            # Include vertices around the transition
            edge_vertices.append(i)
        elif curr_label == all_labels[curr_indices[0]]:
            # Check if next vertex is the start of next cluster
            if (i + 1) % n in next_indices:
                edge_vertices.append(i)

    # If we found vertices between clusters, also include some from next cluster
    if edge_vertices:
        last_idx = edge_vertices[-1]
        for offset in range(1, min(4, n)):
            next_idx = (last_idx + offset) % n
            if all_labels[next_idx] == all_labels[next_indices[0]]:
                edge_vertices.append(next_idx)
            else:
                break

    return edge_vertices if edge_vertices else list(curr_indices)


def _calculate_edge_normal(coords: np.ndarray, indices: list[int]) -> np.ndarray:
    """Calculate the outward normal vector for an edge."""
    if len(indices) < 2:
        return np.array([1.0, 0.0])

    # Calculate edge direction (from first to last vertex in edge)
    edge_start = coords[indices[0]]
    edge_end = coords[indices[-1]]
    edge_vec = edge_end - edge_start

    # Normalize
    edge_length = np.linalg.norm(edge_vec)
    if edge_length < 1e-10:
        return np.array([1.0, 0.0])

    edge_vec = edge_vec / edge_length

    # Calculate perpendicular (rotate 90 degrees counterclockwise)
    normal = np.array([-edge_vec[1], edge_vec[0]])

    # Ensure it points outward (check against polygon centroid)
    centroid = coords.mean(axis=0)
    edge_center = coords[indices].mean(axis=0)
    outward = edge_center - centroid

    # Flip if pointing inward
    if np.dot(normal, outward) < 0:
        normal = -normal

    return normal

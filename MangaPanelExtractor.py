import itertools

import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid


class MangaPanelExtractor:
    def __init__(self, threshold_value: int = 240, min_rel_panel_area: float = 0.01,
                 expansion_pixels: int = 50, max_vertices: int = 6):
        """
        Initialize the manga panel extractor.

        Args:
            threshold_value: Threshold for converting to binary (higher = more selective for white)
            min_rel_panel_area: Minimum panel area as percentage of total page area (0.01 = 1%)
            expansion_pixels: Pixels to expand image borders for edge panel detection
            max_vertices: Maximum vertices allowed in simplified polygon (default 6)
        """
        self.threshold_value = threshold_value
        self.min_rel_panel_area = min_rel_panel_area
        self.expansion_pixels = expansion_pixels
        self.max_vertices = max_vertices
        self.min_panel_area = None  # Will be computed from image size

    def find_empty_corner(self, binary_img: np.ndarray, corner_size: int = 50) -> Tuple[int, int]:
        """
        Find an empty corner (white area) to start flood fill.
        Checks all four corners and returns coordinates of the emptiest one.
        """
        h, w = binary_img.shape
        corners = [
            (0, 0),  # Top-left
            (0, w - 1),  # Top-right
            (h - 1, 0),  # Bottom-left
            (h - 1, w - 1)  # Bottom-right
        ]

        best_corner = corners[0]
        max_white_pixels = 0

        for corner in corners:
            y, x = corner
            # Define sample region around corner
            y_start = max(0, y - corner_size // 2)
            y_end = min(h, y + corner_size // 2)
            x_start = max(0, x - corner_size // 2)
            x_end = min(w, x + corner_size // 2)

            sample_region = binary_img[y_start:y_end, x_start:x_end]
            white_pixels = np.sum(sample_region == 255)

            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                best_corner = corner

        return best_corner

    def expand_image(self, img: np.ndarray, pixels: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Expand image by adding white border around it.

        Returns:
            Expanded image and offset coordinates (for coordinate correction later)
        """
        if len(img.shape) == 3:  # Color image
            expanded = cv2.copyMakeBorder(
                img, pixels, pixels, pixels, pixels,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        else:  # Grayscale
            expanded = cv2.copyMakeBorder(
                img, pixels, pixels, pixels, pixels,
                cv2.BORDER_CONSTANT, value=255
            )

        return expanded, (pixels, pixels)

    def simplify_contour(self, contour: np.ndarray, max_vertices: int = None) -> np.ndarray:
        """
        Simplify contour to reduce vertices, preferring rectangular shapes.
        """
        if max_vertices is None:
            max_vertices = self.max_vertices

        # Calculate contour perimeter for epsilon calculation
        perimeter = cv2.arcLength(contour, True)

        # Start with a small epsilon and gradually increase until we get desired vertex count
        epsilon_factor = 0.01
        simplified = contour

        while len(simplified) > max_vertices and epsilon_factor < 0.1:
            epsilon = epsilon_factor * perimeter
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            epsilon_factor += 0.005

        # If still too many vertices, try more aggressive simplification
        if len(simplified) > max_vertices:
            epsilon = 0.1 * perimeter
            simplified = cv2.approxPolyDP(contour, epsilon, True)

        return simplified

    def contour_to_shapely_polygon(self, contour: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> Polygon:
        """
        Convert OpenCV contour to Shapely polygon, adjusting for image expansion offset.
        """
        # Adjust coordinates back to original image space
        offset_x, offset_y = offset
        points = []

        for point in contour:
            x, y = point[0]
            # Subtract offset to get original coordinates
            original_x = x - offset_x
            original_y = y - offset_y
            points.append((original_x, original_y))

        # Create polygon and ensure it's valid
        try:
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            return polygon
        except Exception:
            # Fallback: create a simple polygon from bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    def calculate_internal_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate the internal angle at point p2 formed by p1-p2-p3.
        Returns angle in degrees.
        """
        # Create vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle floating point errors
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def score_vertex_for_removal(self, contour: np.ndarray, vertex_idx: int) -> Tuple[float, float]:
        """
        Score a vertex for removal based on internal angle and area gain.

        Returns:
            (angle_score, area_score) where higher scores = better candidates for removal
        """
        n_points = len(contour)
        if n_points <= 3:  # Can't remove from triangle or smaller
            return 0.0, 0.0

        # Get the three points for angle calculation
        prev_idx = (vertex_idx - 1) % n_points
        next_idx = (vertex_idx + 1) % n_points

        p1 = contour[prev_idx][0]
        p2 = contour[vertex_idx][0]
        p3 = contour[next_idx][0]

        # Calculate internal angle
        internal_angle = self.calculate_internal_angle(p1, p2, p3)

        # Angle score: higher for angles close to 180° (straight lines/jagged cuts)
        # Peak score at 180°, lower scores for acute angles
        angle_score = 1.0 - abs(internal_angle - 180.0) / 180.0

        # Calculate area gain by removing this vertex
        original_area = cv2.contourArea(contour)

        # Create new contour without this vertex
        new_contour = np.delete(contour, vertex_idx, axis=0)

        if len(new_contour) >= 3:
            new_area = cv2.contourArea(new_contour)
            area_gain = new_area - original_area
            # Normalize area score (higher is better for removal)
            area_score = max(0, area_gain / original_area) if original_area > 0 else 0
        else:
            area_score = 0.0

        return angle_score, area_score

    def convert_nonconvex_to_convex(self, contour: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Attempt to convert a non-convex shape to convex by removing problematic vertices.

        Args:
            contour: OpenCV contour
            max_iterations: Maximum number of vertices to remove

        Returns:
            Modified contour (convex if successful)
        """
        current_contour = contour.copy()

        for iteration in range(max_iterations):
            # Check if already convex
            if self.is_convex(current_contour):
                break

            # Need at least 4 points to continue removing
            if len(current_contour) <= 3:
                break

            # Score all vertices for removal
            vertex_scores = []
            for i in range(len(current_contour)):
                angle_score, area_score = self.score_vertex_for_removal(current_contour, i)

                # Combined score: prioritize high internal angles, then area gain
                combined_score = angle_score * 2.0 + area_score * 1.0
                vertex_scores.append((i, combined_score, angle_score, area_score))

            # Sort by combined score (highest first)
            vertex_scores.sort(key=lambda x: x[1], reverse=True)

            # Remove the vertex with highest score
            if vertex_scores and vertex_scores[0][1] > 0:
                best_vertex_idx = vertex_scores[0][0]
                current_contour = np.delete(current_contour, best_vertex_idx, axis=0)
            else:
                # No good candidates for removal
                break

        return current_contour

    def analyze_nonconvex_recovery(self, results: Dict) -> Dict:
        """
        Attempt to recover panels from non-convex shapes using vertex removal.

        Returns:
            Updated results with recovered panels
        """
        recovered_panels = []
        remaining_nonconvex = []

        for shape in results['non_convex_shapes']:
            original_contour = shape['contour']

            # Attempt to make it convex
            recovered_contour = self.convert_nonconvex_to_convex(original_contour)

            # Check if we successfully made it convex and it's still a reasonable size
            if (self.is_convex(recovered_contour) and
                    cv2.contourArea(recovered_contour) > self.min_panel_area * 0.5):  # Allow smaller after recovery

                # Convert to polygon and analyze
                polygon = self.contour_to_shapely_polygon(recovered_contour, results['offset'])
                shape_analysis = self.analyze_panel_shape(polygon)

                recovered_panels.append({
                    'contour': recovered_contour,
                    'original_contour': original_contour,
                    'polygon': polygon,
                    'area': cv2.contourArea(recovered_contour),
                    'bbox': cv2.boundingRect(recovered_contour),
                    'shape_analysis': shape_analysis,
                    'type': 'recovered_panel',
                    'recovery_info': {
                        'original_vertices': len(original_contour),
                        'final_vertices': len(recovered_contour),
                        'vertices_removed': len(original_contour) - len(recovered_contour)
                    }
                })
            else:
                # Keep as non-convex
                remaining_nonconvex.append(shape)

        # Update results
        results['recovered_panels'] = recovered_panels
        results['non_convex_shapes'] = remaining_nonconvex
        results['panels'].extend(recovered_panels)  # Add to main panels list

        return results

    def analyze_panel_shape(self, polygon: Polygon) -> Dict:
        """
        Analyze the shape characteristics of a panel polygon.
        """
        coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point
        num_vertices = len(coords)

        # Calculate aspect ratio from bounding box
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect_ratio = width / height if height > 0 else 1.0

        # Determine likely panel type
        if num_vertices == 4:
            panel_type = "rectangular"
        elif num_vertices == 3:
            panel_type = "triangular"
        elif num_vertices <= 6:
            panel_type = f"{num_vertices}-sided"
        else:
            panel_type = "complex"

        return {
            'vertices': num_vertices,
            'aspect_ratio': aspect_ratio,
            'panel_type': panel_type,
            'area': polygon.area,
            'bounds': bounds
        }
        """
        Find an empty corner (white area) to start flood fill.
        Checks all four corners and returns coordinates of the emptiest one.
        """
        h, w = binary_img.shape
        corners = [
            (0, 0),  # Top-left
            (0, w - 1),  # Top-right
            (h - 1, 0),  # Bottom-left
            (h - 1, w - 1)  # Bottom-right
        ]

        best_corner = corners[0]
        max_white_pixels = 0

        for corner in corners:
            y, x = corner
            # Define sample region around corner
            y_start = max(0, y - corner_size // 2)
            y_end = min(h, y + corner_size // 2)
            x_start = max(0, x - corner_size // 2)
            x_end = min(w, x + corner_size // 2)

            sample_region = binary_img[y_start:y_end, x_start:x_end]
            white_pixels = np.sum(sample_region == 255)

            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                best_corner = corner

        return best_corner

    def is_convex(self, contour: np.ndarray) -> bool:
        """
        Check if a contour represents a convex shape.
        """
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)

        if hull_area == 0:
            return False

        # If the ratio is close to 1, the shape is convex
        convexity_ratio = contour_area / hull_area
        return convexity_ratio > 0.85  # Allow some tolerance

    def extract_panels(self, img: np.ndarray) -> Dict:
        """
        Extract panels from manga page using refined flood fill approach.

        :param img: openCV ready, in BGR format, image.
        :return: Dictionary containing panels, non-convex shapes, and debug images
        """
        # Load and preprocess image
        # img = cv2.imread(image_path)
        #if img is None:
        #    raise ValueError(f"Could not load image: {image_path}")
        # (image[0].cpu().numpy()[..., ::-1] * 255).astype(np.uint8)

        # Store original dimensions for reference
        original_shape = img.shape
        page_area = original_shape[0] * original_shape[1]  # height * width
        self.min_panel_area = page_area * self.min_rel_panel_area

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Expand image to detect edge panels
        expanded_img, offset = self.expand_image(img, self.expansion_pixels)
        expanded_gray, _ = self.expand_image(gray, self.expansion_pixels)

        # Create binary image (white background, black content)
        _, binary = cv2.threshold(expanded_gray, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Find empty corner for flood fill (now in expanded image)
        start_point = self.find_empty_corner(binary)
        start_point = list(start_point)[::-1]  # swap x and y for floodfill =/

        # Create flood fill mask
        h, w = binary.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Perform flood fill from empty corner
        flood_filled = binary.copy()
        cv2.floodFill(flood_filled, mask, start_point, 128)  # Fill with gray (128)

        # Create mask of non-filled areas (panels and content)
        panel_mask = (flood_filled != 128).astype(np.uint8) * 255

        # Find contours in the panel mask
        contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Classify and process contours
        panels = []
        non_convex_shapes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter out very small areas (noise)
            if area < self.min_panel_area:
                continue

            # Simplify contour
            simplified_contour = self.simplify_contour(contour)

            # Convert to Shapely polygon
            polygon = self.contour_to_shapely_polygon(simplified_contour, offset)

            # Analyze shape
            shape_analysis = self.analyze_panel_shape(polygon)

            # Check if convex and classify
            if self.is_convex(simplified_contour):
                panels.append({
                    'contour': simplified_contour,
                    'original_contour': contour,
                    'polygon': polygon,
                    'area': area,
                    'bbox': cv2.boundingRect(simplified_contour),
                    'shape_analysis': shape_analysis,
                    'type': 'panel'
                })
            else:
                non_convex_shapes.append({
                    'contour': simplified_contour,
                    'original_contour': contour,
                    'polygon': polygon,
                    'area': area,
                    'bbox': cv2.boundingRect(simplified_contour),
                    'shape_analysis': shape_analysis,
                    'type': 'non_convex'
                })

        return {
            'original': img,
            'expanded': expanded_img,
            'binary': binary,
            'flood_filled': flood_filled,
            'panel_mask': panel_mask,
            'panels': panels,
            'non_convex_shapes': non_convex_shapes,
            'recovered_panels': [],  # Will be populated if recovery is run
            'start_point': start_point,
            'offset': offset,
            'original_shape': original_shape
        }

    # optional user "enforced" panel shape related funcs

    def simplify_panels(self, results: Dict, method: str) -> Dict:
        for i, r in enumerate(results["panels"]):
            results["panels"][i]["polygon"] = \
                MangaPanelExtractor.simplify_polygon_to_rectangle(results["panels"][i]["polygon"], method)
        return results

    @staticmethod
    def simplify_polygon_to_rectangle(polygon: Polygon, method: str = 'max_area_combination') -> Polygon:
        """
        Simplify a polygon with >4 vertices to exactly 4 vertices (rectangle) by
        maximizing the area of the resulting shape.

        Args:
            polygon: Shapely polygon with >4 vertices
            method: 'max_area_combination' or 'bounding_box'

        Returns:
            Simplified 4-vertex polygon with maximum possible area
        """
        coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point

        if method == 'bounding_box':
            return MangaPanelExtractor._simplify_to_bounding_box(polygon)

        if len(coords) <= 4:
            return polygon

        return MangaPanelExtractor.simplify_maximize_area(polygon)

    @staticmethod
    def simplify_maximize_area(polygon: Polygon) -> Polygon:
        """
        Simplify polygon to 4 vertices by finding the combination that maximizes area.
        """
        coords, original_area = polygon.exterior.coords, polygon.area

        n_vertices = len(coords)

        if n_vertices <= 4:
            return Polygon(coords)

        # For very large polygons, use a heuristic approach to avoid combinatorial explosion
        if n_vertices > 12:
            return MangaPanelExtractor._simplify_large_polygon_max_area_heuristic(coords)

        # Try all combinations of 4 vertices from the original polygon
        max_area = 0
        best_combination = None

        for combination in itertools.combinations(range(n_vertices), 4):
            # Extract the 4 vertices
            selected_coords = [coords[i] for i in combination]

            # Ensure proper ordering (clockwise or counter-clockwise)
            ordered_coords = MangaPanelExtractor._order_vertices_properly(selected_coords)

            try:
                candidate_polygon = Polygon(ordered_coords)
                if not candidate_polygon.is_valid:
                    candidate_polygon = make_valid(candidate_polygon)

                # We want to maximize area
                candidate_area = candidate_polygon.area

                if candidate_area > max_area:
                    max_area = candidate_area
                    best_combination = ordered_coords

            except Exception:
                continue  # Skip invalid combinations

        if best_combination is None:
            # Fallback to bounding box (guaranteed to be large)
            return MangaPanelExtractor._simplify_to_bounding_box(Polygon(coords))

        return Polygon(best_combination)

    @staticmethod
    def _order_vertices_properly(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Order vertices in proper sequence (clockwise or counter-clockwise) for a valid polygon.
        """
        # Calculate centroid
        centroid_x = sum(v[0] for v in vertices) / len(vertices)
        centroid_y = sum(v[1] for v in vertices) / len(vertices)

        # Calculate angle from centroid to each vertex
        def angle_from_centroid(vertex):
            return np.arctan2(vertex[1] - centroid_y, vertex[0] - centroid_x)

        # Sort vertices by angle
        sorted_vertices = sorted(vertices, key=angle_from_centroid)

        return sorted_vertices

    @staticmethod
    def _simplify_to_bounding_box(polygon: Polygon) -> Polygon:
        """Convert polygon to its bounding box rectangle."""
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds

        rectangle_coords = [
            (minx, miny),
            (maxx, miny),
            (maxx, maxy),
            (minx, maxy)
        ]

        return Polygon(rectangle_coords)

    @staticmethod
    def _simplify_large_polygon_heuristic(coords: List[Tuple[float, float]], original_area: float) -> Polygon:
        """
        Heuristic approach for polygons with high vertex count (>12).
        Selects vertices that are most important for maintaining shape.
        """
        n = len(coords)

        # Calculate importance score for each vertex
        vertex_scores = []

        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            # Calculate the area of triangle formed by this vertex and its neighbors
            p1, p2, p3 = coords[prev_idx], coords[i], coords[next_idx]
            triangle_area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2

            # Calculate distance from centroid (corner vertices are typically further)
            centroid = (sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n)
            distance_from_centroid = ((coords[i][0] - centroid[0]) ** 2 + (coords[i][1] - centroid[1]) ** 2) ** 0.5

            # Combined importance score
            importance = triangle_area * 0.7 + distance_from_centroid * 0.3
            vertex_scores.append((i, importance))

        # Sort by importance and take top 4
        vertex_scores.sort(key=lambda x: x[1], reverse=True)
        top_4_indices = [idx for idx, _ in vertex_scores[:4]]
        top_4_indices.sort()  # Maintain original ordering

        selected_coords = [coords[i] for i in top_4_indices]
        ordered_coords = MangaPanelExtractor._order_vertices_properly(selected_coords)

        return Polygon(ordered_coords)

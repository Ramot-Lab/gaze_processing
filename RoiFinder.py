from typing import List
import cv2
import numpy as np
from PanelSymbols import *
from constants import SCREEN_SIZE

class ROI:
    def __init__(self, center: tuple[float, float], radius: int, idx: int, symbol: Symbol = None):
        self.center = center  # (x, y)
        self.radius = radius
        self.idx = idx
        self.symbol = symbol if symbol is not None else None

    def contains(self, point, factor = None, shape = "circle"):
        """Check if a (x, y) point lies inside this ROI, 
        if there is a factor the radius of the roi will be multiplied by iy for the check"""
        if factor is None:
            factor = 1
        factored_radius = self.radius * factor
        px, py = point
        cx, cy = self.center
        if shape == "circle":
            return (px - cx) ** 2 + (py - cy) ** 2 <= factored_radius ** 2
        
        elif shape == "square":
            return (cx - self.radius <= px <= cx + self.radius) and (cy - factored_radius <= py <= cy + factored_radius)
        
        else:
            raise ValueError(f"Unknown shape '{shape}' for ROI.contains()")
    
    def dist_and_angle(self, point: tuple[float, float]):
        """Calculate distance and angle from ROI center to a given point."""
        px, py = point
        cx, cy = self.center
        dx = px - cx
        dy = py - cy
        distance = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx) # radians
        return distance, angle


class RoiFinder:
    def __init__(self, panel_name: str, img):
        self.panel_name = panel_name
        self.img = img
        self.centers = self._find_centers()
        self.rois : List[ROI] = self.set_rois()


    def _estimate_upper_lower_radii(self, centers, split_ratio=0.175):
        """
        Estimate typical ROI radii separately for dictinary symbols in th upper part and in text in the lower parts of the image.

        Parameters
        ----------
        centers : list of (x, y)
            List of ROI center coordinates.
        image_height : int
            Height of the image in pixels.
        split_ratio : float, optional
            Fraction of image height defining the split between upper and lower regions.

        Returns
        -------
        tuple
            (upper_radius, lower_radius)
        """
        if len(centers) < 2:
            raise ValueError("At least two centers required to estimate ROI radii.")

        centers = np.array(centers)
        split_y = SCREEN_SIZE[0] * split_ratio

        # Split centers into upper and lower regions
        upper = centers[centers[:, 1] <= split_y]
        lower = centers[centers[:, 1] > split_y]

        def median_nn_distance(points):
            """Compute median nearest-neighbor distance."""
            if len(points) < 2:
                return None
            dists = []
            for i, c in enumerate(points):
                others = np.delete(points, i, axis=0)
                nearest = np.min(np.linalg.norm(others - c, axis=1))
                dists.append(nearest)
            return np.median(dists)

        upper_dist = median_nn_distance(upper)
        lower_dist = median_nn_distance(lower)

        # Handle cases where one region has too few points
        if upper_dist is None:
            upper_dist = lower_dist
        if lower_dist is None:
            lower_dist = upper_dist

        # Slightly smaller than half the distance between centers
        upper_radius = 0.48 * upper_dist
        lower_radius = 0.48 * lower_dist

        return upper_radius, lower_radius

    def _find_centers(self):
        """
        Find centers of black framed squares in a white image while in each square there is a black symbol.
        Returns list of (x, y) centers.
        """
        # Load the image
        if self.img is None:
            raise ValueError(f"No image provided")
        
        # Ensure grayscale
        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = self.img

        # Apply threshold to ensure pure black and white
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Try line detection approach first
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical lines  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine horizontal and vertical lines to get grid
        grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Find contours on the grid
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If line detection doesn't work well, try original method
        if len(contours) < 50:
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if 300 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio and minimum size
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0 and w > 10 and h > 10:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    centers.append((center_x, center_y))
        
        # Sort centers by reading order (top to bottom, left to right)
        if centers:
            row_height = 80
            centers.sort(key=lambda point: (point[1] // row_height, point[0]))
        # After finding centers, remove duplicates that are too close
        filtered_centers = []
        min_distance = 60  # Minimum distance between valid centers


        for center in centers:
            too_close = any(
                abs(center[0] - existing[0]) < min_distance and 
                abs(center[1] - existing[1]) < min_distance 
                for existing in filtered_centers
            )
            if not too_close:
                filtered_centers.append(center)
        return filtered_centers

    def set_rois(self, split_ratio=0.175):
        """
        Given list of (x, y) centers and estimated radii, returns list of ROI objects.
        """
        centers = self.centers

        # Only estimate once
        if not hasattr(self, "_radii_estimated"):
            upper_radius, lower_radius = self._estimate_upper_lower_radii(centers)
            self.upper_radius = upper_radius
            self.lower_radius = lower_radius
            self._radii_estimated = True
        else:
            upper_radius = self.upper_radius
            lower_radius = self.lower_radius

        rois = []
        for idx, center in enumerate(centers):
            radius = int(upper_radius if center[1] <= SCREEN_SIZE[0] * split_ratio else lower_radius)
            rois.append(ROI(center, radius, idx))

        self.rois = rois
        return rois


    def get_roi_by_position(self, x, y):
        """
        Given an (x, y) position, return the ROI object that contains this point,
        or None if no ROI contains it.
        """
        if not hasattr(self, 'centers'):
            self.set_rois()
        
        for roi in self.rois:
            if roi.contains((x, y)):
                return roi
        return None
    
    def nearest_roi(self, x, y):
        """
        Given an (x, y) position, return the nearest ROI object.
        """
        if not hasattr(self, 'centers'):
            self.rois()
        
        min_dist = float('inf')
        nearest_roi = None
        
        for roi in self.rois:
            cx, cy = roi.center
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_roi = roi
        
        return nearest_roi

    #sainity check
    def visualize_centers(self, save_path=None):
        """
        Visualize the detected centers on the image with large colored circles.
        
        Args:
            centers: List of (x, y) centers. If None, will call find_centers()
            save_path: Path to save the visualization. If None, displays the image
        """
        # Load the original image
        if self.img is None:
            raise ValueError(f"No image provided")
        
        # Get centers if not provided
        if not self.centers:
            raise(ValueError("No ROIs found. Please run _find_centers() first."))
        
        # Draw centers on the image
        for i, (x, y) in enumerate(self.centers):
            # Draw a large colored circle at each center
            cv2.circle(self.img, (x, y), 15, (0, 0, 255), -1)  # Red filled circle
            # Draw a smaller white circle inside for better visibility
            cv2.circle(self.img, (x, y), 8, (255, 255, 255), -1)  # White filled circle
            # Add index number
            cv2.putText(self.img, str(i), (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, self.img)
            print(f"Visualization saved to {save_path}")
        else:
            # Display the image
            cv2.imshow('Centers Visualization', self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return self.img


from typing import List
import cv2
import numpy as np
from PanelSymbols import *

# Single standardized elongation ratio (decision 2026-09-17, revised 2026-09-18): replaces
# the three different ad hoc factors (1, 1.5, 2) previously passed to ROI.contains() at
# various call sites - elongation is now baked into each ROI's own geometry at
# construction time instead of being a per-call argument, so "is this point inside the
# ROI" has exactly one definition everywhere it's checked. The elongated side is exactly
# ROI_ELONGATION_RATIO times the un-elongated side (never additive).
ROI_ELONGATION_RATIO = 2


class ROI:
    def __init__(self, center: tuple[float, float], x_half_left: float, x_half_right: float,
                 y_half_up: float, y_half_down: float, idx: int, symbol: Symbol = None):
        """
        All ROIs are axis-aligned rectangles ("square" in the sense of a box hit-test, not
        necessarily equal width/height - see the asymmetry below). Never elongated sideways
        - x_half_left/x_half_right are each exactly half the gap to that specific neighbor
        (not a row-wide average), so every adjacent pair's edges meet exactly regardless of
        small pixel-to-pixel jitter in the detected spacing.

        y_half_up / y_half_down: independent half-heights above/below center, since the
        dictionary's two legend rows are elongated toward EACH OTHER only:
          - row 1 (symbols, idx < 9): elongated upward only (y_half_up > y_half_down)
          - row 2 (digits, 9 <= idx < 18): elongated downward only (y_half_down > y_half_up)
          - every text-grid row (idx >= 18): elongated both up and down equally
        See RoiFinder.set_rois for exactly how these are derived.
        """
        self.center = center  # (x, y)
        self.x_half_left = x_half_left
        self.x_half_right = x_half_right
        self.y_half_up = y_half_up
        self.y_half_down = y_half_down
        self.idx = idx
        self.symbol = symbol if symbol is not None else None

    @property
    def radius(self):
        """Backward-compat alias for callers that only draw/measure one size
        (SearchFinder.py's visualizer, build_roi_time_*.py) - the average of the plain,
        un-elongated left/right half-widths, the closest equivalent to the old single-radius
        concept."""
        return (self.x_half_left + self.x_half_right) / 2.0

    def contains(self, point):
        """Is `point` inside this ROI's box - the one, sole definition of containment
        (no factor/shape arguments any more - see ROI_ELONGATION_RATIO)."""
        px, py = point
        cx, cy = self.center
        return (cx - self.x_half_left <= px <= cx + self.x_half_right) and \
               (cy - self.y_half_up <= py <= cy + self.y_half_down)

    def dist_and_angle(self, point: tuple[float, float]):
        """Calculate distance and angle from ROI center to a given point."""
        px, py = point
        cx, cy = self.center
        dx = px - cx
        dy = py - cy
        distance = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)  # radians
        return distance, angle


class RoiFinder:
    def __init__(self, panel_name: str, img):
        self.panel_name = panel_name
        self.img = img
        self.rows = self._find_centers()
        self.centers = [c for row in self.rows for c in row]
        self.rois: List[ROI] = self.set_rois()

    def _find_centers(self):
        """
        Find centers of black framed squares in a white image while in each square there is
        a black symbol. Returns a list of rows (top to bottom), each row a list of
        (x, y, h) in reading order (left to right) - h is the DETECTED box's own height
        (from cv2.boundingRect), kept alongside the center so set_rois can define each
        ROI's vertical "radius" from the symbol's own enclosing square, per 2026-09-18
        decision, rather than from spacing to a neighboring row.
        """
        if self.img is None:
            raise ValueError(f"No image provided")

        if len(self.img.shape) == 3 and self.img.shape[2] == 3:
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = self.img

        # Apply threshold to ensure pure black and white
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

        # Try line detection approach first
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 50:
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 300 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0 and w > 10 and h > 10:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    centers.append((center_x, center_y, h))

        # Remove duplicates that are too close (raw order at this point - not yet in
        # reading order, that's handled by _group_into_rows below)
        filtered_centers = []
        min_distance = 60
        for center in centers:
            too_close = any(
                abs(center[0] - existing[0]) < min_distance and
                abs(center[1] - existing[1]) < min_distance
                for existing in filtered_centers
            )
            if not too_close:
                filtered_centers.append(center)

        return self._group_into_rows(filtered_centers)

    @staticmethod
    def _adaptive_row_gap_threshold(sorted_gaps):
        """Find the y-gap value separating 'jitter within one visual row' from 'a genuine
        gap between rows', as the split point with the single largest jump in the sorted
        gap values - not a fixed pixel constant. A fixed threshold (or the previous
        `y // row_height` bucketing) can silently misclassify a row whose detection jitter
        (a few px) happens to straddle a bucket boundary, scrambling that row's reading
        order (found 2026-09-17: participant AA562 panel a3, ROI indices 123/124/125 ended
        up out of true left-to-right position because 3 of that row's 15 centers landed a
        few px on the other side of a `// 80` boundary from their 12 row-mates)."""
        if len(sorted_gaps) < 2:
            return sorted_gaps[0] / 2 if sorted_gaps else 1
        jumps = [(sorted_gaps[i + 1] - sorted_gaps[i], i) for i in range(len(sorted_gaps) - 1)]
        _, split_i = max(jumps)
        return (sorted_gaps[split_i] + sorted_gaps[split_i + 1]) / 2

    @classmethod
    def _group_into_rows(cls, centers):
        """Cluster (x, y) centers into visual rows (top to bottom, each row left to right),
        using the adaptive gap threshold above instead of a fixed-size bucket."""
        if not centers:
            return []
        pts = sorted(centers, key=lambda p: p[1])
        ys = [p[1] for p in pts]
        gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        if not gaps:
            return [pts]
        threshold = cls._adaptive_row_gap_threshold(sorted(gaps))

        rows = [[pts[0]]]
        for i in range(1, len(pts)):
            if ys[i] - ys[i - 1] > threshold:
                rows.append([])
            rows[-1].append(pts[i])
        for row in rows:
            row.sort(key=lambda p: p[0])
        return rows

    @staticmethod
    def _half_x_left_right(row):
        """Per-center (half-gap-to-left, half-gap-to-right) within one row, so every
        adjacent pair's edges meet EXACTLY (right edge of one == left edge of the next),
        regardless of small pixel-to-pixel jitter in the actual detected spacing - a
        shared row-wide value (e.g. the median gap) would leave ~1px gaps/overlaps
        wherever a specific pair's real spacing differs from that median. Edge ROIs (first/
        last in the row) mirror their one real neighbor's half-gap on the open side."""
        xs = sorted(p[0] for p in row)
        n = len(xs)
        if n < 2:
            return [(20.0, 20.0)]
        halves = []
        for i in range(n):
            left_gap = (xs[i] - xs[i - 1]) / 2.0 if i > 0 else None
            right_gap = (xs[i + 1] - xs[i]) / 2.0 if i < n - 1 else None
            if left_gap is None:
                left_gap = right_gap
            if right_gap is None:
                right_gap = left_gap
            halves.append((left_gap, right_gap))
        return halves

    def set_rois(self, elongation_ratio=ROI_ELONGATION_RATIO):
        """
        Builds every ROI's box from each symbol's OWN detected enclosing square (decision
        2026-09-18, replacing the earlier inter-row-gap-derived version): "radius" for a
        given (x, y, h) center is h / 2 - half of THAT symbol's own detected box height
        (h comes from cv2.boundingRect in _find_centers, carried through unchanged by
        _group_into_rows). x_half_left/x_half_right are unchanged - never elongated, half
        the gap to that specific left/right neighbor (see _half_x_left_right) - that part
        was already correct.
          - Legend row 1 (symbols, idx < 9): y_half_down = radius (own height, unelongated),
            y_half_up = radius * elongation_ratio.
          - Legend row 2 (digits, 9 <= idx < 18): the mirror image - y_half_up = radius
            (unelongated), y_half_down = radius * elongation_ratio.
          - Every text-grid row (idx >= 18): y_half_up = y_half_down = radius * elongation_ratio.
        Row1/row2 meeting exactly at the midpoint is no longer forced by construction (it
        was, under the old gap-derived formula) - it now holds only if the two rows'
        detected box heights and center-to-center distance happen to line up; see the
        printed sanity check below.
        """
        rows = self.rows
        if len(rows) < 3:
            raise ValueError(f"Expected at least 3 ROI rows (2 legend + >=1 text), found {len(rows)}")

        row1, row2, text_rows = rows[0], rows[1], rows[2:]
        rois = []
        idx = 0

        x_halves_row1 = self._half_x_left_right(row1)
        for (cx, cy, h), (xl, xr) in zip(row1, x_halves_row1):
            radius = h / 2.0
            rois.append(ROI((cx, cy), xl, xr, radius * elongation_ratio, radius, idx))
            idx += 1

        x_halves_row2 = self._half_x_left_right(row2)
        for (cx, cy, h), (xl, xr) in zip(row2, x_halves_row2):
            radius = h / 2.0
            rois.append(ROI((cx, cy), xl, xr, radius, radius * elongation_ratio, idx))
            idx += 1

        for row in text_rows:
            x_halves = self._half_x_left_right(row)
            for (cx, cy, h), (xl, xr) in zip(row, x_halves):
                radius = h / 2.0
                rois.append(ROI((cx, cy), xl, xr, radius * elongation_ratio, radius * elongation_ratio, idx))
                idx += 1

        self.rois = rois

        # Sanity check (2026-09-18): report, per column, whether row1's bottom edge
        # actually meets row2's top edge - no longer guaranteed by construction now that
        # each row's radius comes from its own detected box height rather than a shared
        # gap-derived value.
        row1_rois = [r for r in rois if r.idx < len(row1)]
        row2_rois = [r for r in rois if len(row1) <= r.idx < len(row1) + len(row2)]
        for r1, r2 in zip(row1_rois, row2_rois):
            bottom1 = r1.center[1] + r1.y_half_down
            top2 = r2.center[1] - r2.y_half_up
            if abs(bottom1 - top2) > 1.0:
                print(f"  [ROI sanity check] panel {self.panel_name}: row1 idx{r1.idx} bottom={bottom1:.1f} "
                      f"does not meet row2 idx{r2.idx} top={top2:.1f} (gap={top2 - bottom1:.1f}px)")

        return rois

    def get_roi_by_position(self, x, y):
        """
        Given an (x, y) position, return the ROI object that contains this point,
        or None if no ROI contains it.
        """
        for roi in self.rois:
            if roi.contains((x, y)):
                return roi
        return None

    def nearest_roi(self, x, y):
        """
        Given an (x, y) position, return the nearest ROI object.
        """
        min_dist = float('inf')
        nearest_roi = None

        for roi in self.rois:
            cx, cy = roi.center
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_roi = roi

        return nearest_roi

    # sanity check
    def visualize_centers(self, save_path=None):
        """
        Visualize the detected centers on the image with large colored circles.
        """
        if self.img is None:
            raise ValueError(f"No image provided")
        if not self.centers:
            raise (ValueError("No ROIs found. Please run _find_centers() first."))

        for i, (x, y, _h) in enumerate(self.centers):
            cv2.circle(self.img, (x, y), 15, (0, 0, 255), -1)
            cv2.circle(self.img, (x, y), 8, (255, 255, 255), -1)
            cv2.putText(self.img, str(i), (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if save_path:
            cv2.imwrite(save_path, self.img)
            print(f"Visualization saved to {save_path}")
        else:
            cv2.imshow('Centers Visualization', self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return self.img

    def visualize_rois(self, save_path=None):
        """
        Sanity-check visualization of the ACTUAL ROI boxes as built by set_rois (not just
        their centers) - draws each ROI's real x_half/y_half_up/y_half_down box and idx,
        straight from this instance's own `self.rois`, so what's drawn is exactly what
        `contains()` uses, not a re-derived approximation.
        """
        if self.img is None:
            raise ValueError("No image provided")

        img = self.img.copy()
        for roi in self.rois:
            cx, cy = int(roi.center[0]), int(roi.center[1])
            x1, x2 = int(cx - roi.x_half_left), int(cx + roi.x_half_right)
            y1, y2 = int(cy - roi.y_half_up), int(cy + roi.y_half_down)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(img, str(roi.idx), (cx - 8, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"ROI visualization saved to {save_path}")
        return img

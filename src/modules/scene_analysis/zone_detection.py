"""
Zone Detection Module

Identifies valid regions for vehicle insertion based on depth, segmentation,
and geometric constraints.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import cv2
import logging

from ..utils import BoundingBox, ValidZone

logger = logging.getLogger(__name__)


@dataclass
class ZoneDetectionConfig:
    """Configuration for zone detection."""
    min_zone_size: Tuple[int, int] = (100, 60)  # (width, height) in pixels
    max_zone_size: Tuple[int, int] = (800, 600)
    max_depth_variance: float = 0.15  # Max relative depth variance for flat surface
    margin_from_edge: int = 50  # Pixels from image edge
    min_depth: float = 0.1  # Minimum normalized depth (too close)
    max_depth: float = 0.9  # Maximum normalized depth (too far)
    overlap_threshold: float = 0.3  # Max overlap between zones
    max_zones: int = 10  # Maximum number of zones to return


class ZoneDetector:
    """
    Detects valid zones for vehicle insertion.
    
    A valid zone must:
    - Be on a flat surface (low depth variance)
    - Be large enough to fit a vehicle
    - Not overlap with obstacles
    - Be within reasonable depth range
    """
    
    def __init__(self, config: Optional[ZoneDetectionConfig] = None):
        self.config = config or ZoneDetectionConfig()
    
    def detect_zones(
        self,
        depth_map: np.ndarray,
        placement_mask: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> List[ValidZone]:
        """
        Detect valid insertion zones.
        
        Args:
            depth_map: Normalized depth map (H, W), 0-1
            placement_mask: Binary mask of valid placement areas (H, W)
            image: Optional RGB image for additional analysis
            
        Returns:
            List of ValidZone objects sorted by quality
        """
        h, w = depth_map.shape[:2]
        
        # Apply edge margin
        margin_mask = np.zeros_like(placement_mask)
        m = self.config.margin_from_edge
        margin_mask[m:h-m, m:w-m] = 1
        placement_mask = placement_mask & margin_mask
        
        # Find candidate regions
        candidates = self._find_candidate_regions(placement_mask, depth_map)
        
        # Filter and score zones
        valid_zones = []
        for candidate in candidates:
            zone = self._evaluate_zone(candidate, depth_map, image)
            if zone is not None:
                valid_zones.append(zone)
        
        # Remove overlapping zones
        valid_zones = self._remove_overlapping(valid_zones)
        
        # Sort by confidence and limit
        valid_zones.sort(key=lambda z: z.confidence, reverse=True)
        valid_zones = valid_zones[:self.config.max_zones]
        
        logger.info(f"Detected {len(valid_zones)} valid zones")
        return valid_zones
    
    def _find_candidate_regions(
        self,
        placement_mask: np.ndarray,
        depth_map: np.ndarray
    ) -> List[Dict]:
        """Find candidate regions from placement mask."""
        candidates = []
        
        # Find contours
        mask_uint8 = (placement_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip small contours
            min_area = self.config.min_zone_size[0] * self.config.min_zone_size[1]
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small
            if w < self.config.min_zone_size[0] or h < self.config.min_zone_size[1]:
                continue
            
            # Create mask for this contour
            contour_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)
            
            # Sample sub-regions within large contours
            if w > self.config.max_zone_size[0] * 1.5 or h > self.config.max_zone_size[1] * 1.5:
                sub_candidates = self._sample_subregions(x, y, w, h, contour_mask, depth_map)
                candidates.extend(sub_candidates)
            else:
                candidates.append({
                    "bbox": BoundingBox(x, y, w, h),
                    "mask": contour_mask,
                    "contour": contour
                })
        
        return candidates
    
    def _sample_subregions(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        contour_mask: np.ndarray,
        depth_map: np.ndarray
    ) -> List[Dict]:
        """Sample multiple sub-regions from a large region."""
        candidates = []
        
        # Target size for sub-regions
        target_w = min(w, self.config.max_zone_size[0])
        target_h = min(h, self.config.max_zone_size[1])
        
        # Number of samples
        n_x = max(1, w // target_w)
        n_y = max(1, h // target_h)
        
        step_x = (w - target_w) / max(1, n_x - 1) if n_x > 1 else 0
        step_y = (h - target_h) / max(1, n_y - 1) if n_y > 1 else 0
        
        for i in range(n_x):
            for j in range(n_y):
                sx = x + int(i * step_x)
                sy = y + int(j * step_y)
                
                # Create mask for sub-region
                sub_mask = np.zeros_like(contour_mask)
                sub_mask[sy:sy+target_h, sx:sx+target_w] = 1
                sub_mask = sub_mask & contour_mask
                
                # Skip if sub-region doesn't overlap much with contour
                if sub_mask.sum() < target_w * target_h * 0.5:
                    continue
                
                candidates.append({
                    "bbox": BoundingBox(sx, sy, target_w, target_h),
                    "mask": sub_mask,
                    "contour": None
                })
        
        return candidates
    
    def _evaluate_zone(
        self,
        candidate: Dict,
        depth_map: np.ndarray,
        image: Optional[np.ndarray]
    ) -> Optional[ValidZone]:
        """
        Evaluate a candidate zone and return ValidZone if acceptable.
        """
        bbox = candidate["bbox"]
        mask = candidate["mask"]
        
        # Extract depth values within zone
        zone_depth = depth_map[mask > 0]
        
        if len(zone_depth) == 0:
            return None
        
        # Compute depth statistics
        depth_mean = float(zone_depth.mean())
        depth_std = float(zone_depth.std())
        depth_min = float(zone_depth.min())
        depth_max = float(zone_depth.max())
        
        # Check depth range
        if depth_mean < self.config.min_depth or depth_mean > self.config.max_depth:
            return None
        
        # Check depth variance (flatness)
        depth_range = depth_max - depth_min
        relative_variance = depth_std / (depth_mean + 1e-6)
        
        if relative_variance > self.config.max_depth_variance:
            return None
        
        # Compute confidence score
        # Higher confidence for:
        # - Lower depth variance (flatter)
        # - Larger area
        # - Middle depth range
        
        flatness_score = 1.0 - min(relative_variance / self.config.max_depth_variance, 1.0)
        
        area_ratio = mask.sum() / (bbox.width * bbox.height)
        area_score = min(area_ratio, 1.0)
        
        # Prefer middle distances
        depth_score = 1.0 - abs(depth_mean - 0.5) * 2
        
        confidence = 0.4 * flatness_score + 0.3 * area_score + 0.3 * depth_score
        
        # Determine surface type (simplified)
        surface_type = self._classify_surface(mask, image) if image is not None else "ground"
        
        return ValidZone(
            bbox=bbox,
            mask=mask,
            depth_range=(depth_min, depth_max),
            surface_type=surface_type,
            confidence=confidence,
            metadata={
                "depth_mean": depth_mean,
                "depth_std": depth_std,
                "flatness_score": flatness_score,
                "area_score": area_score
            }
        )
    
    def _classify_surface(self, mask: np.ndarray, image: np.ndarray) -> str:
        """Classify surface type based on color and texture."""
        # Extract region colors
        region = image[mask > 0]
        
        if len(region) == 0:
            return "ground"
        
        # Convert to HSV for better color analysis
        mean_rgb = region.mean(axis=0)
        
        # Simple heuristics
        r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
        
        # Gray/dark = asphalt/road
        if abs(r - g) < 30 and abs(g - b) < 30 and mean_rgb.mean() < 100:
            return "road"
        
        # Brown/tan = dirt/terrain
        if r > g and g > b and r - b > 30:
            return "terrain"
        
        # Green = grass
        if g > r and g > b:
            return "grass"
        
        return "ground"
    
    def _remove_overlapping(self, zones: List[ValidZone]) -> List[ValidZone]:
        """Remove zones that overlap too much."""
        if len(zones) <= 1:
            return zones
        
        # Sort by confidence
        zones.sort(key=lambda z: z.confidence, reverse=True)
        
        kept = []
        for zone in zones:
            # Check overlap with already kept zones
            overlap = False
            for kept_zone in kept:
                iou = self._compute_bbox_iou(zone.bbox, kept_zone.bbox)
                if iou > self.config.overlap_threshold:
                    overlap = True
                    break
            
            if not overlap:
                kept.append(zone)
        
        return kept
    
    def _compute_bbox_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / (union + 1e-6)
    
    def visualize_zones(
        self,
        image: np.ndarray,
        zones: List[ValidZone],
        show_mask: bool = True
    ) -> np.ndarray:
        """
        Visualize detected zones on image.
        
        Args:
            image: RGB image
            zones: List of valid zones
            show_mask: Show zone masks in addition to boxes
            
        Returns:
            Annotated image
        """
        vis = image.copy()
        if vis.dtype == np.float32:
            vis = (vis * 255).astype(np.uint8)
        
        colors = [
            (0, 255, 0),    # Green
            (0, 200, 255),  # Orange
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]
            
            # Draw mask overlay
            if show_mask:
                overlay = vis.copy()
                overlay[zone.mask > 0] = color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            
            # Draw bounding box
            x, y, w, h = zone.bbox.to_xywh()
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{zone.surface_type} ({zone.confidence:.2f})"
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis


def detect_insertion_zones(
    image: np.ndarray,
    depth_map: np.ndarray,
    placement_mask: Optional[np.ndarray] = None
) -> List[ValidZone]:
    """
    Convenience function to detect insertion zones.
    
    Args:
        image: RGB image
        depth_map: Normalized depth map
        placement_mask: Optional placement mask (uses bottom half if not provided)
        
    Returns:
        List of valid insertion zones
    """
    if placement_mask is None:
        h, w = image.shape[:2]
        placement_mask = np.zeros((h, w), dtype=np.uint8)
        placement_mask[h // 3:, :] = 1  # Bottom 2/3 of image
    
    detector = ZoneDetector()
    return detector.detect_zones(depth_map, placement_mask, image)

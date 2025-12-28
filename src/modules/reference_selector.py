"""
Reference Image Selector

Selects the best matching pre-rendered vehicle view based on:
1. Estimated camera viewpoint from background image
2. Target insertion zone depth/position
3. Available rendered views

This bridges scene analysis → rendering selection → inpainting.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraViewpoint:
    """Estimated camera viewpoint from a background image."""
    azimuth: float  # Horizontal angle (0-360°)
    elevation: float  # Vertical angle (0-90°, 0=level, 90=top-down)
    confidence: float  # How confident we are in the estimate
    
    # Additional info
    horizon_y: Optional[float] = None  # Y position of horizon (0-1)
    vanishing_points: Optional[List[Tuple[float, float]]] = None


@dataclass  
class RenderedView:
    """A pre-rendered vehicle view."""
    path: Path
    azimuth: float
    elevation: float
    distance: float
    mask_path: Optional[Path] = None
    
    def angle_distance(self, target_az: float, target_el: float) -> float:
        """Calculate angular distance to target viewpoint."""
        # Handle azimuth wrap-around (0° and 360° are the same)
        az_diff = abs(self.azimuth - target_az)
        az_diff = min(az_diff, 360 - az_diff)
        
        el_diff = abs(self.elevation - target_el)
        
        # Weighted combination (azimuth matters more for vehicles)
        return np.sqrt((az_diff * 1.0) ** 2 + (el_diff * 1.5) ** 2)


class ViewpointEstimator:
    """Estimate camera viewpoint from background image."""
    
    def __init__(self):
        self.default_elevation = 15.0  # Typical camera height
    
    def estimate_from_horizon(self, image: np.ndarray) -> CameraViewpoint:
        """
        Estimate viewpoint from horizon line position.
        
        Horizon high in image = looking down (high elevation)
        Horizon in middle = level view (low elevation)
        Horizon low/not visible = looking up or very high elevation
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough lines to find horizon
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=w//4, maxLineGap=20)
        
        horizon_y = 0.5  # Default to middle
        confidence = 0.3
        
        if lines is not None:
            # Find most horizontal lines
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2-y1, x2-x1))
                # Nearly horizontal (within 10 degrees)
                if angle < np.radians(10) or angle > np.radians(170):
                    horizontal_lines.append((y1 + y2) / 2)
            
            if horizontal_lines:
                # Cluster horizontal lines to find horizon
                horizon_y = np.median(horizontal_lines) / h
                confidence = min(0.8, 0.3 + 0.1 * len(horizontal_lines))
        
        # Convert horizon position to elevation
        # Horizon at 0.5 (middle) = elevation ~15°
        # Horizon at 0.3 (upper) = elevation ~30°
        # Horizon at 0.7 (lower) = elevation ~5°
        elevation = 15 + (0.5 - horizon_y) * 50
        elevation = np.clip(elevation, 5, 60)
        
        return CameraViewpoint(
            azimuth=0.0,  # Can't determine from horizon alone
            elevation=elevation,
            confidence=confidence,
            horizon_y=horizon_y
        )
    
    def estimate_from_vanishing_points(self, image: np.ndarray) -> CameraViewpoint:
        """
        Estimate viewpoint from vanishing points.
        
        This is more complex but can give azimuth information for roads/buildings.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            return self.estimate_from_horizon(image)
        
        # Find vanishing point (intersection of lines)
        # Simplified: assume road/path creates vanishing point
        
        # Group lines by angle
        line_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1)
            line_angles.append((angle, line[0]))
        
        # Find dominant non-horizontal directions (potential road edges)
        angles = [a for a, _ in line_angles]
        
        # Simple vanishing point estimation
        # If vanishing point is left of center: looking slightly right (positive azimuth)
        # If vanishing point is right of center: looking slightly left (negative azimuth)
        
        vp_x = w / 2  # Default to center
        vp_y = h * 0.4  # Slightly above center
        
        # Try to find intersection of converging lines
        converging_lines = [(a, l) for a, l in line_angles 
                           if np.radians(20) < abs(a) < np.radians(160)]
        
        if len(converging_lines) >= 2:
            # Find intersection points
            intersections = []
            for i, (a1, l1) in enumerate(converging_lines):
                for a2, l2 in converging_lines[i+1:]:
                    # Skip if angles too similar
                    if abs(a1 - a2) < np.radians(15):
                        continue
                    
                    pt = self._line_intersection(l1, l2)
                    if pt is not None:
                        px, py = pt
                        # Valid if within extended frame
                        if -w < px < 2*w and -h < py < 2*h:
                            intersections.append(pt)
            
            if intersections:
                # Use median intersection point
                vp_x = np.median([p[0] for p in intersections])
                vp_y = np.median([p[1] for p in intersections])
        
        # Convert vanishing point to angles
        # VP left of center = positive azimuth (camera looking right)
        azimuth = (vp_x - w/2) / w * 30  # ±15° range
        azimuth = np.clip(azimuth, -45, 45)
        
        # VP above center = looking up (negative elevation offset)
        elevation = 15 + (h/2 - vp_y) / h * 30
        elevation = np.clip(elevation, 5, 60)
        
        return CameraViewpoint(
            azimuth=azimuth,
            elevation=elevation,
            confidence=0.5,
            horizon_y=vp_y / h,
            vanishing_points=[(vp_x, vp_y)]
        )
    
    def _line_intersection(self, l1, l2) -> Optional[Tuple[float, float]]:
        """Find intersection point of two lines."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        return (px, py)
    
    def estimate(self, image: np.ndarray, method: str = "auto") -> CameraViewpoint:
        """
        Estimate camera viewpoint from image.
        
        Args:
            image: Background image (RGB)
            method: "horizon", "vanishing", or "auto"
        """
        if method == "horizon":
            return self.estimate_from_horizon(image)
        elif method == "vanishing":
            return self.estimate_from_vanishing_points(image)
        else:
            # Try vanishing points first, fall back to horizon
            vp_result = self.estimate_from_vanishing_points(image)
            if vp_result.confidence > 0.4:
                return vp_result
            return self.estimate_from_horizon(image)


class ReferenceSelector:
    """Select best matching rendered view for a background image."""
    
    def __init__(self, renders_dir: str):
        """
        Args:
            renders_dir: Directory containing rendered views with metadata.json
        """
        self.renders_dir = Path(renders_dir)
        self.views = self._load_views()
        self.viewpoint_estimator = ViewpointEstimator()
    
    def _load_views(self) -> List[RenderedView]:
        """Load available rendered views from metadata."""
        metadata_path = self.renders_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"No metadata.json found in {self.renders_dir}")
            return self._scan_for_views()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        views = []
        for view_data in metadata.get('views', []):
            rgb_path = self.renders_dir / view_data.get('rgb_path', '')
            mask_path = self.renders_dir / view_data.get('mask_path', '')
            
            if rgb_path.exists():
                views.append(RenderedView(
                    path=rgb_path,
                    azimuth=view_data.get('azimuth', 0),
                    elevation=view_data.get('elevation', 30),
                    distance=view_data.get('distance', 10),
                    mask_path=mask_path if mask_path.exists() else None
                ))
        
        logger.info(f"Loaded {len(views)} rendered views from {self.renders_dir}")
        return views
    
    def _scan_for_views(self) -> List[RenderedView]:
        """Scan directory for view images if no metadata exists."""
        views = []
        
        # Look for view_azXXX_elXX pattern
        rgb_dir = self.renders_dir / "rgb"
        if not rgb_dir.exists():
            rgb_dir = self.renders_dir
        
        for img_path in rgb_dir.glob("view_az*_el*.png"):
            # Parse filename: view_az030_el15.png
            name = img_path.stem
            try:
                parts = name.split('_')
                azimuth = float(parts[1].replace('az', ''))
                elevation = float(parts[2].replace('el', '').split('_')[0])
                
                mask_path = self.renders_dir / "mask" / img_path.name
                
                views.append(RenderedView(
                    path=img_path,
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=10.0,
                    mask_path=mask_path if mask_path.exists() else None
                ))
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse view filename: {img_path.name}")
        
        logger.info(f"Scanned {len(views)} rendered views")
        return views
    
    def select_best_view(
        self,
        background: np.ndarray,
        target_azimuth: Optional[float] = None,
        target_elevation: Optional[float] = None,
        zone_position: Optional[Tuple[float, float]] = None
    ) -> RenderedView:
        """
        Select the best matching rendered view for a background.
        
        Args:
            background: Background image
            target_azimuth: Override azimuth (degrees), or None to estimate
            target_elevation: Override elevation (degrees), or None to estimate
            zone_position: (x, y) normalized position of insertion zone
                          Used to adjust azimuth (left side = show right side of vehicle)
        
        Returns:
            Best matching RenderedView
        """
        if not self.views:
            raise ValueError("No rendered views available")
        
        # Estimate viewpoint if not provided
        if target_azimuth is None or target_elevation is None:
            viewpoint = self.viewpoint_estimator.estimate(background)
            logger.info(f"Estimated viewpoint: az={viewpoint.azimuth:.1f}°, el={viewpoint.elevation:.1f}° (conf={viewpoint.confidence:.2f})")
            
            if target_azimuth is None:
                target_azimuth = viewpoint.azimuth
            if target_elevation is None:
                target_elevation = viewpoint.elevation
        
        # Adjust azimuth based on zone position
        # If inserting on left side of image, show vehicle's right side (add to azimuth)
        if zone_position is not None:
            x_pos, y_pos = zone_position
            azimuth_offset = (x_pos - 0.5) * 30  # ±15° adjustment
            target_azimuth += azimuth_offset
        
        # Normalize azimuth to 0-360
        target_azimuth = target_azimuth % 360
        
        logger.info(f"Searching for view: az={target_azimuth:.1f}°, el={target_elevation:.1f}°")
        
        # Find closest matching view
        best_view = min(self.views, key=lambda v: v.angle_distance(target_azimuth, target_elevation))
        
        distance = best_view.angle_distance(target_azimuth, target_elevation)
        logger.info(f"Selected: {best_view.path.name} (az={best_view.azimuth}°, el={best_view.elevation}°, dist={distance:.1f})")
        
        return best_view
    
    def select_view_by_angle(
        self,
        azimuth: float,
        elevation: float
    ) -> RenderedView:
        """Select view by explicit angles."""
        azimuth = azimuth % 360
        best_view = min(self.views, key=lambda v: v.angle_distance(azimuth, elevation))
        return best_view
    
    def get_all_views(self) -> List[RenderedView]:
        """Get all available views."""
        return self.views.copy()
    
    def load_reference_image(self, view: RenderedView) -> np.ndarray:
        """Load the reference image for a view."""
        img = cv2.imread(str(view.path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {view.path}")
        
        # Convert BGR(A) to RGB(A)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def load_reference_mask(self, view: RenderedView) -> Optional[np.ndarray]:
        """Load the mask for a view."""
        if view.mask_path is None or not view.mask_path.exists():
            return None
        
        mask = cv2.imread(str(view.mask_path), cv2.IMREAD_GRAYSCALE)
        return mask


def select_reference_for_insertion(
    background: np.ndarray,
    renders_dir: str,
    insertion_zone: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to select and load reference image.
    
    Args:
        background: Background image
        renders_dir: Directory with rendered views
        insertion_zone: (x, y, width, height) of insertion zone
        
    Returns:
        (reference_rgb, reference_mask, metadata)
    """
    selector = ReferenceSelector(renders_dir)
    
    # Calculate zone center position (normalized)
    zone_position = None
    if insertion_zone is not None:
        x, y, w, h = insertion_zone
        bg_h, bg_w = background.shape[:2]
        zone_position = ((x + w/2) / bg_w, (y + h/2) / bg_h)
    
    # Select best view
    view = selector.select_best_view(background, zone_position=zone_position)
    
    # Load images
    reference_rgb = selector.load_reference_image(view)
    reference_mask = selector.load_reference_mask(view)
    
    if reference_mask is None:
        # Create mask from alpha channel
        if reference_rgb.shape[-1] == 4:
            reference_mask = reference_rgb[:, :, 3]
        else:
            reference_mask = np.ones(reference_rgb.shape[:2], dtype=np.uint8) * 255
    
    metadata = {
        'view_name': view.path.stem,
        'azimuth': view.azimuth,
        'elevation': view.elevation,
        'path': str(view.path)
    }
    
    return reference_rgb, reference_mask, metadata


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python reference_selector.py <background.jpg> <renders_dir>")
        sys.exit(1)
    
    bg_path = sys.argv[1]
    renders_dir = sys.argv[2]
    
    background = cv2.imread(bg_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    ref_rgb, ref_mask, metadata = select_reference_for_insertion(background, renders_dir)
    
    print(f"Selected reference: {metadata}")
    print(f"Reference shape: {ref_rgb.shape}")
    print(f"Mask shape: {ref_mask.shape}")
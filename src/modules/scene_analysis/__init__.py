"""
Scene Analysis Module

Analyzes background images to identify valid zones for vehicle insertion.
Includes depth estimation, segmentation, and zone detection.
"""

from .depth_estimation import (
    DepthEstimator,
    DepthEstimationConfig,
    estimate_depth,
    depth_to_colormap,
    get_depth_statistics
)

from .segmentation import (
    SegmentationConfig,
    SegmentationMask,
    SAMSegmenter,
    SceneSegmenter
)

from .zone_detection import (
    ZoneDetector,
    ZoneDetectionConfig,
    detect_insertion_zones
)

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SceneAnalysisResult:
    """Complete scene analysis result."""
    depth_map: np.ndarray
    placement_mask: np.ndarray
    valid_zones: List
    segmentation_masks: List[SegmentationMask]
    lighting_info: Dict
    metadata: Dict


class SceneAnalyzer:
    """
    Complete scene analysis pipeline.
    
    Combines depth estimation, segmentation, and zone detection
    to identify valid regions for vehicle insertion.
    """
    
    def __init__(
        self,
        depth_config: Optional[DepthEstimationConfig] = None,
        segmentation_config: Optional[SegmentationConfig] = None,
        zone_config: Optional[ZoneDetectionConfig] = None
    ):
        self.depth_estimator = DepthEstimator(depth_config)
        self.segmenter = SceneSegmenter(segmentation_config)
        self.zone_detector = ZoneDetector(zone_config)
    
    def analyze(self, image: np.ndarray) -> SceneAnalysisResult:
        """
        Perform complete scene analysis.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            SceneAnalysisResult with all analysis components
        """
        logger.info("Starting scene analysis")
        
        # Step 1: Depth estimation
        logger.debug("Estimating depth")
        depth_map = self.depth_estimator.estimate(image)
        
        # Step 2: Segmentation and placement mask
        logger.debug("Segmenting scene")
        placement_mask = self.segmenter.get_placement_mask(image, depth_map)
        
        # Get full segmentation for metadata
        scene_analysis = self.segmenter.analyze_scene(image, depth_map)
        
        # Step 3: Zone detection
        logger.debug("Detecting insertion zones")
        valid_zones = self.zone_detector.detect_zones(depth_map, placement_mask, image)
        
        # Step 4: Lighting analysis
        lighting_info = self._analyze_lighting(image)
        
        logger.info(f"Scene analysis complete. Found {len(valid_zones)} valid zones")
        
        return SceneAnalysisResult(
            depth_map=depth_map,
            placement_mask=placement_mask,
            valid_zones=valid_zones,
            segmentation_masks=scene_analysis.get("masks", []),
            lighting_info=lighting_info,
            metadata={
                "image_shape": image.shape,
                "num_segments": scene_analysis.get("num_segments", 0),
                "depth_stats": get_depth_statistics(depth_map)
            }
        )
    
    def _analyze_lighting(self, image: np.ndarray) -> Dict:
        """
        Analyze scene lighting.
        
        Simple analysis based on image brightness and gradients.
        """
        import cv2
        
        # Convert to grayscale
        if image.ndim == 3:
            if image.dtype == np.float32:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32)
        
        # Overall brightness
        mean_brightness = float(gray.mean())
        
        # Brightness gradient (for light direction estimation)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
        
        # Average gradient direction (normalized)
        avg_gx = float(grad_x.mean())
        avg_gy = float(grad_y.mean())
        
        length = np.sqrt(avg_gx**2 + avg_gy**2) + 1e-8
        light_direction = (avg_gx / length, avg_gy / length)
        
        # Estimate color temperature from upper region (sky/ambient)
        h = gray.shape[0]
        upper_region = image[:h//4] if image.ndim == 3 else gray[:h//4]
        
        if image.ndim == 3:
            mean_color = upper_region.mean(axis=(0, 1))
            # Simple color temperature estimate
            if mean_color[0] > mean_color[2]:  # More red than blue
                color_temp = 3000  # Warm
            elif mean_color[2] > mean_color[0]:  # More blue
                color_temp = 7000  # Cool
            else:
                color_temp = 5500  # Neutral
        else:
            color_temp = 5500
        
        return {
            "brightness": mean_brightness / 255.0,
            "light_direction": light_direction,
            "color_temperature": color_temp,
            "intensity": mean_brightness / 255.0
        }
    
    def visualize(
        self,
        image: np.ndarray,
        result: SceneAnalysisResult,
        show_depth: bool = True,
        show_zones: bool = True
    ) -> np.ndarray:
        """
        Create visualization of analysis results.
        """
        import cv2
        
        vis_images = []
        
        # Original
        if image.dtype == np.float32:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image
        vis_images.append(("Original", image_uint8))
        
        # Depth
        if show_depth:
            depth_vis = depth_to_colormap(result.depth_map)
            vis_images.append(("Depth", depth_vis))
        
        # Zones
        if show_zones:
            zones_vis = self.zone_detector.visualize_zones(
                image_uint8, result.valid_zones
            )
            vis_images.append(("Zones", zones_vis))
        
        # Combine visualizations
        if len(vis_images) == 1:
            return vis_images[0][1]
        
        # Create side-by-side visualization
        h, w = image.shape[:2]
        n = len(vis_images)
        
        # Resize all to same height
        resized = []
        for name, img in vis_images:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            resized.append(img)
        
        combined = np.hstack(resized)
        
        return combined


__all__ = [
    # Depth estimation
    'DepthEstimator',
    'DepthEstimationConfig',
    'estimate_depth',
    'depth_to_colormap',
    'get_depth_statistics',
    
    # Segmentation
    'SegmentationConfig',
    'SegmentationMask',
    'SAMSegmenter',
    'SceneSegmenter',
    
    # Zone detection
    'ZoneDetector',
    'ZoneDetectionConfig',
    'detect_insertion_zones',
    
    # High-level
    'SceneAnalyzer',
    'SceneAnalysisResult'
]

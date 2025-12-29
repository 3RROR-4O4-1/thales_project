"""
Scene Analysis Module

Provides tools for analyzing background images:
- Depth estimation (Depth Anything V2)
- Semantic segmentation (DeepLabV3)
- Road and obstacle detection
- Valid placement zone detection

Usage:
    from modules.scene_analysis import (
        # Depth
        estimate_depth,
        depth_to_colormap,
        
        # Segmentation
        segment_image,
        detect_road,
        detect_obstacles,
        get_placement_mask,
        SegmentationResult,
        
        # Zone detection
        ZoneDetector,
        ZoneDetectionConfig,
        detect_insertion_zones,
    )
"""

from .depth_estimation import (
    estimate_depth,
    depth_to_colormap,
)

from .segmentation import (
    segment_image,
    detect_road,
    detect_obstacles,
    get_placement_mask,
    SegmentationResult,
    segmentation_to_colormap,
    create_overlay,
    save_segmentation_visualization,
    PASCAL_VOC_CLASSES,
    PASCAL_VOC_COLORS,
)

from .zone_detection import (
    ZoneDetector,
    ZoneDetectionConfig,
    detect_insertion_zones,
)

__all__ = [
    # Depth estimation
    "estimate_depth",
    "depth_to_colormap",
    
    # Segmentation
    "segment_image",
    "detect_road",
    "detect_obstacles",
    "get_placement_mask",
    "SegmentationResult",
    "segmentation_to_colormap",
    "create_overlay",
    "save_segmentation_visualization",
    "PASCAL_VOC_CLASSES",
    "PASCAL_VOC_COLORS",
    
    # Zone detection
    "ZoneDetector",
    "ZoneDetectionConfig",
    "detect_insertion_zones",
]
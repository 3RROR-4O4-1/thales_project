#!/usr/bin/env python3
"""
Zone Detection Script

Runs the full zone detection pipeline:
1. Depth estimation (Depth Anything V2)
2. Semantic segmentation (DeepLabV3)
3. Zone detection (combining depth + segmentation)

Outputs visualizations at each step.

Usage:
    python run_zone_detection.py <image_path> [--output-dir <dir>]
    
    # From src/ directory:
    python scripts/run_zone_detection.py assets/backgrounds/test.jpg -o outputs/zones
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import logging
import sys
from typing import List, Tuple

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # src/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project imports
from modules.scene_analysis.depth_estimation import estimate_depth, depth_to_colormap
from modules.scene_analysis.segmentation import (
    segment_image,
    SegmentationResult,
    create_overlay,
)
from modules.scene_analysis.zone_detection import (
    ZoneDetector,
    ZoneDetectionConfig,
)
from modules.utils import ValidZone

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def save_original(image: np.ndarray, output_dir: Path) -> Path:
    """Save original image."""
    path = output_dir / "1_original.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


def save_depth_map(depth_map: np.ndarray, output_dir: Path) -> Path:
    """Save depth map visualization."""
    depth_colored = depth_to_colormap(depth_map, colormap="turbo")
    path = output_dir / "2_depth_map.png"
    cv2.imwrite(str(path), cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    
    # Also save raw depth as .npy
    npy_path = output_dir / "2_depth_map.npy"
    np.save(npy_path, depth_map)
    logger.info(f"Saved: {npy_path}")
    
    return path


def save_segmentation(
    image: np.ndarray,
    seg_result: SegmentationResult,
    output_dir: Path
) -> Path:
    """Save segmentation visualization."""
    seg_colored = seg_result.to_colormap()
    overlay = cv2.addWeighted(image, 0.5, seg_colored, 0.5, 0)
    
    path = output_dir / "3_segmentation.png"
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


def save_road_mask(
    image: np.ndarray,
    seg_result: SegmentationResult,
    output_dir: Path
) -> Path:
    """Save road detection visualization."""
    road_overlay = create_overlay(image, seg_result.road_mask, (0, 255, 0), 0.4)
    
    # Draw contours
    contours, _ = cv2.findContours(
        seg_result.road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(road_overlay, contours, -1, (0, 200, 0), 2)
    
    path = output_dir / "4_road_detection.png"
    cv2.imwrite(str(path), cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


def save_obstacle_mask(
    image: np.ndarray,
    seg_result: SegmentationResult,
    output_dir: Path
) -> Path:
    """Save obstacle detection visualization."""
    obstacle_overlay = create_overlay(image, seg_result.obstacle_mask, (255, 0, 0), 0.4)
    
    path = output_dir / "5_obstacle_detection.png"
    cv2.imwrite(str(path), cv2.cvtColor(obstacle_overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


def save_placement_mask(
    image: np.ndarray,
    placement_mask: np.ndarray,
    output_dir: Path
) -> Path:
    """Save placement mask visualization."""
    placement_overlay = create_overlay(image, placement_mask, (0, 255, 255), 0.4)
    
    path = output_dir / "6_placement_mask.png"
    cv2.imwrite(str(path), cv2.cvtColor(placement_overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


def save_valid_zones(
    image: np.ndarray,
    zones: List[ValidZone],
    output_dir: Path
) -> Path:
    """Save valid zones visualization."""
    vis = image.copy()
    
    colors = [
        (0, 255, 0),    # Green
        (0, 200, 255),  # Orange
        (255, 100, 100),# Light blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    for i, zone in enumerate(zones):
        color = colors[i % len(colors)]
        
        # Draw mask overlay
        overlay = vis.copy()
        overlay[zone.mask > 0] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        # Draw bounding box
        x, y, w, h = zone.bbox.x, zone.bbox.y, zone.bbox.width, zone.bbox.height
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        label = f"#{i+1} {zone.surface_type} ({zone.confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(vis, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    path = output_dir / "7_valid_zones.png"
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


def save_final_summary(
    image: np.ndarray,
    depth_map: np.ndarray,
    seg_result: SegmentationResult,
    zones: List[ValidZone],
    output_dir: Path
) -> Path:
    """Save final summary with all stages."""
    h, w = image.shape[:2]
    
    # Create 2x2 grid
    summary = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    
    # Panel 1: Original
    summary[:h, :w] = image
    cv2.putText(summary, "1. Original Image", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Panel 2: Depth
    depth_colored = depth_to_colormap(depth_map, colormap="turbo")
    summary[:h, w:] = depth_colored
    cv2.putText(summary, "2. Depth Anything V2", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Panel 3: Segmentation + road
    seg_colored = seg_result.to_colormap()
    seg_overlay = cv2.addWeighted(image, 0.5, seg_colored, 0.5, 0)
    # Add road highlight
    road_color = np.zeros_like(seg_overlay)
    road_color[seg_result.road_mask > 0] = (0, 255, 0)
    seg_overlay = cv2.addWeighted(seg_overlay, 0.7, road_color, 0.3, 0)
    summary[h:, :w] = seg_overlay
    cv2.putText(summary, "3. DeepLabV3 + Road (green)", (10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Panel 4: Valid zones
    zones_vis = image.copy()
    zone_colors = [(0, 255, 0), (0, 200, 255), (255, 100, 100), (255, 255, 0), (255, 0, 255)]
    for i, zone in enumerate(zones):
        color = zone_colors[i % len(zone_colors)]
        overlay = zones_vis.copy()
        overlay[zone.mask > 0] = color
        zones_vis = cv2.addWeighted(zones_vis, 0.6, overlay, 0.4, 0)
        x, y, zw, zh = zone.bbox.x, zone.bbox.y, zone.bbox.width, zone.bbox.height
        cv2.rectangle(zones_vis, (x, y), (x + zw, y + zh), color, 3)
        cv2.putText(zones_vis, f"#{i+1} {zone.surface_type}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    summary[h:, w:] = zones_vis
    cv2.putText(summary, f"4. Valid Zones ({len(zones)} found)", (w + 10, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    path = output_dir / "8_summary.png"
    cv2.imwrite(str(path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved: {path}")
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Zone Detection Pipeline")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output-dir", "-o", default="./zone_output",
                        help="Output directory for visualizations")
    parser.add_argument("--depth-model", "-d", default="depth-anything-v2-large",
                        choices=["depth-anything-v2-large", "depth-anything-v2-base", 
                                 "depth-anything-v2-small", "zoedepth", "midas"],
                        help="Depth estimation model")
    parser.add_argument("--seg-model", "-s", default="resnet101",
                        choices=["resnet101", "mobilenet"],
                        help="Segmentation model (resnet101 is better, mobilenet is faster)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Device for inference (auto-detect if not specified)")
    args = parser.parse_args()
    
    # Auto-detect device
    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {args.device}")
    
    # Setup
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return 1
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    logger.info(f"Image size: {w}x{h}")
    
    # Step 1: Save original
    logger.info("=" * 60)
    logger.info("Step 1: Original Image")
    logger.info("=" * 60)
    save_original(image, output_dir)
    
    # Step 2: Depth estimation
    logger.info("=" * 60)
    logger.info("Step 2: Depth Estimation (Depth Anything V2)")
    logger.info("=" * 60)
    depth_map = estimate_depth(image, model=args.depth_model, device=args.device)
    logger.info(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
    save_depth_map(depth_map, output_dir)
    
    # Step 3: Semantic segmentation
    logger.info("=" * 60)
    logger.info("Step 3: Semantic Segmentation (DeepLabV3)")
    logger.info("=" * 60)
    seg_result = segment_image(image, model_type=args.seg_model, device=args.device)
    
    logger.info(f"Detected classes:")
    for class_name, pct in sorted(seg_result.detected_classes.items(), key=lambda x: -x[1]):
        logger.info(f"  {class_name}: {pct:.1f}%")
    
    save_segmentation(image, seg_result, output_dir)
    
    # Step 4: Road detection
    logger.info("=" * 60)
    logger.info("Step 4: Road Detection")
    logger.info("=" * 60)
    road_pct = 100 * seg_result.road_mask.sum() / (h * w)
    logger.info(f"Road area: {road_pct:.1f}%")
    save_road_mask(image, seg_result, output_dir)
    
    # Step 5: Obstacle detection
    logger.info("=" * 60)
    logger.info("Step 5: Obstacle Detection")
    logger.info("=" * 60)
    obstacle_pct = 100 * seg_result.obstacle_mask.sum() / (h * w)
    logger.info(f"Obstacle area: {obstacle_pct:.1f}%")
    save_obstacle_mask(image, seg_result, output_dir)
    
    # Step 6: Placement mask
    logger.info("=" * 60)
    logger.info("Step 6: Placement Mask")
    logger.info("=" * 60)
    placement_mask = seg_result.get_placement_mask(margin=50)
    placement_pct = 100 * placement_mask.sum() / (h * w)
    logger.info(f"Valid placement area: {placement_pct:.1f}%")
    save_placement_mask(image, placement_mask, output_dir)
    
    # Step 7: Zone detection
    logger.info("=" * 60)
    logger.info("Step 7: Zone Detection")
    logger.info("=" * 60)
    
    zone_detector = ZoneDetector(ZoneDetectionConfig())
    zones = zone_detector.detect_zones(depth_map, placement_mask, image)
    
    logger.info(f"Found {len(zones)} valid zones:")
    for i, zone in enumerate(zones):
        logger.info(f"  Zone {i+1}: {zone.surface_type}")
        logger.info(f"    - Confidence: {zone.confidence:.3f}")
        logger.info(f"    - Depth: {zone.metadata.get('depth_mean', 0):.3f}")
        logger.info(f"    - BBox: ({zone.bbox.x}, {zone.bbox.y}, {zone.bbox.width}, {zone.bbox.height})")
    
    save_valid_zones(image, zones, output_dir)
    
    # Step 8: Final summary
    logger.info("=" * 60)
    logger.info("Step 8: Final Summary")
    logger.info("=" * 60)
    save_final_summary(image, depth_map, seg_result, zones, output_dir)
    
    logger.info("=" * 60)
    logger.info("COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    for f in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {f.name}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
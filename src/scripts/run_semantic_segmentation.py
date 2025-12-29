#!/usr/bin/env python3
"""
Semantic Segmentation Script (DeepLabV3)

Runs DeepLabV3 semantic segmentation to detect roads, sidewalks,
buildings, vegetation, obstacles, etc.

Usage:
    python run_semantic_segmentation.py <image_path> [--output-dir <dir>]
    
    # From src/ directory:
    python scripts/run_semantic_segmentation.py assets/backgrounds/test.jpg -o outputs/semantic
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import logging
import sys

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # src/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project imports
from modules.scene_analysis.segmentation import (
    segment_image,
    save_segmentation_visualization,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Semantic Segmentation (DeepLabV3)")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output-dir", "-o", default="./semantic_output",
                        help="Output directory for visualizations")
    parser.add_argument("--model", "-m", default="resnet101",
                        choices=["resnet101", "mobilenet"],
                        help="Model backbone (resnet101 is better, mobilenet is faster)")
    parser.add_argument("--device", "-d", default=None,
                        choices=["cuda", "cpu"],
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
    
    # Run semantic segmentation
    logger.info("=" * 60)
    logger.info("Running Semantic Segmentation (DeepLabV3)")
    logger.info("=" * 60)
    
    result = segment_image(image, model_type=args.model, device=args.device)
    
    # Print detected classes
    logger.info(f"\nDetected classes:")
    for class_name, pct in sorted(result.detected_classes.items(), key=lambda x: -x[1]):
        logger.info(f"  {class_name}: {pct:.1f}%")
    
    # Print road/obstacle info
    road_pct = 100 * result.road_mask.sum() / (h * w)
    obstacle_pct = 100 * result.obstacle_mask.sum() / (h * w)
    placement_mask = result.get_placement_mask()
    placement_pct = 100 * placement_mask.sum() / (h * w)
    
    logger.info(f"\nDerived masks:")
    logger.info(f"  Road area: {road_pct:.1f}%")
    logger.info(f"  Obstacle area: {obstacle_pct:.1f}%")
    logger.info(f"  Valid placement area: {placement_pct:.1f}%")
    
    # Save visualizations
    logger.info("=" * 60)
    logger.info("Saving Visualizations")
    logger.info("=" * 60)
    
    saved_paths = save_segmentation_visualization(image, result, output_dir)
    
    logger.info("=" * 60)
    logger.info("COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Files created:")
    for p in saved_paths:
        logger.info(f"  - {p.name}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
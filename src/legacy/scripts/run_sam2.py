#!/usr/bin/env python3
"""
SAM2 Segmentation Script

Runs SAM2 (Segment Anything Model 2) on an image and outputs
visualizations of all detected segments.

Usage:
    python run_sam2.py <image_path> [--output-dir <dir>]
    
    # From src/ directory:
    python scripts/run_sam2.py assets/backgrounds/test.jpg -o outputs/sam2
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import logging
import sys
from typing import List, Dict, Tuple, Optional
import torch

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # src/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# SAM2 imports - try different import paths
try:
    # If installed via pip or editable install
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    try:
        # Alternative import path
        from sam2.sam2.build_sam import build_sam2
        from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "SAM2 not found. Install with:\n"
            "  git clone https://github.com/facebookresearch/sam2.git\n"
            "  cd sam2 && pip install -e . --no-build-isolation"
        )

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SAM2Segmenter:
    """
    Segment Anything Model 2 (SAM2) for automatic segmentation.
    
    SAM2 provides:
    - Automatic mask generation (segment everything)
    - Point-prompted segmentation
    - Box-prompted segmentation
    """
    
    # Model checkpoints: (config_path, checkpoint_name)
    # Config paths are relative to sam2/configs/ in the package
    CHECKPOINTS = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "base": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt")
    }
    
    # Possible checkpoint locations
    CHECKPOINT_DIRS = [
        Path.home() / ".cache" / "sam2",
        Path.home() / ".cache" / "vehicle_inpainting" / "sam2",
        Path("./checkpoints"),
        Path("../checkpoints"),
    ]
    
    def __init__(self, model_size: str = "large", device: str = None):
        """
        Args:
            model_size: "tiny", "small", "base", or "large"
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mask_generator = None
        self._load_model()
    
    def _find_checkpoint(self) -> Tuple[str, Optional[Path]]:
        """Find the checkpoint file."""
        config_path, checkpoint_name = self.CHECKPOINTS.get(
            self.model_size, self.CHECKPOINTS["large"]
        )
        
        # Search for checkpoint in known locations
        for search_dir in self.CHECKPOINT_DIRS:
            checkpoint_path = search_dir / checkpoint_name
            if checkpoint_path.exists():
                logger.info(f"Found checkpoint: {checkpoint_path}")
                return config_path, checkpoint_path
        
        # If not found, return None for checkpoint (will fail with helpful message)
        logger.error(f"Checkpoint '{checkpoint_name}' not found!")
        logger.error(f"Searched in: {[str(d) for d in self.CHECKPOINT_DIRS]}")
        logger.error(f"Download with: python setup_models.py --component sam2")
        return config_path, None
    
    def _load_model(self):
        """Load SAM2 model."""
        logger.info(f"Loading SAM2 model ({self.model_size}) on {self.device}...")
        
        config_path, checkpoint_path = self._find_checkpoint()
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"SAM2 checkpoint not found. Download with:\n"
                f"  python setup_models.py --component sam2\n"
                f"Or manually download to ~/.cache/sam2/"
            )
        
        self.model = build_sam2(config_path, str(checkpoint_path), device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        logger.info("SAM2 model loaded successfully")
    
    def segment(self, image: np.ndarray) -> List[Dict]:
        """
        Generate automatic segmentation masks.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            List of mask dictionaries with keys:
            - 'segmentation': Binary mask (H, W)
            - 'area': Number of pixels
            - 'bbox': [x, y, w, h]
            - 'predicted_iou': Confidence score
            - 'stability_score': Mask stability
        """
        logger.info("Running SAM2 automatic mask generation...")
        
        masks = self.mask_generator.generate(image)
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        logger.info(f"SAM2 generated {len(masks)} masks")
        return masks


def visualize_masks(
    image: np.ndarray,
    masks: List[Dict],
    max_masks: int = 50
) -> np.ndarray:
    """
    Create visualization of segmentation masks.
    
    Args:
        image: RGB image
        masks: List of SAM2 mask dicts
        max_masks: Maximum number of masks to show
        
    Returns:
        Visualization image with colored masks
    """
    vis = image.copy()
    
    # Generate random colors for each mask
    np.random.seed(42)
    colors = np.random.randint(0, 255, (max_masks, 3), dtype=np.uint8)
    
    # Draw masks (from smallest to largest so large ones are at bottom)
    for i, mask_data in enumerate(reversed(masks[:max_masks])):
        mask = mask_data['segmentation'].astype(np.uint8)
        color = tuple(map(int, colors[i % max_masks]))
        
        # Create colored overlay
        overlay = vis.copy()
        overlay[mask > 0] = color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
    
    return vis


def visualize_masks_grid(
    image: np.ndarray,
    masks: List[Dict],
    grid_size: Tuple[int, int] = (4, 4)
) -> np.ndarray:
    """
    Create a grid showing individual masks.
    
    Args:
        image: RGB image
        masks: List of SAM2 mask dicts
        grid_size: (rows, cols) for the grid
        
    Returns:
        Grid visualization
    """
    h, w = image.shape[:2]
    rows, cols = grid_size
    n_cells = rows * cols
    
    # Calculate cell size
    cell_h = h // rows
    cell_w = w // cols
    
    # Create output grid
    grid = np.zeros((h, w, 3), dtype=np.uint8)
    
    for idx in range(min(n_cells, len(masks))):
        row = idx // cols
        col = idx % cols
        
        mask_data = masks[idx]
        mask = mask_data['segmentation'].astype(np.uint8)
        
        # Create cell image
        cell = image.copy()
        
        # Highlight the mask in green
        overlay = cell.copy()
        overlay[mask > 0] = (0, 255, 0)
        cell = cv2.addWeighted(cell, 0.5, overlay, 0.5, 0)
        
        # Darken areas outside mask
        cell[mask == 0] = (cell[mask == 0] * 0.3).astype(np.uint8)
        
        # Add mask info
        area = mask_data['area']
        iou = mask_data.get('predicted_iou', 0)
        cv2.putText(cell, f"#{idx+1} area:{area}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cell, f"iou:{iou:.2f}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Resize and place in grid
        cell_resized = cv2.resize(cell, (cell_w, cell_h))
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        grid[y1:y2, x1:x2] = cell_resized
    
    return grid


def create_full_visualization(
    image: np.ndarray,
    masks: List[Dict],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create and save all visualizations.
    
    Returns:
        Dict mapping visualization name to file path
    """
    output_paths = {}
    
    # 1. Original image
    orig_path = output_dir / "1_original.png"
    cv2.imwrite(str(orig_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    output_paths['original'] = orig_path
    logger.info(f"Saved: {orig_path}")
    
    # 2. All masks overlay
    all_masks_vis = visualize_masks(image, masks)
    all_masks_path = output_dir / "2_sam2_all_masks.png"
    cv2.imwrite(str(all_masks_path), cv2.cvtColor(all_masks_vis, cv2.COLOR_RGB2BGR))
    output_paths['all_masks'] = all_masks_path
    logger.info(f"Saved: {all_masks_path}")
    
    # 3. Top masks grid
    grid_vis = visualize_masks_grid(image, masks, grid_size=(4, 4))
    grid_path = output_dir / "3_sam2_top_masks_grid.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid_vis, cv2.COLOR_RGB2BGR))
    output_paths['masks_grid'] = grid_path
    logger.info(f"Saved: {grid_path}")
    
    # 4. Summary panel
    h, w = image.shape[:2]
    summary = np.zeros((h, w * 2, 3), dtype=np.uint8)
    summary[:, :w] = image
    summary[:, w:] = all_masks_vis
    
    # Add labels
    cv2.putText(summary, "Original", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(summary, f"SAM2 Segmentation ({len(masks)} masks)", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    summary_path = output_dir / "4_sam2_summary.png"
    cv2.imwrite(str(summary_path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    output_paths['summary'] = summary_path
    logger.info(f"Saved: {summary_path}")
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 Segmentation")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output-dir", "-o", default="./sam2_output",
                        help="Output directory for visualizations")
    parser.add_argument("--model-size", "-m", default="base",
                        choices=["tiny", "small", "base", "large"],
                        help="SAM2 model size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for inference")
    args = parser.parse_args()
    
    # Setup
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run SAM2
    logger.info("=" * 60)
    logger.info("Running SAM2 Segmentation")
    logger.info("=" * 60)
    
    segmenter = SAM2Segmenter(model_size=args.model_size, device=args.device)
    masks = segmenter.segment(image)
    
    # Print mask statistics
    logger.info(f"\nMask Statistics:")
    logger.info(f"  Total masks: {len(masks)}")
    if masks:
        areas = [m['area'] for m in masks]
        logger.info(f"  Largest mask: {max(areas):,} pixels")
        logger.info(f"  Smallest mask: {min(areas):,} pixels")
        logger.info(f"  Average mask: {np.mean(areas):,.0f} pixels")
    
    # Create visualizations
    logger.info("=" * 60)
    logger.info("Creating Visualizations")
    logger.info("=" * 60)
    
    output_paths = create_full_visualization(image, masks, output_dir)
    
    logger.info("=" * 60)
    logger.info("COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    return output_paths


if __name__ == "__main__":
    main()



# python scripts/run_sam2.py assets/backgrounds/test.jpg -o outputs/sam2 --device cpu -m large

""" 
wget -O ~/.cache/sam2/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
"""
# python scripts/run_sam2.py assets/backgrounds/test.jpg -o outputs/sam2 --device cpu -m base

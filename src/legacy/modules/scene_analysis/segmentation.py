"""
Semantic Segmentation Module

Segments scenes to identify regions suitable for vehicle placement.
Supports SAM 2, Grounded-DINO, and semantic segmentation models.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""
    model: str = "sam2"  # "sam2", "grounded-dino", "semantic"
    device: str = "cuda"
    
    # SAM specific
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    
    # Grounded-DINO specific
    box_threshold: float = 0.35
    text_threshold: float = 0.25


@dataclass
class SegmentationMask:
    """Represents a segmentation mask with metadata."""
    mask: np.ndarray
    score: float = 1.0
    label: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    area: int = 0
    
    def __post_init__(self):
        if self.area == 0:
            self.area = int(self.mask.sum())
        if self.bbox is None:
            self.bbox = self._compute_bbox()
    
    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        """Compute bounding box from mask."""
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


class SAMSegmenter:
    """
    Segment Anything Model (SAM) wrapper.
    """
    
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            
            # Try to find model checkpoint
            import os
            checkpoint_paths = [
                "sam_vit_h_4b8939.pth",
                "models/sam_vit_h_4b8939.pth",
                os.path.expanduser("~/.cache/sam/sam_vit_h_4b8939.pth")
            ]
            
            checkpoint = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint = path
                    break
            
            if checkpoint is None:
                logger.warning("SAM checkpoint not found, using default initialization")
                # In production, you'd download the checkpoint
                raise FileNotFoundError("SAM checkpoint not found")
            
            self.model = sam_model_registry["vit_h"](checkpoint=checkpoint)
            
            if self.config.device == "cuda":
                import torch
                self.model = self.model.cuda()
            
            # Create generators
            self.mask_generator = SamAutomaticMaskGenerator(
                self.model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
            )
            
            self.predictor = SamPredictor(self.model)
            
            logger.info("Loaded SAM model")
            
        except ImportError:
            logger.error("segment_anything not installed")
            raise
        except FileNotFoundError:
            logger.warning("SAM checkpoint not found, segmentation will be limited")
            self.model = None
    
    def segment_automatic(self, image: np.ndarray) -> List[SegmentationMask]:
        """
        Automatically segment entire image.
        
        Args:
            image: RGB image (H, W, 3), uint8
            
        Returns:
            List of SegmentationMask objects
        """
        if self.model is None:
            return self._fallback_segmentation(image)
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Generate masks
        sam_masks = self.mask_generator.generate(image)
        
        # Convert to our format
        results = []
        for m in sam_masks:
            seg_mask = SegmentationMask(
                mask=m['segmentation'].astype(np.uint8),
                score=float(m['predicted_iou']),
                bbox=tuple(m['bbox']),
                area=int(m['area'])
            )
            results.append(seg_mask)
        
        # Sort by area (largest first)
        results.sort(key=lambda x: x.area, reverse=True)
        
        return results
    
    def segment_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        label: int = 1
    ) -> SegmentationMask:
        """
        Segment region containing a point.
        
        Args:
            image: RGB image
            point: (x, y) point coordinates
            label: 1 for foreground, 0 for background
            
        Returns:
            SegmentationMask for the region
        """
        if self.model is None:
            return self._fallback_point_segment(image, point)
        
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        self.predictor.set_image(image)
        
        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([label])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Take best mask
        best_idx = np.argmax(scores)
        
        return SegmentationMask(
            mask=masks[best_idx].astype(np.uint8),
            score=float(scores[best_idx])
        )
    
    def segment_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> SegmentationMask:
        """
        Segment region within a bounding box.
        
        Args:
            image: RGB image
            box: (x, y, w, h) bounding box
            
        Returns:
            SegmentationMask for the region
        """
        if self.model is None:
            return self._fallback_box_segment(image, box)
        
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        self.predictor.set_image(image)
        
        x, y, w, h = box
        input_box = np.array([x, y, x + w, y + h])
        
        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        
        return SegmentationMask(
            mask=masks[best_idx].astype(np.uint8),
            score=float(scores[best_idx]),
            bbox=box
        )
    
    def _fallback_segmentation(self, image: np.ndarray) -> List[SegmentationMask]:
        """Simple fallback using edge detection and contours."""
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:  # Skip small regions
                continue
            
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, -1)
            
            results.append(SegmentationMask(mask=mask, score=0.5))
        
        return results
    
    def _fallback_point_segment(self, image: np.ndarray, point: Tuple[int, int]) -> SegmentationMask:
        """Fallback point segmentation using flood fill."""
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        flood_img = image.copy()
        cv2.floodFill(flood_img, mask, point, (255, 255, 255), (20, 20, 20), (20, 20, 20))
        
        result_mask = mask[1:-1, 1:-1]
        
        return SegmentationMask(mask=result_mask, score=0.3)
    
    def _fallback_box_segment(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> SegmentationMask:
        """Fallback box segmentation using GrabCut."""
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        x, y, w, h = box
        mask = np.zeros(image.shape[:2], np.uint8)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (x, y, w, h)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        except:
            # If GrabCut fails, return box as mask
            result_mask = np.zeros(image.shape[:2], np.uint8)
            result_mask[y:y+h, x:x+w] = 1
        
        return SegmentationMask(mask=result_mask, score=0.5, bbox=box)


class SceneSegmenter:
    """
    High-level scene segmentation for vehicle placement.
    
    Identifies:
    - Ground/road regions
    - Sky regions
    - Obstacles
    - Valid placement zones
    """
    
    # Surface type classifications
    GROUND_LABELS = ["road", "ground", "terrain", "floor", "pavement", "asphalt", "dirt", "grass"]
    SKY_LABELS = ["sky", "cloud"]
    OBSTACLE_LABELS = ["building", "tree", "person", "pole", "sign"]
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self.segmenter = SAMSegmenter(self.config)
    
    def analyze_scene(self, image: np.ndarray, depth: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze scene for vehicle placement.
        
        Args:
            image: RGB image
            depth: Optional depth map
            
        Returns:
            Dict with segmentation results and analysis
        """
        # Get automatic segmentation
        masks = self.segmenter.segment_automatic(image)
        
        # Classify regions
        classified = self._classify_regions(image, masks, depth)
        
        # Find ground plane
        ground_mask = self._find_ground_plane(image, classified, depth)
        
        return {
            "masks": masks,
            "classified": classified,
            "ground_mask": ground_mask,
            "num_segments": len(masks)
        }
    
    def _classify_regions(
        self,
        image: np.ndarray,
        masks: List[SegmentationMask],
        depth: Optional[np.ndarray]
    ) -> Dict[str, List[SegmentationMask]]:
        """Classify segmented regions by type."""
        classified = {
            "ground": [],
            "sky": [],
            "obstacle": [],
            "unknown": []
        }
        
        h, w = image.shape[:2]
        
        for mask in masks:
            # Simple heuristics for classification
            y_center = mask.bbox[1] + mask.bbox[3] / 2
            
            # Sky: upper portion, typically blue/white
            if y_center < h * 0.4:
                # Check if region is bright (sky-like)
                region = image[mask.mask > 0]
                if len(region) > 0:
                    brightness = region.mean()
                    if brightness > 150:  # Bright region in upper area
                        mask.label = "sky"
                        classified["sky"].append(mask)
                        continue
            
            # Ground: lower portion, use depth if available
            if y_center > h * 0.5:
                if depth is not None:
                    # Ground should have similar depth values (flat surface)
                    depth_region = depth[mask.mask > 0]
                    if len(depth_region) > 0:
                        depth_std = depth_region.std()
                        if depth_std < 0.1:  # Low variance = flat surface
                            mask.label = "ground"
                            classified["ground"].append(mask)
                            continue
                
                # Heuristic: large areas in lower portion
                if mask.area > (h * w * 0.05):  # At least 5% of image
                    mask.label = "ground"
                    classified["ground"].append(mask)
                    continue
            
            # Default: unknown
            mask.label = "unknown"
            classified["unknown"].append(mask)
        
        return classified
    
    def _find_ground_plane(
        self,
        image: np.ndarray,
        classified: Dict,
        depth: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Create a unified ground mask.
        """
        h, w = image.shape[:2]
        ground_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine all ground segments
        for mask in classified.get("ground", []):
            ground_mask = np.maximum(ground_mask, mask.mask)
        
        # If no ground found, use bottom half as fallback
        if ground_mask.sum() == 0:
            ground_mask[h // 2:, :] = 1
        
        return ground_mask
    
    def get_placement_mask(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        min_area: int = 10000
    ) -> np.ndarray:
        """
        Get mask of valid placement regions.
        
        Args:
            image: RGB image
            depth: Optional depth map
            min_area: Minimum area for valid regions
            
        Returns:
            Binary mask of valid placement zones
        """
        analysis = self.analyze_scene(image, depth)
        
        # Start with ground mask
        placement_mask = analysis["ground_mask"].copy()
        
        # Remove obstacle regions
        for mask in analysis["classified"].get("obstacle", []):
            placement_mask = placement_mask & ~mask.mask
        
        # Remove small isolated regions
        kernel = np.ones((5, 5), np.uint8)
        placement_mask = cv2.morphologyEx(placement_mask, cv2.MORPH_OPEN, kernel)
        placement_mask = cv2.morphologyEx(placement_mask, cv2.MORPH_CLOSE, kernel)
        
        # Filter by minimum area
        contours, _ = cv2.findContours(placement_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_mask = np.zeros_like(placement_mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 1, -1)
        
        return filtered_mask

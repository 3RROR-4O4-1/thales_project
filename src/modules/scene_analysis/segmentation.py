"""
Semantic Segmentation Module

Uses DeepLabV3 for semantic segmentation to detect:
- Roads, sidewalks, terrain
- Vehicles, pedestrians, obstacles
- Buildings, vegetation, sky

This replaces the old SAM2-based segmentation which was better for
instance segmentation but poor at detecting uniform surfaces like roads.

Usage:
    from modules.scene_analysis.segmentation import (
        segment_image,
        detect_road,
        detect_obstacles,
        SegmentationResult,
    )
    
    # Full segmentation
    result = segment_image(image)
    
    # Road detection only
    road_mask = detect_road(image)
    
    # Get obstacles (cars, people, etc.)
    obstacle_mask = detect_obstacles(image)
"""
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PASCAL VOC CLASSES (DeepLabV3 pretrained)
# =============================================================================

PASCAL_VOC_CLASSES = [
    "background",    # 0
    "aeroplane",     # 1
    "bicycle",       # 2
    "bird",          # 3
    "boat",          # 4
    "bottle",        # 5
    "bus",           # 6
    "car",           # 7
    "cat",           # 8
    "chair",         # 9
    "cow",           # 10
    "diningtable",   # 11
    "dog",           # 12
    "horse",         # 13
    "motorbike",     # 14
    "person",        # 15
    "pottedplant",   # 16
    "sheep",         # 17
    "sofa",          # 18
    "train",         # 19
    "tvmonitor",     # 20
]

# RGB colors for visualization
PASCAL_VOC_COLORS = np.array([
    [0, 0, 0],        # background - black
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus - teal
    [128, 128, 128],  # car - gray
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # diningtable
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person - pink
    [0, 64, 0],       # pottedplant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128],     # tvmonitor
], dtype=np.uint8)

# Semantic groupings
VEHICLE_CLASSES = {"car", "bus", "train", "motorbike", "bicycle", "aeroplane", "boat"}
PERSON_CLASSES = {"person"}
ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow"}
OBSTACLE_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES | ANIMAL_CLASSES
FURNITURE_CLASSES = {"chair", "diningtable", "sofa", "tvmonitor", "bottle", "pottedplant"}

# Class ID lookups
VEHICLE_IDS = [i for i, c in enumerate(PASCAL_VOC_CLASSES) if c in VEHICLE_CLASSES]
PERSON_IDS = [i for i, c in enumerate(PASCAL_VOC_CLASSES) if c in PERSON_CLASSES]
OBSTACLE_IDS = [i for i, c in enumerate(PASCAL_VOC_CLASSES) if c in OBSTACLE_CLASSES]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SegmentationResult:
    """Result from semantic segmentation."""
    
    # Raw outputs
    class_mask: np.ndarray          # (H, W) uint8, class IDs 0-20
    confidence: np.ndarray          # (H, W) float32, 0-1
    
    # Derived masks
    road_mask: np.ndarray           # (H, W) uint8, binary
    obstacle_mask: np.ndarray       # (H, W) uint8, binary
    vehicle_mask: np.ndarray        # (H, W) uint8, binary
    person_mask: np.ndarray         # (H, W) uint8, binary
    
    # Metadata
    detected_classes: Dict[str, float] = field(default_factory=dict)  # class -> percentage
    image_size: Tuple[int, int] = (0, 0)  # (height, width)
    
    def get_class_mask(self, class_name: str) -> np.ndarray:
        """Get binary mask for a specific class."""
        if class_name not in PASCAL_VOC_CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        class_id = PASCAL_VOC_CLASSES.index(class_name)
        return (self.class_mask == class_id).astype(np.uint8)
    
    def get_placement_mask(self, margin: int = 50) -> np.ndarray:
        """
        Get mask of valid placement areas (road minus obstacles).
        
        Args:
            margin: Pixels to shrink from edges
            
        Returns:
            Binary mask where 1 = valid placement area
        """
        h, w = self.image_size
        
        # Start with road
        placement = self.road_mask.copy()
        
        # Remove obstacles (dilated)
        kernel = np.ones((15, 15), np.uint8)
        dilated_obstacles = cv2.dilate(self.obstacle_mask, kernel, iterations=2)
        placement = placement & ~dilated_obstacles
        
        # Apply edge margin
        if margin > 0:
            margin_mask = np.zeros((h, w), dtype=np.uint8)
            margin_mask[margin:h-margin, margin:w-margin] = 1
            placement = placement & margin_mask
        
        return placement
    
    def to_colormap(self) -> np.ndarray:
        """Convert class mask to colored visualization (RGB)."""
        h, w = self.class_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(len(PASCAL_VOC_CLASSES)):
            colored[self.class_mask == class_id] = PASCAL_VOC_COLORS[class_id]
        return colored


# =============================================================================
# MODEL LOADING (SINGLETON)
# =============================================================================

_model_cache: Dict[str, torch.nn.Module] = {}
_transform = None


def _get_transform():
    """Get preprocessing transform."""
    global _transform
    if _transform is None:
        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    return _transform


def _get_model(
    model_type: str = "resnet101",
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Get DeepLabV3 model (cached singleton).
    
    Args:
        model_type: "resnet101" (better) or "mobilenet" (faster)
        device: "cuda" or "cpu"
        
    Returns:
        Loaded model
    """
    # Auto-detect device if cuda requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    cache_key = f"{model_type}_{device}"
    
    if cache_key not in _model_cache:
        logger.info(f"Loading DeepLabV3 ({model_type}) on {device}...")
        
        if model_type == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            model = deeplabv3_resnet101(weights=weights)
        elif model_type == "mobilenet":
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            model = deeplabv3_mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(device).eval()
        _model_cache[cache_key] = model
        logger.info("DeepLabV3 loaded successfully")
    
    return _model_cache[cache_key]


# =============================================================================
# CORE SEGMENTATION FUNCTIONS
# =============================================================================

def segment_image(
    image: np.ndarray,
    model_type: str = "resnet101",
    device: str = None
) -> SegmentationResult:
    """
    Run semantic segmentation on an image.
    
    Args:
        image: RGB image (H, W, 3) uint8
        model_type: "resnet101" or "mobilenet"
        device: "cuda", "cpu", or None (auto-detect)
        
    Returns:
        SegmentationResult with all masks and metadata
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    h, w = image.shape[:2]
    
    # Get model and transform
    model = _get_model(model_type, device)
    transform = _get_transform()
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']
    
    # Get class predictions
    probs = F.softmax(output, dim=1)
    confidence, class_mask = probs.max(dim=1)
    
    # Resize to original size
    class_mask = F.interpolate(
        class_mask.unsqueeze(1).float(),
        size=(h, w),
        mode='nearest'
    ).squeeze().cpu().numpy().astype(np.uint8)
    
    confidence = F.interpolate(
        confidence.unsqueeze(1),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy().astype(np.float32)
    
    # Compute derived masks
    road_mask = _detect_road_from_segmentation(image, class_mask)
    obstacle_mask = _compute_obstacle_mask(class_mask)
    vehicle_mask = _compute_class_group_mask(class_mask, VEHICLE_IDS)
    person_mask = _compute_class_group_mask(class_mask, PERSON_IDS)
    
    # Compute class percentages
    detected_classes = {}
    total_pixels = h * w
    for class_id, class_name in enumerate(PASCAL_VOC_CLASSES):
        pct = 100 * (class_mask == class_id).sum() / total_pixels
        if pct > 0.1:  # Only include classes > 0.1%
            detected_classes[class_name] = round(pct, 2)
    
    return SegmentationResult(
        class_mask=class_mask,
        confidence=confidence,
        road_mask=road_mask,
        obstacle_mask=obstacle_mask,
        vehicle_mask=vehicle_mask,
        person_mask=person_mask,
        detected_classes=detected_classes,
        image_size=(h, w),
    )


def detect_road(
    image: np.ndarray,
    model_type: str = "resnet101",
    device: str = None
) -> np.ndarray:
    """
    Detect road regions in an image.
    
    Args:
        image: RGB image (H, W, 3)
        model_type: "resnet101" or "mobilenet"
        device: "cuda", "cpu", or None (auto-detect)
        
    Returns:
        Binary mask (H, W) where 1 = road
    """
    result = segment_image(image, model_type, device)
    return result.road_mask


def detect_obstacles(
    image: np.ndarray,
    model_type: str = "resnet101",
    device: str = None
) -> np.ndarray:
    """
    Detect obstacles (vehicles, people, animals) in an image.
    
    Args:
        image: RGB image (H, W, 3)
        model_type: "resnet101" or "mobilenet"
        device: "cuda", "cpu", or None (auto-detect)
        
    Returns:
        Binary mask (H, W) where 1 = obstacle
    """
    result = segment_image(image, model_type, device)
    return result.obstacle_mask


def get_placement_mask(
    image: np.ndarray,
    margin: int = 50,
    model_type: str = "resnet101",
    device: str = None
) -> np.ndarray:
    """
    Get mask of valid vehicle placement areas.
    
    Combines road detection with obstacle avoidance.
    
    Args:
        image: RGB image (H, W, 3)
        margin: Pixels to exclude from edges
        model_type: "resnet101" or "mobilenet"
        device: "cuda", "cpu", or None (auto-detect)
        
    Returns:
        Binary mask (H, W) where 1 = valid placement
    """
    result = segment_image(image, model_type, device)
    return result.get_placement_mask(margin=margin)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _detect_road_from_segmentation(
    image: np.ndarray,
    class_mask: np.ndarray
) -> np.ndarray:
    """
    Detect road regions by combining semantic segmentation with color analysis.
    
    Pascal VOC doesn't have a "road" class, so we use heuristics:
    1. Background class (often contains road)
    2. Gray/asphalt color detection
    3. Position prior (roads in lower 2/3)
    4. Exclude known obstacles
    """
    h, w = image.shape[:2]
    
    # 1. Background class (often includes road in outdoor scenes)
    background_mask = (class_mask == 0).astype(np.uint8)
    
    # 2. Obstacle mask (dilated)
    obstacle_mask = _compute_obstacle_mask(class_mask)
    kernel = np.ones((5, 5), np.uint8)
    obstacle_dilated = cv2.dilate(obstacle_mask, kernel, iterations=2)
    
    # 3. Gray/asphalt color detection
    gray_mask = _detect_gray_regions(image)
    
    # 4. Position prior - roads more likely in lower 2/3
    position_mask = np.zeros((h, w), dtype=np.uint8)
    position_mask[h // 4:, :] = 1
    
    # 5. Combine: (background OR gray) AND position AND NOT obstacle
    road_candidate = (
        ((background_mask > 0) | (gray_mask > 0)) &
        (position_mask > 0) &
        (obstacle_dilated == 0)
    ).astype(np.uint8)
    
    # 6. Morphological cleanup
    road_mask = _cleanup_mask(road_candidate)
    
    # 7. Keep only large connected components
    road_mask = _keep_large_components(road_mask, min_area=5000)
    
    return road_mask


def _detect_gray_regions(image: np.ndarray) -> np.ndarray:
    """Detect gray/asphalt colored regions using HSV analysis."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value
    
    # Gray = low saturation, medium brightness
    gray_mask = (
        (s < 50) &    # Low saturation (not colorful)
        (v > 40) &    # Not too dark
        (v < 200)     # Not too bright (not sky)
    ).astype(np.uint8)
    
    return gray_mask


def _compute_obstacle_mask(class_mask: np.ndarray) -> np.ndarray:
    """Compute mask of all obstacles."""
    return _compute_class_group_mask(class_mask, OBSTACLE_IDS)


def _compute_class_group_mask(
    class_mask: np.ndarray,
    class_ids: List[int]
) -> np.ndarray:
    """Compute mask for a group of class IDs."""
    mask = np.zeros_like(class_mask, dtype=np.uint8)
    for class_id in class_ids:
        mask |= (class_mask == class_id).astype(np.uint8)
    return mask


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological cleanup of binary mask."""
    kernel = np.ones((7, 7), np.uint8)
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Open to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask


def _keep_large_components(
    mask: np.ndarray,
    min_area: int = 5000
) -> np.ndarray:
    """Keep only connected components larger than min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    result = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 1
    
    return result


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def segmentation_to_colormap(
    class_mask: np.ndarray,
    colormap: str = "pascal_voc"
) -> np.ndarray:
    """
    Convert class mask to colored visualization.
    
    Args:
        class_mask: (H, W) class IDs
        colormap: "pascal_voc" or "random"
        
    Returns:
        RGB image (H, W, 3)
    """
    h, w = class_mask.shape
    
    if colormap == "pascal_voc":
        colors = PASCAL_VOC_COLORS
    else:
        np.random.seed(42)
        colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
    
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        colored[class_mask == class_id] = colors[class_id]
    
    return colored


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create colored overlay on image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB color tuple
        alpha: Overlay transparency
        
    Returns:
        Image with colored overlay
    """
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def save_segmentation_visualization(
    image: np.ndarray,
    result: SegmentationResult,
    output_dir: Path,
    prefix: str = ""
) -> List[Path]:
    """
    Save all segmentation visualizations.
    
    Args:
        image: Original RGB image
        result: SegmentationResult
        output_dir: Output directory
        prefix: Optional filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved = []
    p = f"{prefix}_" if prefix else ""
    
    # 1. Original
    path = output_dir / f"{p}1_original.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    # 2. Semantic segmentation colored
    seg_colored = result.to_colormap()
    path = output_dir / f"{p}2_segmentation.png"
    cv2.imwrite(str(path), cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    # 3. Segmentation overlay
    overlay = cv2.addWeighted(image, 0.5, seg_colored, 0.5, 0)
    path = output_dir / f"{p}3_segmentation_overlay.png"
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    # 4. Road mask
    road_overlay = create_overlay(image, result.road_mask, (0, 255, 0), 0.4)
    path = output_dir / f"{p}4_road_mask.png"
    cv2.imwrite(str(path), cv2.cvtColor(road_overlay, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    # 5. Obstacle mask
    obstacle_overlay = create_overlay(image, result.obstacle_mask, (255, 0, 0), 0.4)
    path = output_dir / f"{p}5_obstacle_mask.png"
    cv2.imwrite(str(path), cv2.cvtColor(obstacle_overlay, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    # 6. Placement mask
    placement = result.get_placement_mask()
    placement_overlay = create_overlay(image, placement, (0, 255, 255), 0.4)
    path = output_dir / f"{p}6_placement_mask.png"
    cv2.imwrite(str(path), cv2.cvtColor(placement_overlay, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    # 7. Summary grid
    h, w = image.shape[:2]
    summary = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    summary[:h, :w] = image
    summary[:h, w:] = seg_colored
    summary[h:, :w] = road_overlay
    summary[h:, w:] = placement_overlay
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(summary, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(summary, "Segmentation", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(summary, "Road Detection", (10, h + 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(summary, "Placement Mask", (w + 10, h + 30), font, 0.8, (255, 255, 255), 2)
    
    path = output_dir / f"{p}7_summary.png"
    cv2.imwrite(str(path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR))
    saved.append(path)
    
    return saved
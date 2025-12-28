"""
Base utilities for the vehicle inpainting pipeline.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlendingMethod(Enum):
    DIFFERENTIAL = "differential"
    POISSON = "poisson"
    LAPLACIAN = "laplacian"
    LINEAR = "linear"


class ColorTransferMethod(Enum):
    LAB_TRANSFER = "lab_transfer"
    HISTOGRAM_MATCHING = "histogram_matching"


@dataclass
class BoundingBox:
    """Bounding box representation."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def expand(self, pixels: int = 0, factor: float = 1.0) -> 'BoundingBox':
        """Expand bbox by pixels and/or factor."""
        new_w = int(self.width * factor) + 2 * pixels
        new_h = int(self.height * factor) + 2 * pixels
        new_x = self.x - (new_w - self.width) // 2
        new_y = self.y - (new_h - self.height) // 2
        return BoundingBox(new_x, new_y, new_w, new_h)
    
    def clip(self, max_w: int, max_h: int) -> 'BoundingBox':
        """Clip bbox to image boundaries."""
        x = max(0, self.x)
        y = max(0, self.y)
        x2 = min(max_w, self.x2)
        y2 = min(max_h, self.y2)
        return BoundingBox(x, y, x2 - x, y2 - y)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    def to_yolo(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (normalized x_center, y_center, width, height)."""
        x_center = (self.x + self.width / 2) / img_w
        y_center = (self.y + self.height / 2) / img_h
        w_norm = self.width / img_w
        h_norm = self.height / img_h
        return (x_center, y_center, w_norm, h_norm)
    
    @classmethod
    def from_mask(cls, mask: np.ndarray) -> 'BoundingBox':
        """Create bbox from binary mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return cls(0, 0, 0, 0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return cls(int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))


@dataclass
class ValidZone:
    """Represents a valid insertion zone in an image."""
    bbox: BoundingBox
    mask: np.ndarray
    depth_range: Tuple[float, float]
    surface_type: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class VehicleRender:
    """Represents a rendered vehicle with all passes."""
    rgb: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    mask: np.ndarray
    ambient_occlusion: Optional[np.ndarray] = None
    camera_params: Dict = field(default_factory=dict)
    model_name: str = ""
    

@dataclass 
class GenerationResult:
    """Result of a single generation."""
    image: np.ndarray
    mask: np.ndarray
    vehicle_bbox: BoundingBox
    vehicle_class: str
    vehicle_class_id: int
    quality_scores: Dict[str, float] = field(default_factory=dict)
    passed_qa: bool = True
    metadata: Dict = field(default_factory=dict)


def load_config(config_path: str = None) -> Dict:
    """Load pipeline configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.max() > 1.0:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """Convert 0-1 image to uint8."""
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def resize_image(img: np.ndarray, size: Tuple[int, int], method: str = 'lanczos') -> np.ndarray:
    """Resize image with specified interpolation method."""
    import cv2
    
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'area': cv2.INTER_AREA
    }
    
    interp = methods.get(method, cv2.INTER_LANCZOS4)
    return cv2.resize(img, size, interpolation=interp)


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """Convert binary mask to polygon coordinates (COCO format)."""
    import cv2
    
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    
    return polygons


def polygon_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert polygon coordinates to binary mask."""
    import cv2
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in polygons:
        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    
    return mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)


def get_device():
    """Get available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

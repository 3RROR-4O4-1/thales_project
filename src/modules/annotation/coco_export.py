"""
COCO Format Annotation Export

Exports annotations in COCO format for object detection training.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path

import sys
sys.path.append('..')
from ..utils import BoundingBox, mask_to_polygon


@dataclass
class COCOCategory:
    """COCO category definition."""
    id: int
    name: str
    supercategory: str = ""


@dataclass
class COCOImage:
    """COCO image metadata."""
    id: int
    file_name: str
    width: int
    height: int
    date_captured: str = ""
    
    def __post_init__(self):
        if not self.date_captured:
            self.date_captured = datetime.now().isoformat()


@dataclass
class COCOAnnotation:
    """COCO object annotation."""
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    area: float
    segmentation: List[List[float]] = field(default_factory=list)
    iscrowd: int = 0
    
    @classmethod
    def from_mask(
        cls,
        annotation_id: int,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
        include_segmentation: bool = True
    ) -> 'COCOAnnotation':
        """Create annotation from binary mask."""
        bbox = BoundingBox.from_mask(mask)
        area = float(mask.sum())
        
        segmentation = []
        if include_segmentation:
            segmentation = mask_to_polygon(mask)
        
        return cls(
            id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            bbox=list(bbox.to_xywh()),
            area=area,
            segmentation=segmentation
        )


@dataclass
class COCODataset:
    """Complete COCO dataset structure."""
    info: Dict = field(default_factory=lambda: {
        "description": "Synthetic Vehicle Dataset",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "Vehicle Inpainting Pipeline",
        "date_created": datetime.now().isoformat()
    })
    licenses: List[Dict] = field(default_factory=list)
    categories: List[COCOCategory] = field(default_factory=list)
    images: List[COCOImage] = field(default_factory=list)
    annotations: List[COCOAnnotation] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to COCO JSON format."""
        return {
            "info": self.info,
            "licenses": self.licenses,
            "categories": [asdict(c) for c in self.categories],
            "images": [asdict(i) for i in self.images],
            "annotations": [asdict(a) for a in self.annotations]
        }
    
    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'COCODataset':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        dataset = cls()
        dataset.info = data.get("info", dataset.info)
        dataset.licenses = data.get("licenses", [])
        dataset.categories = [
            COCOCategory(**c) for c in data.get("categories", [])
        ]
        dataset.images = [
            COCOImage(**i) for i in data.get("images", [])
        ]
        dataset.annotations = [
            COCOAnnotation(**a) for a in data.get("annotations", [])
        ]
        
        return dataset


class COCOExporter:
    """
    Export generated images and annotations in COCO format.
    """
    
    # Default vehicle categories
    DEFAULT_CATEGORIES = [
        COCOCategory(1, "tank", "military_vehicle"),
        COCOCategory(2, "apc", "military_vehicle"),
        COCOCategory(3, "truck", "military_vehicle"),
        COCOCategory(4, "car", "civilian_vehicle"),
        COCOCategory(5, "helicopter", "aircraft"),
    ]
    
    def __init__(
        self,
        output_dir: str,
        categories: Optional[List[COCOCategory]] = None,
        include_segmentation: bool = True,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        """
        Args:
            output_dir: Output directory for dataset
            categories: List of category definitions
            include_segmentation: Include polygon segmentation
            split_ratio: (train, val, test) split ratios
        """
        self.output_dir = Path(output_dir)
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.include_segmentation = include_segmentation
        self.split_ratio = split_ratio
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize datasets for each split
        self.datasets = {
            'train': self._create_empty_dataset(),
            'val': self._create_empty_dataset(),
            'test': self._create_empty_dataset()
        }
        
        # Counters
        self._image_counter = 0
        self._annotation_counter = 0
    
    def _setup_directories(self):
        """Create output directory structure."""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    def _create_empty_dataset(self) -> COCODataset:
        """Create empty COCO dataset with categories."""
        return COCODataset(categories=list(self.categories))
    
    def _get_split(self) -> str:
        """Determine split for next image based on ratios."""
        import random
        r = random.random()
        
        if r < self.split_ratio[0]:
            return 'train'
        elif r < self.split_ratio[0] + self.split_ratio[1]:
            return 'val'
        else:
            return 'test'
    
    def add_image(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        filename: Optional[str] = None,
        split: Optional[str] = None
    ) -> int:
        """
        Add an image with annotations to the dataset.
        
        Args:
            image: Image as numpy array
            annotations: List of annotation dicts with keys:
                - mask: Binary mask (np.ndarray)
                - category_id: Category ID (int)
                - OR bbox: Bounding box [x, y, w, h]
            filename: Optional filename (auto-generated if not provided)
            split: Force specific split ('train', 'val', 'test')
            
        Returns:
            Image ID
        """
        import cv2
        
        # Determine split
        if split is None:
            split = self._get_split()
        
        # Generate filename if not provided
        self._image_counter += 1
        if filename is None:
            filename = f"synthetic_{self._image_counter:06d}.png"
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Save image
        image_path = self.output_dir / split / 'images' / filename
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Create image entry
        image_id = self._image_counter
        coco_image = COCOImage(
            id=image_id,
            file_name=filename,
            width=w,
            height=h
        )
        self.datasets[split].images.append(coco_image)
        
        # Add annotations
        for ann_data in annotations:
            self._annotation_counter += 1
            
            if 'mask' in ann_data:
                # Create from mask
                coco_ann = COCOAnnotation.from_mask(
                    annotation_id=self._annotation_counter,
                    image_id=image_id,
                    category_id=ann_data['category_id'],
                    mask=ann_data['mask'],
                    include_segmentation=self.include_segmentation
                )
            else:
                # Create from bbox
                bbox = ann_data['bbox']
                coco_ann = COCOAnnotation(
                    id=self._annotation_counter,
                    image_id=image_id,
                    category_id=ann_data['category_id'],
                    bbox=bbox,
                    area=bbox[2] * bbox[3]
                )
            
            self.datasets[split].annotations.append(coco_ann)
        
        return image_id
    
    def save(self):
        """Save all dataset splits to JSON files."""
        for split, dataset in self.datasets.items():
            if dataset.images:  # Only save non-empty splits
                output_path = self.output_dir / 'annotations' / f'instances_{split}.json'
                dataset.save(str(output_path))
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {}
        
        for split, dataset in self.datasets.items():
            category_counts = {}
            for ann in dataset.annotations:
                cat_id = ann.category_id
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            
            stats[split] = {
                'num_images': len(dataset.images),
                'num_annotations': len(dataset.annotations),
                'annotations_per_category': category_counts
            }
        
        return stats


def create_coco_annotation_from_generation(
    image_id: int,
    annotation_id: int,
    vehicle_mask: np.ndarray,
    vehicle_class: str,
    category_mapping: Dict[str, int]
) -> COCOAnnotation:
    """
    Create COCO annotation from generation result.
    
    Args:
        image_id: ID of the image
        annotation_id: Unique annotation ID
        vehicle_mask: Binary mask of the vehicle
        vehicle_class: Class name (e.g., "tank")
        category_mapping: Dict mapping class names to category IDs
        
    Returns:
        COCO annotation object
    """
    category_id = category_mapping.get(vehicle_class, 1)
    
    return COCOAnnotation.from_mask(
        annotation_id=annotation_id,
        image_id=image_id,
        category_id=category_id,
        mask=vehicle_mask,
        include_segmentation=True
    )

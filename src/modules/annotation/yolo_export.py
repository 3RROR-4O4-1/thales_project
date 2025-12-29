"""
YOLO Format Annotation Export

Exports annotations in YOLO format for object detection training.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import yaml

from ..utils import BoundingBox


class YOLOExporter:
    """
    Export generated images and annotations in YOLO format.
    
    YOLO format:
    - One .txt file per image with same name
    - Each line: <class_id> <x_center> <y_center> <width> <height>
    - All values normalized to 0-1
    
    Directory structure:
    dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
    """
    
    DEFAULT_CLASSES = [
        "tank",
        "apc", 
        "truck",
        "car",
        "helicopter"
    ]
    
    def __init__(
        self,
        output_dir: str,
        classes: Optional[List[str]] = None,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        include_segmentation: bool = False
    ):
        """
        Args:
            output_dir: Output directory for dataset
            classes: List of class names
            split_ratio: (train, val, test) split ratios
            include_segmentation: Include polygon segmentation (YOLO-seg format)
        """
        self.output_dir = Path(output_dir)
        self.classes = classes or self.DEFAULT_CLASSES
        self.split_ratio = split_ratio
        self.include_segmentation = include_segmentation
        
        # Class name to ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        
        # Create directory structure
        self._setup_directories()
        
        # Counter for unique filenames
        self._counter = 0
        
        # Track statistics
        self._stats = {
            'train': {'images': 0, 'annotations': 0},
            'val': {'images': 0, 'annotations': 0},
            'test': {'images': 0, 'annotations': 0}
        }
    
    def _setup_directories(self):
        """Create YOLO directory structure."""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
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
    
    def _bbox_to_yolo(
        self,
        bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Convert bbox (x, y, w, h) to YOLO format (normalized x_center, y_center, w, h).
        """
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        return (x_center, y_center, w_norm, h_norm)
    
    def _mask_to_yolo_seg(
        self,
        mask: np.ndarray,
        img_width: int,
        img_height: int
    ) -> List[float]:
        """
        Convert binary mask to YOLO segmentation format.
        
        Returns normalized polygon coordinates: [x1, y1, x2, y2, ...]
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify polygon
        epsilon = 0.005 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        # Normalize coordinates
        coords = []
        for point in simplified.reshape(-1, 2):
            coords.append(point[0] / img_width)
            coords.append(point[1] / img_height)
        
        return coords
    
    def add_image(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        filename: Optional[str] = None,
        split: Optional[str] = None
    ) -> str:
        """
        Add an image with annotations to the dataset.
        
        Args:
            image: Image as numpy array
            annotations: List of annotation dicts with keys:
                - class_name: Class name (str) OR class_id: Class ID (int)
                - bbox: Bounding box [x, y, w, h] OR mask: Binary mask
            filename: Optional filename (auto-generated if not provided)
            split: Force specific split ('train', 'val', 'test')
            
        Returns:
            Filename of saved image
        """
        # Determine split
        if split is None:
            split = self._get_split()
        
        # Generate filename if not provided
        self._counter += 1
        if filename is None:
            filename = f"synthetic_{self._counter:06d}.png"
        
        # Remove extension for label file
        name_without_ext = Path(filename).stem
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Save image
        image_path = self.output_dir / 'images' / split / filename
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Build label content
        label_lines = []
        
        for ann in annotations:
            # Get class ID
            if 'class_id' in ann:
                class_id = ann['class_id']
            elif 'class_name' in ann:
                class_id = self.class_to_id.get(ann['class_name'], 0)
            else:
                class_id = 0
            
            # Get bbox/mask
            if 'mask' in ann:
                mask = ann['mask']
                bbox = BoundingBox.from_mask(mask)
                yolo_bbox = self._bbox_to_yolo(bbox.to_xywh(), w, h)
                
                if self.include_segmentation:
                    # YOLO-seg format: class_id x1 y1 x2 y2 ...
                    seg_coords = self._mask_to_yolo_seg(mask, w, h)
                    if seg_coords:
                        coords_str = ' '.join(f'{c:.6f}' for c in seg_coords)
                        label_lines.append(f"{class_id} {coords_str}")
                    else:
                        # Fallback to bbox if segmentation fails
                        label_lines.append(
                            f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                            f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                        )
                else:
                    label_lines.append(
                        f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                        f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                    )
            
            elif 'bbox' in ann:
                # Handle both BoundingBox objects and tuple/list
                bbox = ann['bbox']
                if isinstance(bbox, BoundingBox):
                    bbox_tuple = bbox.to_xywh()
                else:
                    bbox_tuple = tuple(bbox)
                yolo_bbox = self._bbox_to_yolo(bbox_tuple, w, h)
                label_lines.append(
                    f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                    f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                )
        
        # Save label file
        label_path = self.output_dir / 'labels' / split / f"{name_without_ext}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
        
        # Update stats
        self._stats[split]['images'] += 1
        self._stats[split]['annotations'] += len(annotations)
        
        return filename
    
    def save_yaml(self, yaml_name: str = "data.yaml"):
        """
        Save YOLO data.yaml configuration file.
        """
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.output_dir / yaml_name
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    def save(self):
        """Save dataset configuration."""
        self.save_yaml()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return self._stats.copy()


class YOLOv8Exporter(YOLOExporter):
    """
    YOLO v8 format exporter with additional features.
    
    Adds support for:
    - Instance segmentation labels
    - OBB (Oriented Bounding Box) format
    - Pose estimation format
    """
    
    def __init__(
        self,
        output_dir: str,
        classes: Optional[List[str]] = None,
        task: str = "detect",  # "detect", "segment", "obb"
        **kwargs
    ):
        self.task = task
        
        # Enable segmentation for segment task
        if task == "segment":
            kwargs['include_segmentation'] = True
        
        super().__init__(output_dir, classes, **kwargs)
    
    def save_yaml(self, yaml_name: str = "data.yaml"):
        """Save YOLOv8 compatible data.yaml."""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        # Add task-specific fields
        if self.task == "obb":
            yaml_content['obb'] = True
        
        yaml_path = self.output_dir / yaml_name
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)


def convert_coco_to_yolo(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    classes: Optional[List[str]] = None
):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO annotations JSON
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format
        classes: Optional list of class names (uses COCO categories if not provided)
    """
    import json
    
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)
    
    # Build category mapping
    if classes is None:
        classes = [cat['name'] for cat in sorted(coco['categories'], key=lambda x: x['id'])]
    
    cat_id_to_idx = {}
    for cat in coco['categories']:
        if cat['name'] in classes:
            cat_id_to_idx[cat['id']] = classes.index(cat['name'])
    
    # Create exporter
    exporter = YOLOExporter(output_dir, classes, split_ratio=(1.0, 0.0, 0.0))
    
    # Build image ID to annotations mapping
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process each image
    for img_info in coco['images']:
        img_id = img_info['id']
        img_path = Path(images_dir) / img_info['file_name']
        
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        annotations = []
        for ann in img_to_anns.get(img_id, []):
            if ann['category_id'] not in cat_id_to_idx:
                continue
            
            annotations.append({
                'class_id': cat_id_to_idx[ann['category_id']],
                'bbox': ann['bbox']  # COCO bbox is already [x, y, w, h]
            })
        
        exporter.add_image(image, annotations, img_info['file_name'], split='train')
    
    exporter.save()

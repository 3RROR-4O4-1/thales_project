"""
Annotation Export Module

Exports generated images with annotations in COCO and YOLO formats.
"""

from .coco_export import (
    COCOCategory,
    COCOImage,
    COCOAnnotation,
    COCODataset,
    COCOExporter,
    create_coco_annotation_from_generation
)

from .yolo_export import (
    YOLOExporter,
    YOLOv8Exporter,
    convert_coco_to_yolo
)

__all__ = [
    # COCO format
    'COCOCategory',
    'COCOImage', 
    'COCOAnnotation',
    'COCODataset',
    'COCOExporter',
    'create_coco_annotation_from_generation',
    
    # YOLO format
    'YOLOExporter',
    'YOLOv8Exporter',
    'convert_coco_to_yolo'
]

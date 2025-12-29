"""
Vehicle Inpainting Pipeline - Master Orchestrator (Updated)

Coordinates all pipeline modules for batch processing of synthetic vehicle
insertion into background images.

UPDATED: Integrates ReferenceSelector for viewpoint-aware vehicle selection.
FIXED: Corrected QAConfig field names and SceneAnalysisResult attribute access.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2
import yaml

# Import pipeline modules
from modules.utils import (
    load_config, ensure_dir, BoundingBox, ValidZone, 
    VehicleRender, GenerationResult, normalize_image, denormalize_image
)
from modules.scene_analysis import SceneAnalyzer, SceneAnalysisResult
from modules.inpainting import ScaleAwareInpainter, ScaleAwareConfig, create_inpaint_function
from modules.harmonization import EdgeHarmonizer, HarmonizationConfig
from modules.quality import QualityAssurance, QAConfig
from modules.annotation import COCOExporter, YOLOExporter
from modules.reference_selector import ReferenceSelector, ViewpointEstimator, RenderedView

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    # Paths
    backgrounds_dir: str = "assets/backgrounds"
    models_3d_dir: str = "assets/3d_models"
    renders_dir: str = "assets/renders"
    output_dir: str = "output/generated"
    dataset_dir: str = "output/dataset"
    rejected_dir: str = "output/rejected"
    
    # Processing
    num_insertions_per_background: int = 3
    processing_resolution: int = 1024
    batch_size: int = 10
    num_workers: int = 4
    
    # Reference selection
    use_viewpoint_matching: bool = True  # Enable viewpoint-aware selection
    
    # Quality
    min_clip_score: float = 0.25
    max_artifact_score: float = 0.15
    min_edge_consistency: float = 0.85
    
    # Export
    export_coco: bool = True
    export_yolo: bool = True
    save_rejected: bool = True
    save_intermediate: bool = False
    
    # Misc
    random_seed: Optional[int] = None
    verbose: bool = True


@dataclass
class GenerationJob:
    """Represents a single generation job."""
    job_id: str
    background_path: str
    vehicle_render: VehicleRender
    insertion_zone: ValidZone
    vehicle_class: str
    vehicle_class_id: int
    
    # Reference selection metadata
    selected_view: Optional[Dict] = None
    
    # Results
    result: Optional[GenerationResult] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class PipelineOrchestrator:
    """
    Master orchestrator for the vehicle inpainting pipeline.
    
    Coordinates:
    - Scene analysis for valid insertion zones
    - Viewpoint-aware vehicle render selection
    - Scale-aware inpainting
    - Edge harmonization
    - Quality assurance
    - Annotation export
    """
    
    def __init__(
        self,
        config: Optional[Union[PipelineConfig, Dict, str]] = None
    ):
        """
        Args:
            config: Pipeline configuration (PipelineConfig, dict, or path to YAML)
        """
        self.config = self._load_config(config)
        
        # Set random seed
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Initialize modules
        self._init_modules()
        
        # Setup directories
        self._setup_directories()
        
        # Reference selectors by vehicle class
        self.reference_selectors: Dict[str, ReferenceSelector] = {}
        
        # Viewpoint estimator
        self.viewpoint_estimator = ViewpointEstimator()
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "by_class": {},
            "by_rejection_reason": {},
            "viewpoint_matches": []  # Track viewpoint matching stats
        }
    
    def _load_config(self, config) -> PipelineConfig:
        """Load and validate configuration."""
        if config is None:
            return PipelineConfig()
        elif isinstance(config, PipelineConfig):
            return config
        elif isinstance(config, dict):
            return PipelineConfig(**config)
        elif isinstance(config, str):
            with open(config, 'r') as f:
                data = yaml.safe_load(f)
            return PipelineConfig(**data.get('orchestrator', data))
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def _init_modules(self):
        """Initialize pipeline modules."""
        logger.info("Initializing pipeline modules...")
        
        # Scene analyzer
        self.scene_analyzer = SceneAnalyzer()
        
        # Inpainting
        inpaint_config = ScaleAwareConfig(
            processing_resolution=self.config.processing_resolution
        )
        self.inpainter = ScaleAwareInpainter(inpaint_config, inpaint_fn=None)
        
        # Harmonization
        harm_config = HarmonizationConfig(
            blend_method="hybrid",
            feather_pixels=32,
            shadow_generation=True
        )
        self.harmonizer = EdgeHarmonizer(harm_config)
        
        # Quality assurance - FIXED: Use correct field names
        qa_config = QAConfig(
            clip_score=self.config.min_clip_score,
            artifact_score=self.config.max_artifact_score,
            edge_consistency=self.config.min_edge_consistency
        )
        self.qa = QualityAssurance(qa_config)
        
        logger.info("Pipeline modules initialized")
    
    def _setup_directories(self):
        """Create output directories."""
        ensure_dir(self.config.output_dir)
        ensure_dir(self.config.dataset_dir)
        if self.config.save_rejected:
            ensure_dir(self.config.rejected_dir)
    
    def set_inpaint_function(self, inpaint_fn):
        """Set the inpainting function (e.g., from ComfyUI)."""
        self.inpainter.inpaint_fn = inpaint_fn
        logger.info("Inpainting function set")
    
    def load_vehicle_renders(self, renders_dir: Optional[str] = None) -> Dict[str, List[VehicleRender]]:
        """
        Load pre-rendered vehicle images organized by class.
        Also initializes ReferenceSelector for each vehicle class.
        
        Args:
            renders_dir: Directory containing rendered vehicles
            
        Returns:
            Dict mapping class names to lists of VehicleRender objects
        """
        renders_dir = Path(renders_dir or self.config.renders_dir)
        renders_by_class = {}
        
        if not renders_dir.exists():
            logger.warning(f"Renders directory not found: {renders_dir}")
            return renders_by_class
        
        # Scan for vehicle model directories
        for model_dir in renders_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            meta_path = model_dir / "metadata.json"
            if not meta_path.exists():
                # Try to scan directory structure
                meta_path = None
            
            try:
                model_name = model_dir.name
                vehicle_class = self._infer_class(model_name)
                
                if vehicle_class not in renders_by_class:
                    renders_by_class[vehicle_class] = []
                
                # Initialize ReferenceSelector for this model
                if self.config.use_viewpoint_matching:
                    if vehicle_class not in self.reference_selectors:
                        self.reference_selectors[vehicle_class] = {}
                    self.reference_selectors[vehicle_class][model_name] = ReferenceSelector(str(model_dir))
                    logger.info(f"Initialized ReferenceSelector for {model_name}")
                
                # Load renders
                if meta_path and meta_path.exists():
                    renders = self._load_renders_from_metadata(meta_path, model_name, vehicle_class)
                else:
                    renders = self._load_renders_from_structure(model_dir, model_name, vehicle_class)
                
                renders_by_class[vehicle_class].extend(renders)
                
            except Exception as e:
                logger.warning(f"Failed to load renders from {model_dir}: {e}")
        
        # Log summary
        total = sum(len(v) for v in renders_by_class.values())
        logger.info(f"Loaded {total} renders across {len(renders_by_class)} classes")
        for cls, renders in renders_by_class.items():
            logger.info(f"  {cls}: {len(renders)} renders")
        
        return renders_by_class
    
    def _load_renders_from_metadata(
        self,
        meta_path: Path,
        model_name: str,
        vehicle_class: str
    ) -> List[VehicleRender]:
        """Load renders using metadata.json file."""
        renders = []
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        base_dir = meta_path.parent
        
        for view in metadata.get("views", []):
            rgb_path = base_dir / view.get("rgb_path", "")
            mask_path = base_dir / view.get("mask_path", "")
            
            if not rgb_path.exists():
                continue
            
            # Load RGBA image
            rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            if rgb_img is None:
                continue
            
            # Handle RGBA vs RGB
            if rgb_img.shape[-1] == 4:
                rgb = cv2.cvtColor(rgb_img[:, :, :3], cv2.COLOR_BGR2RGB)
                mask = rgb_img[:, :, 3]
            else:
                rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            
            render = VehicleRender(
                rgb=rgb,
                depth=None,  # Load on demand if needed
                normal=None,
                mask=mask,
                camera_params={
                    "azimuth": view.get("azimuth", 0),
                    "elevation": view.get("elevation", 30),
                    "distance": view.get("distance", 10)
                },
                model_name=model_name
            )
            renders.append(render)
        
        return renders
    
    def _load_renders_from_structure(
        self,
        model_dir: Path,
        model_name: str,
        vehicle_class: str
    ) -> List[VehicleRender]:
        """Load renders by scanning directory structure."""
        renders = []
        
        rgb_dir = model_dir / "rgb"
        mask_dir = model_dir / "mask"
        
        if not rgb_dir.exists():
            rgb_dir = model_dir
        
        for rgb_path in sorted(rgb_dir.glob("*.png")):
            # Parse view info from filename: view_az030_el15.png
            view_info = self._parse_view_filename(rgb_path.stem)
            
            # Load image
            rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            if rgb_img is None:
                continue
            
            # Handle RGBA
            if rgb_img.shape[-1] == 4:
                rgb = cv2.cvtColor(rgb_img[:, :, :3], cv2.COLOR_BGR2RGB)
                mask = rgb_img[:, :, 3]
            else:
                rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                mask_path = mask_dir / rgb_path.name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            
            render = VehicleRender(
                rgb=rgb,
                depth=None,
                normal=None,
                mask=mask,
                camera_params=view_info,
                model_name=model_name
            )
            renders.append(render)
        
        return renders
    
    def _parse_view_filename(self, filename: str) -> Dict:
        """Parse azimuth/elevation from filename like view_az030_el15."""
        info = {"azimuth": 0, "elevation": 30, "distance": 10}
        
        parts = filename.lower().split('_')
        for part in parts:
            if part.startswith('az'):
                try:
                    info["azimuth"] = float(part[2:])
                except ValueError:
                    pass
            elif part.startswith('el'):
                try:
                    info["elevation"] = float(part[2:])
                except ValueError:
                    pass
            elif part.startswith('d'):
                try:
                    info["distance"] = float(part[1:])
                except ValueError:
                    pass
        
        return info
    
    def _infer_class(self, model_name: str) -> str:
        """Infer vehicle class from model name."""
        name_lower = model_name.lower()
        
        class_keywords = {
            "tank": ["tank", "t-90", "t90", "m1", "abrams", "leopard", "challenger"],
            "apc": ["apc", "btr", "bmp", "stryker", "bradley"],
            "truck": ["truck", "ural", "kamaz", "hemtt"],
            "helicopter": ["heli", "helicopter", "apache", "blackhawk", "mi-"],
            "jet": ["jet", "fighter", "f-16", "f-35", "su-"],
        }
        
        for vehicle_class, keywords in class_keywords.items():
            if any(kw in name_lower for kw in keywords):
                return vehicle_class
        
        return "vehicle"  # Default class
    
    def _get_class_id(self, class_name: str) -> int:
        """Get numeric class ID."""
        class_map = {
            "tank": 1,
            "apc": 2,
            "truck": 3,
            "helicopter": 4,
            "jet": 5,
            "vehicle": 0
        }
        return class_map.get(class_name, 0)
    
    def select_best_render(
        self,
        background: np.ndarray,
        vehicle_class: str,
        insertion_zone: ValidZone,
        available_renders: List[VehicleRender]
    ) -> Tuple[VehicleRender, Dict]:
        """
        Select the best vehicle render based on background viewpoint.
        
        Args:
            background: Background image
            vehicle_class: Vehicle class to insert
            insertion_zone: Target insertion zone
            available_renders: List of available renders for this class
            
        Returns:
            (best_render, selection_metadata)
        """
        if not self.config.use_viewpoint_matching or not available_renders:
            # Fallback to random selection
            render = random.choice(available_renders)
            return render, {"method": "random"}
        
        # Get image dimensions
        h, w = background.shape[:2]
        
        # Calculate zone center position (normalized)
        zone_center_x = (insertion_zone.bbox.x + insertion_zone.bbox.width / 2) / w
        zone_center_y = (insertion_zone.bbox.y + insertion_zone.bbox.height / 2) / h
        
        # Estimate viewpoint from background
        viewpoint = self.viewpoint_estimator.estimate(background)
        
        # Adjust azimuth based on zone position
        # If zone is on left side, show vehicle's right side (add azimuth)
        azimuth_offset = (zone_center_x - 0.5) * 30  # ±15° adjustment
        target_azimuth = (viewpoint.azimuth + azimuth_offset) % 360
        target_elevation = viewpoint.elevation
        
        logger.debug(f"Target viewpoint: az={target_azimuth:.1f}°, el={target_elevation:.1f}°")
        
        # Find best matching render
        def angle_distance(render: VehicleRender) -> float:
            params = render.camera_params
            az = params.get("azimuth", 0)
            el = params.get("elevation", 30)
            
            # Handle azimuth wrap-around
            az_diff = abs(az - target_azimuth)
            az_diff = min(az_diff, 360 - az_diff)
            
            el_diff = abs(el - target_elevation)
            
            return (az_diff ** 2 + el_diff ** 2) ** 0.5
        
        best_render = min(available_renders, key=angle_distance)
        
        # Calculate match quality
        distance = angle_distance(best_render)
        
        selection_metadata = {
            "method": "viewpoint_matching",
            "estimated_viewpoint": {
                "azimuth": viewpoint.azimuth,
                "elevation": viewpoint.elevation,
                "confidence": viewpoint.confidence
            },
            "target_viewpoint": {
                "azimuth": target_azimuth,
                "elevation": target_elevation
            },
            "selected_viewpoint": {
                "azimuth": best_render.camera_params.get("azimuth", 0),
                "elevation": best_render.camera_params.get("elevation", 30)
            },
            "angle_distance": distance,
            "zone_position": {"x": zone_center_x, "y": zone_center_y}
        }
        
        # Track stats
        self.stats["viewpoint_matches"].append({
            "target_az": target_azimuth,
            "target_el": target_elevation,
            "selected_az": best_render.camera_params.get("azimuth", 0),
            "selected_el": best_render.camera_params.get("elevation", 30),
            "distance": distance
        })
        
        return best_render, selection_metadata
    
    def process_background(
        self,
        background_path: str,
        vehicle_renders: Dict[str, List[VehicleRender]],
        num_insertions: Optional[int] = None
    ) -> List[GenerationResult]:
        """
        Process a single background image with multiple vehicle insertions.
        
        Args:
            background_path: Path to background image
            vehicle_renders: Dict of renders by class
            num_insertions: Number of insertions (uses config default if None)
            
        Returns:
            List of successful generation results
        """
        num_insertions = num_insertions or self.config.num_insertions_per_background
        results = []
        
        # Load background
        background = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
        
        # Analyze scene
        logger.debug(f"Analyzing scene: {background_path}")
        scene_info: SceneAnalysisResult = self.scene_analyzer.analyze(background)
        
        # FIXED: Access as dataclass attribute, not dict
        valid_zones = scene_info.valid_zones
        if not valid_zones:
            logger.warning(f"No valid insertion zones found in {background_path}")
            return results
        
        # Generate insertions
        for i in range(min(num_insertions, len(valid_zones))):
            zone = valid_zones[i]
            
            # Select random vehicle class
            vehicle_class = random.choice(list(vehicle_renders.keys()))
            if not vehicle_renders[vehicle_class]:
                continue
            
            # Select BEST render based on viewpoint
            vehicle_render, selection_metadata = self.select_best_render(
                background,
                vehicle_class,
                zone,
                vehicle_renders[vehicle_class]
            )
            
            logger.info(f"Selected render: az={vehicle_render.camera_params.get('azimuth', 0)}°, "
                       f"el={vehicle_render.camera_params.get('elevation', 30)}° "
                       f"(method: {selection_metadata.get('method', 'unknown')})")
            
            # Create job
            job = GenerationJob(
                job_id=f"{Path(background_path).stem}_{i:03d}",
                background_path=background_path,
                vehicle_render=vehicle_render,
                insertion_zone=zone,
                vehicle_class=vehicle_class,
                vehicle_class_id=self._get_class_id(vehicle_class),
                selected_view=selection_metadata  # Store selection info
            )
            
            # Process
            result = self._process_job(job, background, scene_info)
            
            if result is not None:
                results.append(result)
                
                # Update background for next insertion (accumulate vehicles)
                background = result.image.copy()
        
        return results
    
    def _process_job(
        self,
        job: GenerationJob,
        background: np.ndarray,
        scene_info: SceneAnalysisResult
    ) -> Optional[GenerationResult]:
        """Process a single generation job."""
        import time
        start_time = time.time()
        
        try:
            zone = job.insertion_zone
            render = job.vehicle_render
            
            # Create insertion mask at target location
            mask = self._create_insertion_mask(
                background.shape[:2],
                zone.bbox,
                render.mask
            )
            
            # Scale vehicle render to fit zone
            scaled_render = self._scale_render_to_zone(render, zone)
            
            # Run inpainting
            if self.inpainter.inpaint_fn is not None:
                generated = self.inpainter.process(
                    background,
                    mask,
                    vehicle_render=scaled_render.rgb,
                    depth_map=scene_info.depth_map,  # FIXED: Access as attribute
                    inpaint_kwargs={
                        'prompt': f"Insert {job.vehicle_class} naturally into the scene",
                        'reference': scaled_render.rgb
                    }
                )
            else:
                # Fallback: simple composite
                generated = self._simple_composite(background, scaled_render, zone)
            
            # FIXED: Access lighting_info as attribute, then get light_direction from dict
            light_direction = None
            if scene_info.lighting_info:
                light_direction = scene_info.lighting_info.get('light_direction')
            
            # Harmonize
            harmonized = self.harmonizer.harmonize(
                background,
                generated,
                mask,
                vehicle_mask=mask,
                light_direction=light_direction
            )
            
            # Quality check
            qa_result = self.qa.evaluate(
                harmonized,
                background,
                mask,
                prompt=f"{job.vehicle_class} in natural scene"
            )
            
            # Create result with view selection metadata
            result = GenerationResult(
                image=harmonized,
                mask=mask,
                vehicle_bbox=zone.bbox,
                vehicle_class=job.vehicle_class,
                vehicle_class_id=job.vehicle_class_id,
                quality_scores=qa_result.scores,
                passed_qa=qa_result.passed,
                metadata={
                    "job_id": job.job_id,
                    "background": job.background_path,
                    "render": render.model_name,
                    "render_view": render.camera_params,  # Include view info
                    "view_selection": job.selected_view,  # Include selection metadata
                    "zone_depth_range": zone.depth_range,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Update stats
            self.stats["total_processed"] += 1
            if result.passed_qa:
                self.stats["total_accepted"] += 1
            else:
                self.stats["total_rejected"] += 1
                if self.config.save_rejected:
                    self._save_rejected(result, qa_result.failure_reasons)
            
            return result if result.passed_qa else None
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            import traceback
            traceback.print_exc()
            job.error = str(e)
            return None
    
    def _create_insertion_mask(
        self,
        image_shape: Tuple[int, int],
        bbox: BoundingBox,
        vehicle_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Create insertion mask at target location."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.float32)
        
        x, y, bw, bh = bbox.to_xywh()
        
        if vehicle_mask is not None:
            resized_mask = cv2.resize(vehicle_mask, (bw, bh), interpolation=cv2.INTER_LINEAR)
            resized_mask = resized_mask.astype(np.float32) / 255.0 if resized_mask.max() > 1 else resized_mask
        else:
            resized_mask = np.ones((bh, bw), dtype=np.float32)
        
        y1, y2 = max(0, y), min(h, y + bh)
        x1, x2 = max(0, x), min(w, x + bw)
        
        my1, my2 = max(0, -y), bh - max(0, y + bh - h)
        mx1, mx2 = max(0, -x), bw - max(0, x + bw - w)
        
        mask[y1:y2, x1:x2] = resized_mask[my1:my2, mx1:mx2]
        
        return mask
    
    def _scale_render_to_zone(
        self,
        render: VehicleRender,
        zone: ValidZone
    ) -> VehicleRender:
        """Scale vehicle render to fit insertion zone."""
        target_h, target_w = zone.bbox.height, zone.bbox.width
        
        def resize(img):
            if img is None:
                return None
            return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        return VehicleRender(
            rgb=resize(render.rgb),
            depth=resize(render.depth),
            normal=resize(render.normal),
            mask=resize(render.mask),
            camera_params=render.camera_params,
            model_name=render.model_name
        )
    
    def _simple_composite(
        self,
        background: np.ndarray,
        render: VehicleRender,
        zone: ValidZone
    ) -> np.ndarray:
        """Simple alpha composite as fallback when no inpainting function."""
        result = background.copy()
        
        x, y, w, h = zone.bbox.to_xywh()
        
        if render.mask is not None:
            alpha = render.mask.astype(np.float32) / 255.0
            alpha = alpha[:, :, np.newaxis]
            
            y1, y2 = max(0, y), min(background.shape[0], y + h)
            x1, x2 = max(0, x), min(background.shape[1], x + w)
            
            result[y1:y2, x1:x2] = (
                result[y1:y2, x1:x2] * (1 - alpha[:y2-y1, :x2-x1]) +
                render.rgb[:y2-y1, :x2-x1] * alpha[:y2-y1, :x2-x1]
            ).astype(np.uint8)
        
        return result
    
    def _save_rejected(self, result: GenerationResult, reasons: List[str]):
        """Save rejected result for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rejected_{timestamp}_{result.metadata.get('job_id', 'unknown')}.png"
        
        output_path = Path(self.config.rejected_dir) / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        meta_path = output_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                "reasons": reasons,
                "scores": result.quality_scores,
                "metadata": result.metadata
            }, f, indent=2, default=str)
    
    def run_batch(
        self,
        background_paths: List[str],
        vehicle_renders: Dict[str, List[VehicleRender]],
        progress_callback=None
    ) -> Tuple[List[GenerationResult], Dict]:
        """
        Process multiple backgrounds in batch.
        
        Args:
            background_paths: List of background image paths
            vehicle_renders: Dict of renders by class
            progress_callback: Optional callback(current, total, result)
            
        Returns:
            (list of results, statistics dict)
        """
        all_results = []
        total = len(background_paths)
        
        for i, bg_path in enumerate(background_paths):
            logger.info(f"Processing background {i+1}/{total}: {bg_path}")
            
            results = self.process_background(bg_path, vehicle_renders)
            all_results.extend(results)
            
            if progress_callback:
                progress_callback(i + 1, total, results)
        
        return all_results, self.get_statistics()
    
    def export_dataset(
        self,
        results: List[GenerationResult],
        dataset_name: str = "synthetic_vehicles"
    ):
        """Export results to COCO and/or YOLO format."""
        logger.info(f"Exporting dataset: {dataset_name}")
        
        if self.config.export_coco:
            coco_dir = Path(self.config.dataset_dir) / "coco" / dataset_name
            exporter = COCOExporter(str(coco_dir))
            
            for result in results:
                exporter.add_image(
                    result.image,
                    [{
                        "mask": result.mask,
                        "category_id": result.vehicle_class_id,
                        "category_name": result.vehicle_class
                    }],
                    metadata=result.metadata
                )
            
            exporter.save()
            logger.info(f"COCO dataset saved to {coco_dir}")
        
        if self.config.export_yolo:
            yolo_dir = Path(self.config.dataset_dir) / "yolo" / dataset_name
            classes = ["vehicle", "tank", "apc", "truck", "helicopter", "jet"]
            exporter = YOLOExporter(str(yolo_dir), classes)
            
            for result in results:
                exporter.add_image(
                    result.image,
                    [{
                        "bbox": result.vehicle_bbox,
                        "class_id": result.vehicle_class_id
                    }]
                )
            
            exporter.save()
            logger.info(f"YOLO dataset saved to {yolo_dir}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        # Calculate viewpoint matching statistics
        if stats["viewpoint_matches"]:
            distances = [m["distance"] for m in stats["viewpoint_matches"]]
            stats["viewpoint_stats"] = {
                "count": len(distances),
                "mean_angle_distance": np.mean(distances),
                "max_angle_distance": np.max(distances),
                "min_angle_distance": np.min(distances)
            }
        
        # Calculate acceptance rate
        if stats["total_processed"] > 0:
            stats["acceptance_rate"] = stats["total_accepted"] / stats["total_processed"]
        
        return stats
    
    def print_statistics(self):
        """Print statistics summary."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("PIPELINE STATISTICS")
        print("=" * 60)
        print(f"Total processed:  {stats['total_processed']}")
        print(f"Accepted:         {stats['total_accepted']}")
        print(f"Rejected:         {stats['total_rejected']}")
        
        if "acceptance_rate" in stats:
            print(f"Acceptance rate:  {stats['acceptance_rate']*100:.1f}%")
        
        if "viewpoint_stats" in stats:
            vs = stats["viewpoint_stats"]
            print(f"\nViewpoint Matching:")
            print(f"  Matches made:       {vs['count']}")
            print(f"  Mean angle error:   {vs['mean_angle_distance']:.1f}°")
            print(f"  Max angle error:    {vs['max_angle_distance']:.1f}°")
        
        print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle Inpainting Pipeline')
    parser.add_argument('--config', '-c', help='Path to config YAML')
    parser.add_argument('--backgrounds', '-b', required=True, help='Backgrounds directory')
    parser.add_argument('--renders', '-r', required=True, help='Renders directory')
    parser.add_argument('--output', '-o', default='./output', help='Output directory')
    parser.add_argument('--num-insertions', '-n', type=int, default=1, help='Insertions per background')
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        backgrounds_dir=args.backgrounds,
        renders_dir=args.renders,
        output_dir=args.output,
        num_insertions_per_background=args.num_insertions
    )
    
    # Initialize pipeline
    pipeline = PipelineOrchestrator(config)
    
    # Load renders
    renders = pipeline.load_vehicle_renders()
    
    # Get background images
    bg_dir = Path(args.backgrounds)
    backgrounds = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
    
    # Process
    results, stats = pipeline.run_batch([str(p) for p in backgrounds], renders)
    
    # Export
    pipeline.export_dataset(results)
    
    # Print stats
    pipeline.print_statistics()

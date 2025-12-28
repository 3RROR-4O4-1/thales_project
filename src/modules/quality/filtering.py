"""
Quality Filtering Module

Filters generated images based on quality thresholds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import cv2

from .metrics import QualityMetrics, compute_all_metrics, CLIPScorer

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Thresholds for quality filtering."""
    clip_score: float = 0.25
    artifact_score: float = 0.15  # Lower is better
    edge_consistency: float = 0.85
    color_harmony: float = 0.80
    geometry_score: float = 0.75
    blur_score: float = 0.5
    overall_score: float = 0.7


@dataclass
class FilterResult:
    """Result of quality filtering for a single image."""
    passed: bool
    metrics: QualityMetrics
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "metrics": self.metrics.to_dict(),
            "failure_reasons": self.failure_reasons
        }


class QualityFilter:
    """
    Filter generated images based on quality metrics.
    """
    
    def __init__(
        self,
        thresholds: Optional[QualityThresholds] = None,
        reject_on_any_fail: bool = True,
        use_clip: bool = True,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda"
    ):
        """
        Args:
            thresholds: Quality thresholds
            reject_on_any_fail: Reject if any metric fails
            use_clip: Use CLIP for text-image alignment scoring
            clip_model_name: CLIP model variant
            device: Device for CLIP
        """
        self.thresholds = thresholds or QualityThresholds()
        self.reject_on_any_fail = reject_on_any_fail
        
        # Load CLIP if requested
        self.clip_scorer = None
        if use_clip:
            try:
                self.clip_scorer = CLIPScorer(clip_model_name, device)
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")
    
    def check(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "",
        reference_image: Optional[np.ndarray] = None
    ) -> FilterResult:
        """
        Check if an image passes quality thresholds.
        
        Args:
            image: Generated image
            mask: Mask of generated region
            prompt: Text prompt used
            reference_image: Optional reference for geometry comparison
            
        Returns:
            FilterResult with pass/fail status and details
        """
        # Compute metrics
        metrics = compute_all_metrics(
            image, mask, prompt, reference_image, self.clip_scorer
        )
        
        # Check against thresholds
        failures = []
        
        if metrics.clip_score < self.thresholds.clip_score:
            failures.append(f"clip_score ({metrics.clip_score:.3f} < {self.thresholds.clip_score})")
        
        if metrics.artifact_score > self.thresholds.artifact_score:
            failures.append(f"artifact_score ({metrics.artifact_score:.3f} > {self.thresholds.artifact_score})")
        
        if metrics.edge_consistency < self.thresholds.edge_consistency:
            failures.append(f"edge_consistency ({metrics.edge_consistency:.3f} < {self.thresholds.edge_consistency})")
        
        if metrics.color_harmony < self.thresholds.color_harmony:
            failures.append(f"color_harmony ({metrics.color_harmony:.3f} < {self.thresholds.color_harmony})")
        
        if metrics.geometry_score < self.thresholds.geometry_score:
            failures.append(f"geometry_score ({metrics.geometry_score:.3f} < {self.thresholds.geometry_score})")
        
        if metrics.blur_score < self.thresholds.blur_score:
            failures.append(f"blur_score ({metrics.blur_score:.3f} < {self.thresholds.blur_score})")
        
        # Determine pass/fail
        if self.reject_on_any_fail:
            passed = len(failures) == 0
        else:
            passed = metrics.overall_score >= self.thresholds.overall_score
        
        return FilterResult(
            passed=passed,
            metrics=metrics,
            failure_reasons=failures
        )
    
    def filter_batch(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        prompts: Optional[List[str]] = None,
        references: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[Tuple[np.ndarray, FilterResult]], List[Tuple[np.ndarray, FilterResult]]]:
        """
        Filter a batch of images.
        
        Args:
            images: List of generated images
            masks: List of masks
            prompts: Optional list of prompts
            references: Optional list of reference images
            
        Returns:
            Tuple of (accepted, rejected) lists, each containing (image, result) tuples
        """
        accepted = []
        rejected = []
        
        for i, (image, mask) in enumerate(zip(images, masks)):
            prompt = prompts[i] if prompts else ""
            reference = references[i] if references else None
            
            result = self.check(image, mask, prompt, reference)
            
            if result.passed:
                accepted.append((image, result))
            else:
                rejected.append((image, result))
        
        logger.info(f"Filtered {len(images)} images: {len(accepted)} accepted, {len(rejected)} rejected")
        
        return accepted, rejected


class QualityLogger:
    """
    Log quality metrics and filtering results.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.output_dir / "quality_log.jsonl"
        self.summary_file = self.output_dir / "quality_summary.json"
        
        self._results = []
    
    def log(
        self,
        image_id: str,
        result: FilterResult,
        metadata: Optional[Dict] = None
    ):
        """Log a single result."""
        entry = {
            "image_id": image_id,
            "passed": result.passed,
            "metrics": result.metrics.to_dict(),
            "failure_reasons": result.failure_reasons
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        self._results.append(entry)
        
        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def save_rejected(
        self,
        image: np.ndarray,
        image_id: str,
        result: FilterResult
    ):
        """Save a rejected image with its metrics."""
        rejected_dir = self.output_dir / "rejected"
        rejected_dir.mkdir(exist_ok=True)
        
        # Save image
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        image_path = rejected_dir / f"{image_id}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        meta_path = rejected_dir / f"{image_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self._results:
            return {}
        
        passed = [r for r in self._results if r["passed"]]
        failed = [r for r in self._results if not r["passed"]]
        
        # Compute metric averages
        metrics_keys = ["clip_score", "artifact_score", "edge_consistency", 
                       "color_harmony", "geometry_score", "overall_score"]
        
        avg_metrics = {}
        for key in metrics_keys:
            values = [r["metrics"].get(key, 0) for r in self._results]
            avg_metrics[key] = float(np.mean(values)) if values else 0
        
        # Count failure reasons
        failure_counts = {}
        for r in failed:
            for reason in r.get("failure_reasons", []):
                # Extract metric name
                metric = reason.split("(")[0].strip()
                failure_counts[metric] = failure_counts.get(metric, 0) + 1
        
        return {
            "total": len(self._results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / len(self._results) if self._results else 0,
            "average_metrics": avg_metrics,
            "failure_counts": failure_counts
        }
    
    def save_summary(self):
        """Save summary to file."""
        summary = self.get_summary()
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Quality summary: {summary['passed']}/{summary['total']} passed "
                   f"({summary['pass_rate']*100:.1f}%)")

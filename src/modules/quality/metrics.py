"""
Quality Metrics Module

Computes quality metrics for generated images to filter out failures.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import cv2
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for all quality metrics."""
    clip_score: float = 0.0
    artifact_score: float = 0.0
    edge_consistency: float = 0.0
    color_harmony: float = 0.0
    geometry_score: float = 0.0
    blur_score: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "clip_score": self.clip_score,
            "artifact_score": self.artifact_score,
            "edge_consistency": self.edge_consistency,
            "color_harmony": self.color_harmony,
            "geometry_score": self.geometry_score,
            "blur_score": self.blur_score,
            "overall_score": self.overall_score
        }


class CLIPScorer:
    """
    Compute CLIP score for image-text alignment.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model = None
        self.preprocess = None
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load CLIP model."""
        try:
            import torch
            import clip
            
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            logger.info(f"Loaded CLIP model: {model_name}")
            
        except ImportError:
            logger.warning("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            self.model = None
    
    def score(self, image: np.ndarray, text: str) -> float:
        """
        Compute CLIP similarity between image and text.
        
        Args:
            image: RGB image
            text: Text description
            
        Returns:
            Similarity score (0-1)
        """
        if self.model is None:
            return 0.5  # Default score if CLIP not available
        
        import torch
        from PIL import Image
        
        # Convert to PIL
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Preprocess
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        import clip
        text_input = clip.tokenize([text]).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (image_features @ text_features.T).item()
        
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2


class BlurDetector:
    """
    Detect blur and sharpness in images.
    """
    
    @staticmethod
    def laplacian_variance(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Laplacian variance as blur measure.
        
        Higher values = sharper image
        Lower values = more blur
        
        Args:
            image: Input image
            mask: Optional mask for region-specific analysis
            
        Returns:
            Laplacian variance (higher = sharper)
        """
        # Convert to grayscale
        if image.ndim == 3:
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        if mask is not None:
            # Only compute within mask
            values = laplacian[mask > 0.5]
            if len(values) == 0:
                return 0.0
            return float(values.var())
        else:
            return float(laplacian.var())
    
    @staticmethod
    def gradient_magnitude(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute average gradient magnitude as sharpness measure.
        """
        if image.ndim == 3:
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if mask is not None:
            values = magnitude[mask > 0.5]
            if len(values) == 0:
                return 0.0
            return float(values.mean())
        else:
            return float(magnitude.mean())
    
    @staticmethod
    def score(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Compute blur score (0 = very blurry, 1 = sharp).
        """
        lap_var = BlurDetector.laplacian_variance(image, mask)
        
        # Normalize to 0-1 range
        # These thresholds are empirically determined
        # Variance below 100 is quite blurry, above 500 is sharp
        score = np.clip((lap_var - 50) / 500, 0, 1)
        
        return float(score)


class EdgeConsistencyChecker:
    """
    Check for edge artifacts at mask boundaries.
    """
    
    @staticmethod
    def check_boundary_continuity(
        image: np.ndarray,
        mask: np.ndarray,
        boundary_width: int = 10
    ) -> float:
        """
        Check if edges are continuous across mask boundary.
        
        Args:
            image: Generated image
            mask: Binary mask of generated region
            boundary_width: Width of boundary region to analyze
            
        Returns:
            Consistency score (0-1), higher = better
        """
        # Get boundary region
        kernel = np.ones((boundary_width, boundary_width), np.uint8)
        dilated = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
        eroded = cv2.erode((mask > 0.5).astype(np.uint8), kernel)
        
        boundary = dilated.astype(bool) & ~eroded.astype(bool)
        
        if not boundary.any():
            return 1.0
        
        # Convert to grayscale
        if image.ndim == 3:
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge continuity in boundary region
        boundary_edges = edges[boundary]
        
        # High edge values in boundary = visible seam
        edge_density = boundary_edges.mean() / 255.0
        
        # Lower edge density = better (less visible seams)
        return 1.0 - min(edge_density * 5, 1.0)
    
    @staticmethod
    def check_gradient_continuity(
        image: np.ndarray,
        mask: np.ndarray,
        boundary_width: int = 5
    ) -> float:
        """
        Check if gradients are continuous across mask boundary.
        """
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Get inner and outer boundary regions
        kernel = np.ones((boundary_width, boundary_width), np.uint8)
        inner = cv2.erode((mask > 0.5).astype(np.uint8), kernel)
        outer = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
        
        inner_boundary = (mask > 0.5).astype(np.uint8) - inner
        outer_boundary = outer - (mask > 0.5).astype(np.uint8)
        
        # Compare gradient statistics
        if inner_boundary.sum() == 0 or outer_boundary.sum() == 0:
            return 1.0
        
        inner_grad_mag = np.sqrt(grad_x**2 + grad_y**2)[inner_boundary > 0]
        outer_grad_mag = np.sqrt(grad_x**2 + grad_y**2)[outer_boundary > 0]
        
        # Compare distributions
        inner_mean = inner_grad_mag.mean()
        outer_mean = outer_grad_mag.mean()
        
        # Similar gradient magnitudes = good continuity
        diff = abs(inner_mean - outer_mean)
        max_val = max(inner_mean, outer_mean, 1)
        
        return 1.0 - min(diff / max_val, 1.0)


class ColorHarmonyChecker:
    """
    Check color harmony between generated region and background.
    """
    
    @staticmethod
    def check_color_distribution(
        image: np.ndarray,
        mask: np.ndarray,
        context_dilation: int = 30
    ) -> float:
        """
        Compare color distributions inside and around the mask.
        
        Args:
            image: Full image
            mask: Mask of generated region
            context_dilation: Pixels to sample around mask
            
        Returns:
            Harmony score (0-1), higher = better match
        """
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        # Get context region
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_dilation * 2 + 1, context_dilation * 2 + 1)
        )
        dilated = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
        context = dilated.astype(bool) & ~(mask > 0.5)
        
        if not context.any() or not (mask > 0.5).any():
            return 1.0
        
        # Convert to LAB for better color comparison
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Get color statistics
        inner = lab[mask > 0.5]
        outer = lab[context]
        
        # Compare histograms for each channel
        scores = []
        for c in range(3):
            inner_hist, _ = np.histogram(inner[:, c], bins=32, range=(0, 256), density=True)
            outer_hist, _ = np.histogram(outer[:, c], bins=32, range=(0, 256), density=True)
            
            # Histogram intersection
            intersection = np.minimum(inner_hist, outer_hist).sum()
            scores.append(intersection)
        
        return float(np.mean(scores))
    
    @staticmethod
    def check_brightness_match(
        image: np.ndarray,
        mask: np.ndarray,
        context_dilation: int = 30
    ) -> float:
        """
        Check if brightness matches between inside and outside mask.
        """
        if image.ndim == 3:
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Get regions
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (context_dilation * 2 + 1, context_dilation * 2 + 1)
        )
        dilated = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
        context = dilated.astype(bool) & ~(mask > 0.5)
        
        if not context.any() or not (mask > 0.5).any():
            return 1.0
        
        inner_mean = gray[mask > 0.5].mean()
        outer_mean = gray[context].mean()
        
        # Similar brightness = good match
        diff = abs(inner_mean - outer_mean)
        max_val = max(inner_mean, outer_mean, 1)
        
        return 1.0 - min(diff / max_val, 1.0)


def compute_all_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str = "",
    reference_image: Optional[np.ndarray] = None,
    clip_model: Optional[CLIPScorer] = None
) -> QualityMetrics:
    """
    Compute all quality metrics for a generated image.
    
    Args:
        image: Generated image
        mask: Mask of generated region
        prompt: Text prompt used for generation
        reference_image: Optional reference image for comparison
        clip_model: Optional pre-loaded CLIP model
        
    Returns:
        QualityMetrics object with all scores
    """
    metrics = QualityMetrics()
    
    # CLIP score
    if prompt and clip_model is not None:
        metrics.clip_score = clip_model.score(image, prompt)
    else:
        metrics.clip_score = 0.5  # Neutral if no CLIP
    
    # Blur/artifact score
    blur_score = BlurDetector.score(image, mask)
    metrics.blur_score = blur_score
    metrics.artifact_score = 1.0 - blur_score  # Invert: lower blur = lower artifact
    
    # Edge consistency
    boundary_score = EdgeConsistencyChecker.check_boundary_continuity(image, mask)
    gradient_score = EdgeConsistencyChecker.check_gradient_continuity(image, mask)
    metrics.edge_consistency = (boundary_score + gradient_score) / 2
    
    # Color harmony
    color_score = ColorHarmonyChecker.check_color_distribution(image, mask)
    brightness_score = ColorHarmonyChecker.check_brightness_match(image, mask)
    metrics.color_harmony = (color_score + brightness_score) / 2
    
    # Geometry score (placeholder - would compare with reference)
    if reference_image is not None:
        # Could use structural similarity or feature matching
        metrics.geometry_score = 0.8  # Placeholder
    else:
        metrics.geometry_score = 0.8
    
    # Overall score (weighted average)
    weights = {
        "clip": 0.15,
        "artifact": 0.20,
        "edge": 0.25,
        "color": 0.25,
        "geometry": 0.15
    }
    
    metrics.overall_score = (
        weights["clip"] * metrics.clip_score +
        weights["artifact"] * (1 - metrics.artifact_score) +  # Lower artifact = better
        weights["edge"] * metrics.edge_consistency +
        weights["color"] * metrics.color_harmony +
        weights["geometry"] * metrics.geometry_score
    )
    
    return metrics

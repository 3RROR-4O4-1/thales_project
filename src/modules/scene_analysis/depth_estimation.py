"""
Depth Estimation Module

Generates depth maps from monocular images using state-of-the-art models.
Supports Depth Anything V2, ZoeDepth, and MiDaS.
"""

import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthEstimationConfig:
    """Configuration for depth estimation."""
    model: str = "depth-anything-v2-large"  # or "zoedepth", "midas"
    device: str = "cuda"
    normalize: bool = True
    invert: bool = False  # Some models output inverted depth
    resize_to: Optional[Tuple[int, int]] = None  # Resize input before inference


class DepthEstimator:
    """
    Depth estimation from monocular images.
    
    Supports multiple backends:
    - Depth Anything V2 (recommended)
    - ZoeDepth
    - MiDaS
    """
    
    def __init__(self, config: Optional[DepthEstimationConfig] = None):
        self.config = config or DepthEstimationConfig()
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load the depth estimation model."""
        model_name = self.config.model.lower()
        
        if "depth-anything" in model_name or "depthanything" in model_name:
            self._load_depth_anything()
        elif "zoe" in model_name:
            self._load_zoedepth()
        elif "midas" in model_name:
            self._load_midas()
        else:
            logger.warning(f"Unknown model {model_name}, defaulting to Depth Anything V2")
            self._load_depth_anything()
    
    def _load_depth_anything(self):
        """Load Depth Anything V2 model."""
        try:
            from transformers import pipeline
            
            # Determine model variant
            if "large" in self.config.model.lower():
                model_id = "depth-anything/Depth-Anything-V2-Large-hf"
            elif "base" in self.config.model.lower():
                model_id = "depth-anything/Depth-Anything-V2-Base-hf"
            else:
                model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            
            self.model = pipeline(
                "depth-estimation",
                model=model_id,
                device=0 if self.config.device == "cuda" else -1
            )
            self._model_type = "depth_anything"
            logger.info(f"Loaded Depth Anything V2: {model_id}")
            
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise
    
    def _load_zoedepth(self):
        """Load ZoeDepth model."""
        try:
            import torch
            
            self.model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_NK",
                pretrained=True
            )
            
            if self.config.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            self._model_type = "zoedepth"
            logger.info("Loaded ZoeDepth")
            
        except Exception as e:
            logger.error(f"Failed to load ZoeDepth: {e}")
            raise
    
    def _load_midas(self):
        """Load MiDaS model."""
        try:
            import torch
            
            model_type = "DPT_Large"  # or "DPT_Hybrid", "MiDaS_small"
            
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            
            if self.config.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            
            self._model_type = "midas"
            logger.info(f"Loaded MiDaS: {model_type}")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            raise
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.
        
        Args:
            image: RGB image (H, W, 3), uint8 or float32
            
        Returns:
            Depth map (H, W), float32, normalized to 0-1 if config.normalize=True
        """
        import torch
        from PIL import Image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Resize if configured
        original_size = pil_image.size  # (W, H)
        if self.config.resize_to:
            pil_image = pil_image.resize(self.config.resize_to, Image.BILINEAR)
        
        # Run inference based on model type
        if self._model_type == "depth_anything":
            result = self.model(pil_image)
            depth = np.array(result["depth"])
            
        elif self._model_type == "zoedepth":
            with torch.no_grad():
                depth = self.model.infer_pil(pil_image)
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                if depth.ndim == 3:
                    depth = depth.squeeze()
                    
        elif self._model_type == "midas":
            input_tensor = self.transform(pil_image)
            if self.config.device == "cuda":
                input_tensor = input_tensor.cuda()
            
            with torch.no_grad():
                depth = self.model(input_tensor)
                depth = depth.squeeze().cpu().numpy()
        
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
        
        # Resize back to original size if needed
        if self.config.resize_to and depth.shape[::-1] != original_size:
            import cv2
            depth = cv2.resize(depth, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Invert if needed (some models output inverse depth)
        if self.config.invert:
            depth = 1.0 / (depth + 1e-6)
        
        # Normalize to 0-1
        if self.config.normalize:
            depth = self._normalize_depth(depth)
        
        return depth.astype(np.float32)
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to 0-1 range."""
        d_min = depth.min()
        d_max = depth.max()
        
        if d_max - d_min > 1e-6:
            return (depth - d_min) / (d_max - d_min)
        else:
            return np.zeros_like(depth)
    
    def estimate_batch(self, images: list) -> list:
        """
        Estimate depth for a batch of images.
        
        Args:
            images: List of RGB images
            
        Returns:
            List of depth maps
        """
        return [self.estimate(img) for img in images]


def get_depth_statistics(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Compute statistics of a depth map.
    
    Args:
        depth: Depth map (H, W)
        mask: Optional mask to compute stats within
        
    Returns:
        Dict with min, max, mean, std, range
    """
    if mask is not None:
        values = depth[mask > 0.5]
    else:
        values = depth.flatten()
    
    if len(values) == 0:
        return {
            "min": 0, "max": 0, "mean": 0, "std": 0, "range": 0
        }
    
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "range": float(values.max() - values.min())
    }


def depth_to_colormap(
    depth: np.ndarray,
    colormap: str = "turbo",
    normalize: bool = True
) -> np.ndarray:
    """
    Convert depth map to colored visualization.
    
    Args:
        depth: Depth map (H, W)
        colormap: Matplotlib colormap name
        normalize: Normalize depth to 0-1 first
        
    Returns:
        Colored depth image (H, W, 3), uint8
    """
    import matplotlib.pyplot as plt
    
    if normalize:
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
    
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth)[:, :, :3]  # Remove alpha
    colored = (colored * 255).astype(np.uint8)
    
    return colored


# Convenience function
def estimate_depth(
    image: np.ndarray,
    model: str = "depth-anything-v2-large",
    device: str = "cuda"
) -> np.ndarray:
    """
    Quick depth estimation with default settings.
    
    Args:
        image: RGB image
        model: Model name
        device: Device to use
        
    Returns:
        Normalized depth map
    """
    config = DepthEstimationConfig(model=model, device=device)
    estimator = DepthEstimator(config)
    return estimator.estimate(image)

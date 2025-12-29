"""
Scale-Aware Inpainting Module

Handles the crop-upscale-process-downscale-stitch workflow for inserting
small objects (like distant vehicles) with preserved detail.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import logging

from ..utils import BoundingBox, normalize_image, denormalize_image, resize_image

logger = logging.getLogger(__name__)


@dataclass
class ScaleAwareConfig:
    """Configuration for scale-aware inpainting."""
    # Processing resolution
    processing_resolution: int = 1024
    min_crop_size: int = 256
    max_crop_size: int = 2048
    
    # Context expansion
    context_expand_factor: float = 2.0
    context_expand_pixels: int = 96
    
    # Mask processing
    mask_blur_pixels: int = 28
    mask_dilate_pixels: int = 4
    
    # Resize methods
    upscale_method: str = "lanczos"
    downscale_method: str = "lanczos"
    mask_resize_method: str = "nearest"
    
    # Quality
    antialias_downscale: bool = True
    preserve_aspect_ratio: bool = True


@dataclass
class CropRegion:
    """Represents a cropped region for processing."""
    image: np.ndarray
    mask: np.ndarray
    bbox: BoundingBox
    original_size: Tuple[int, int]  # (height, width) of crop before upscale
    processing_size: Tuple[int, int]  # (height, width) after upscale
    scale_factor: float
    
    # Optional additional data
    depth: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    vehicle_reference: Optional[np.ndarray] = None


class ScaleAwareInpainter:
    """
    Handles scale-aware inpainting for small object insertion.
    
    The key insight is that processing small regions at native resolution
    produces blurry results. Instead, we:
    1. Crop with expanded context
    2. Upscale to processing resolution
    3. Run inpainting at high resolution
    4. Downscale with antialiasing
    5. Stitch back with feathered blending
    """
    
    def __init__(
        self,
        config: Optional[ScaleAwareConfig] = None,
        inpaint_fn: Optional[Callable] = None
    ):
        """
        Args:
            config: Scale-aware configuration
            inpaint_fn: Function that performs actual inpainting
                       Signature: (image, mask, **kwargs) -> image
        """
        self.config = config or ScaleAwareConfig()
        self.inpaint_fn = inpaint_fn
        
        # Import harmonization module
        from ..harmonization import CropStitcher
        self.stitcher = CropStitcher(
            feather_pixels=self.config.mask_blur_pixels,
            downscale_method=self.config.downscale_method
        )
    
    def process(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        vehicle_render: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        normal_map: Optional[np.ndarray] = None,
        inpaint_kwargs: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Full scale-aware inpainting pipeline.
        
        Args:
            image: Input image (H, W, C)
            mask: Binary mask for inpainting region (H, W)
            vehicle_render: Optional vehicle reference image
            depth_map: Optional depth map for ControlNet
            normal_map: Optional normal map for ControlNet
            inpaint_kwargs: Additional kwargs for inpainting function
            
        Returns:
            Inpainted image with vehicle inserted
        """
        logger.info("Starting scale-aware inpainting")
        
        # Step 1: Extract and prepare crop
        crop_region = self._extract_crop(image, mask, depth_map, normal_map)
        crop_region.vehicle_reference = vehicle_render
        
        logger.debug(f"Crop: {crop_region.original_size} -> {crop_region.processing_size}")
        logger.debug(f"Scale factor: {crop_region.scale_factor:.2f}x")
        
        # Step 2: Upscale crop to processing resolution
        upscaled = self._upscale_crop(crop_region)
        
        # Step 3: Run inpainting
        if self.inpaint_fn is not None:
            kwargs = inpaint_kwargs or {}
            
            # Add conditioning if available
            if upscaled.depth is not None:
                kwargs['depth'] = upscaled.depth
            if upscaled.normal is not None:
                kwargs['normal'] = upscaled.normal
            if upscaled.vehicle_reference is not None:
                kwargs['reference'] = upscaled.vehicle_reference
            
            inpainted = self.inpaint_fn(upscaled.image, upscaled.mask, **kwargs)
        else:
            # Placeholder: just return the input
            logger.warning("No inpainting function provided, returning input")
            inpainted = upscaled.image
        
        # Step 4: Downscale with antialiasing
        downscaled = self._downscale_result(inpainted, crop_region)
        
        # Step 5: Stitch back into original
        result = self._stitch_result(image, downscaled, crop_region)
        
        logger.info("Scale-aware inpainting complete")
        return result
    
    def _extract_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None
    ) -> CropRegion:
        """Extract crop region with expanded context."""
        
        # Get mask bounding box
        bbox = BoundingBox.from_mask(mask)
        
        if bbox.area == 0:
            raise ValueError("Empty mask provided")
        
        # Expand bounding box for context
        expanded = bbox.expand(
            pixels=self.config.context_expand_pixels,
            factor=self.config.context_expand_factor
        )
        
        # Ensure minimum size
        if expanded.width < self.config.min_crop_size:
            expand_w = (self.config.min_crop_size - expanded.width) // 2
            expanded = BoundingBox(
                expanded.x - expand_w,
                expanded.y,
                self.config.min_crop_size,
                expanded.height
            )
        if expanded.height < self.config.min_crop_size:
            expand_h = (self.config.min_crop_size - expanded.height) // 2
            expanded = BoundingBox(
                expanded.x,
                expanded.y - expand_h,
                expanded.width,
                self.config.min_crop_size
            )
        
        # Clip to image bounds
        h, w = image.shape[:2]
        clipped = expanded.clip(w, h)
        
        # Extract crops
        x, y, cw, ch = clipped.to_xywh()
        crop_image = image[y:y+ch, x:x+cw].copy()
        crop_mask = mask[y:y+ch, x:x+cw].copy()
        
        # Process mask
        crop_mask = self._process_mask(crop_mask)
        
        # Extract depth/normal if available
        crop_depth = depth[y:y+ch, x:x+cw].copy() if depth is not None else None
        crop_normal = normal[y:y+ch, x:x+cw].copy() if normal is not None else None
        
        # Calculate scale factor
        max_dim = max(ch, cw)
        scale_factor = self.config.processing_resolution / max_dim
        
        # Compute processing size
        if self.config.preserve_aspect_ratio:
            proc_h = int(ch * scale_factor)
            proc_w = int(cw * scale_factor)
        else:
            proc_h = proc_w = self.config.processing_resolution
        
        return CropRegion(
            image=crop_image,
            mask=crop_mask,
            bbox=clipped,
            original_size=(ch, cw),
            processing_size=(proc_h, proc_w),
            scale_factor=scale_factor,
            depth=crop_depth,
            normal=crop_normal
        )
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply mask preprocessing (dilate + blur)."""
        
        # Ensure binary
        mask = (mask > 0.5).astype(np.uint8)
        
        # Dilate
        if self.config.mask_dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.mask_dilate_pixels * 2 + 1,
                 self.config.mask_dilate_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)
        
        # Convert to float for blur
        mask = mask.astype(np.float32)
        
        # Blur edges
        if self.config.mask_blur_pixels > 0:
            mask = cv2.GaussianBlur(
                mask,
                (0, 0),
                self.config.mask_blur_pixels / 3  # sigma from pixels
            )
        
        return mask
    
    def _upscale_crop(self, crop: CropRegion) -> CropRegion:
        """Upscale crop to processing resolution."""
        
        proc_h, proc_w = crop.processing_size
        
        # Get interpolation method
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'area': cv2.INTER_AREA
        }
        interp = interp_map.get(self.config.upscale_method, cv2.INTER_LANCZOS4)
        
        # Upscale image
        upscaled_image = cv2.resize(crop.image, (proc_w, proc_h), interpolation=interp)
        
        # Upscale mask (use nearest to preserve binary nature)
        upscaled_mask = cv2.resize(crop.mask, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        
        # Upscale depth/normal if available
        upscaled_depth = None
        upscaled_normal = None
        
        if crop.depth is not None:
            upscaled_depth = cv2.resize(crop.depth, (proc_w, proc_h), interpolation=interp)
        if crop.normal is not None:
            upscaled_normal = cv2.resize(crop.normal, (proc_w, proc_h), interpolation=interp)
        
        # Upscale vehicle reference if needed
        upscaled_ref = None
        if crop.vehicle_reference is not None:
            ref = crop.vehicle_reference
            # Resize reference to match processing size
            upscaled_ref = cv2.resize(ref, (proc_w, proc_h), interpolation=interp)
        
        return CropRegion(
            image=upscaled_image,
            mask=upscaled_mask,
            bbox=crop.bbox,
            original_size=crop.original_size,
            processing_size=crop.processing_size,
            scale_factor=crop.scale_factor,
            depth=upscaled_depth,
            normal=upscaled_normal,
            vehicle_reference=upscaled_ref
        )
    
    def _downscale_result(
        self,
        inpainted: np.ndarray,
        crop: CropRegion
    ) -> np.ndarray:
        """Downscale inpainted result back to original crop size."""
        
        orig_h, orig_w = crop.original_size
        
        # Get interpolation method
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'area': cv2.INTER_AREA
        }
        
        if self.config.antialias_downscale:
            # Use AREA interpolation for antialiased downscaling
            interp = cv2.INTER_AREA
        else:
            interp = interp_map.get(self.config.downscale_method, cv2.INTER_LANCZOS4)
        
        downscaled = cv2.resize(inpainted, (orig_w, orig_h), interpolation=interp)
        
        return downscaled
    
    def _stitch_result(
        self,
        original: np.ndarray,
        processed_crop: np.ndarray,
        crop: CropRegion
    ) -> np.ndarray:
        """Stitch processed crop back into original image."""
        
        x, y, w, h = crop.bbox.to_xywh()
        
        # Downscale mask for stitching
        orig_h, orig_w = crop.original_size
        stitch_mask = cv2.resize(
            crop.mask,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Use the stitcher
        result = self.stitcher.stitch(
            original,
            processed_crop,
            stitch_mask,
            (x, y, w, h),
            original_crop_size=crop.original_size
        )
        
        return result
    
    def process_batch(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        vehicle_renders: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Process multiple insertion regions in a single image.
        
        Args:
            image: Input image
            masks: List of masks for each insertion
            vehicle_renders: Optional list of vehicle references
            **kwargs: Additional inpainting arguments
            
        Returns:
            Image with all vehicles inserted
        """
        result = image.copy()
        
        for i, mask in enumerate(masks):
            vehicle = vehicle_renders[i] if vehicle_renders else None
            
            try:
                result = self.process(result, mask, vehicle_render=vehicle, **kwargs)
            except Exception as e:
                logger.error(f"Failed to process region {i}: {e}")
                continue
        
        return result


def compute_optimal_processing_size(
    mask: np.ndarray,
    min_size: int = 512,
    max_size: int = 1024,
    target_object_pixels: int = 256
) -> int:
    """
    Compute optimal processing resolution based on mask size.
    
    The goal is to ensure the object (mask region) has enough pixels
    for detailed generation.
    
    Args:
        mask: Binary mask
        min_size: Minimum processing resolution
        max_size: Maximum processing resolution
        target_object_pixels: Target size for object in processed image
        
    Returns:
        Optimal processing resolution
    """
    bbox = BoundingBox.from_mask(mask)
    
    if bbox.area == 0:
        return min_size
    
    # Current object size
    object_size = max(bbox.width, bbox.height)
    
    # Scale needed to reach target
    scale = target_object_pixels / object_size
    
    # Apply to get processing size
    processing_size = int(object_size * scale * 2)  # 2x for context
    
    # Clamp to range
    processing_size = max(min_size, min(max_size, processing_size))
    
    # Round to multiple of 64 (good for neural networks)
    processing_size = (processing_size // 64) * 64
    
    return processing_size


# Convenience function
def scale_aware_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    inpaint_fn: Callable,
    processing_resolution: int = 1024,
    **kwargs
) -> np.ndarray:
    """
    Quick scale-aware inpainting with sensible defaults.
    
    Args:
        image: Input image
        mask: Inpainting mask
        inpaint_fn: Inpainting function (image, mask, **kwargs) -> image
        processing_resolution: Resolution for processing
        **kwargs: Additional arguments for inpainting
        
    Returns:
        Inpainted image
    """
    config = ScaleAwareConfig(processing_resolution=processing_resolution)
    inpainter = ScaleAwareInpainter(config, inpaint_fn)
    return inpainter.process(image, mask, inpaint_kwargs=kwargs)

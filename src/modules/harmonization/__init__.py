"""
Edge Harmonization Module

Main integration module that combines differential blending, Poisson compositing,
color transfer, and shadow generation for seamless vehicle insertion.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
import logging

from .differential_blend import (
    DifferentialBlender,
    create_differential_mask,
    create_edge_feather_mask,
    apply_differential_blend
)
from .poisson import PoissonBlender, poisson_blend
from .color_transfer import ColorHarmonizer, harmonize_colors, match_lighting
from .shadow import ShadowGenerator, estimate_light_direction

logger = logging.getLogger(__name__)


@dataclass
class HarmonizationConfig:
    """Configuration for harmonization pipeline."""
    # Blending
    blend_method: str = "differential"  # differential, poisson, hybrid
    feather_pixels: int = 32
    use_dual_mask: bool = True
    inner_feather: int = 8
    
    # Color
    color_harmonization: bool = True
    color_method: str = "lab_transfer"
    color_strength: float = 0.8
    context_dilation: int = 20
    match_lighting_enabled: bool = True
    
    # Shadow
    shadow_generation: bool = True
    contact_shadow: bool = True
    cast_shadow: bool = True
    shadow_intensity: float = 0.4
    
    # Fallback
    fallback_to_poisson: bool = True


class EdgeHarmonizer:
    """
    Complete edge harmonization pipeline.
    
    Handles the full process of blending a generated/inpainted region
    back into the original image seamlessly.
    """
    
    def __init__(self, config: Optional[HarmonizationConfig] = None):
        self.config = config or HarmonizationConfig()
        
        # Initialize sub-modules
        self.differential_blender = DifferentialBlender(
            feather_pixels=self.config.feather_pixels,
            use_dual_mask=self.config.use_dual_mask,
            inner_feather=self.config.inner_feather
        )
        
        self.poisson_blender = PoissonBlender(
            mode="normal",
            use_region_optimization=True,
            fallback_on_error=True
        )
        
        self.color_harmonizer = ColorHarmonizer(
            method=self.config.color_method,
            context_dilation=self.config.context_dilation,
            strength=self.config.color_strength,
            match_lighting=self.config.match_lighting_enabled
        )
        
        self.shadow_generator = ShadowGenerator(
            contact_shadow=self.config.contact_shadow,
            cast_shadow=self.config.cast_shadow,
            contact_intensity=self.config.shadow_intensity,
            cast_intensity=self.config.shadow_intensity * 0.7
        )
    
    def harmonize(
        self,
        background: np.ndarray,
        generated: np.ndarray,
        mask: np.ndarray,
        vehicle_mask: Optional[np.ndarray] = None,
        light_direction: Optional[Tuple[float, float]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Full harmonization pipeline.
        
        Args:
            background: Original background image (H, W, C)
            generated: Generated/inpainted image (H, W, C)
            mask: Mask of the modified region (H, W)
            vehicle_mask: Optional separate mask for the vehicle itself
            light_direction: Optional light direction for shadow casting
            bbox: Optional bounding box of the region (x, y, w, h)
            
        Returns:
            Harmonized composite image
        """
        logger.info("Starting harmonization pipeline")
        
        # Use mask as vehicle_mask if not provided separately
        if vehicle_mask is None:
            vehicle_mask = mask
        
        result = generated.copy()
        
        # Step 1: Color harmonization
        if self.config.color_harmonization:
            logger.debug("Applying color harmonization")
            result = self.color_harmonizer.harmonize(result, background, mask)
        
        # Step 2: Shadow generation
        if self.config.shadow_generation:
            logger.debug("Generating shadows")
            
            # Estimate light direction if not provided
            if light_direction is None:
                light_direction = estimate_light_direction(background)
            
            result = self.shadow_generator.apply(
                result,
                vehicle_mask,
                light_direction=light_direction
            )
        
        # Step 3: Blending
        logger.debug(f"Applying {self.config.blend_method} blending")
        
        if self.config.blend_method == "differential":
            result = self.differential_blender.blend(background, result, mask)
            
        elif self.config.blend_method == "poisson":
            result = self.poisson_blender.blend(background, result, mask, bbox)
            
        elif self.config.blend_method == "hybrid":
            # Hybrid: differential for outer edges, Poisson for inner content
            result = self._hybrid_blend(background, result, mask, vehicle_mask, bbox)
        
        logger.info("Harmonization complete")
        return result
    
    def _hybrid_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        vehicle_mask: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Hybrid blending: Poisson for vehicle, differential for surrounding.
        """
        # First, Poisson blend the vehicle itself
        try:
            poisson_result = self.poisson_blender.blend(
                background, foreground, vehicle_mask, bbox
            )
        except Exception as e:
            logger.warning(f"Poisson blend failed, using differential only: {e}")
            poisson_result = foreground
        
        # Then, differential blend the outer mask region
        # (areas in mask but not in vehicle_mask)
        outer_mask = mask.astype(float) - vehicle_mask.astype(float)
        outer_mask = np.clip(outer_mask, 0, 1)
        
        if outer_mask.sum() > 0:
            result = apply_differential_blend(
                background, poisson_result, mask,
                feather_pixels=self.config.feather_pixels
            )
        else:
            result = poisson_result
        
        return result
    
    def stitch_crop(
        self,
        original: np.ndarray,
        crop: np.ndarray,
        crop_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        vehicle_mask_in_crop: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Stitch a processed crop back into the original image.
        
        This is specifically for the crop-upscale-process-downscale-stitch workflow.
        
        Args:
            original: Original full image
            crop: Processed crop region
            crop_mask: Mask within the crop
            bbox: Position of crop in original (x, y, w, h)
            vehicle_mask_in_crop: Vehicle mask within the crop
            
        Returns:
            Stitched image
        """
        x, y, w, h = bbox
        
        # Extract corresponding region from original
        original_region = original[y:y+h, x:x+w]
        
        # Create stitch mask with edge feathering
        stitch_mask = self.differential_blender.create_stitch_mask(
            (h, w), vehicle_mask_in_crop
        )
        
        # Harmonize and blend
        harmonized_crop = self.harmonize(
            original_region,
            crop,
            crop_mask,
            vehicle_mask_in_crop,
            bbox=(0, 0, w, h)
        )
        
        # Place back into original
        result = original.copy()
        result[y:y+h, x:x+w] = harmonized_crop
        
        return result


class CropStitcher:
    """
    Specialized class for crop/stitch operations with edge-aware blending.
    """
    
    def __init__(
        self,
        feather_pixels: int = 32,
        use_poisson_fallback: bool = True,
        downscale_method: str = "lanczos"
    ):
        self.feather_pixels = feather_pixels
        self.use_poisson_fallback = use_poisson_fallback
        self.downscale_method = downscale_method
        
        self.harmonizer = EdgeHarmonizer(HarmonizationConfig(
            blend_method="hybrid" if use_poisson_fallback else "differential",
            feather_pixels=feather_pixels,
            color_harmonization=True,
            shadow_generation=False  # Shadows handled separately for crops
        ))
    
    def extract_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        context_factor: float = 2.0,
        context_pixels: int = 96,
        min_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract crop region around mask with expanded context.
        
        Args:
            image: Source image
            mask: Binary mask
            context_factor: Expand bbox by this factor
            context_pixels: Additional pixels to add
            min_size: Minimum crop dimension
            
        Returns:
            Tuple of (crop_image, crop_mask, bbox)
        """
        from ..utils import BoundingBox
        
        # Get mask bbox
        bbox = BoundingBox.from_mask(mask)
        
        # Expand
        expanded = bbox.expand(pixels=context_pixels, factor=context_factor)
        
        # Ensure minimum size
        if expanded.width < min_size:
            expanded = BoundingBox(
                expanded.x - (min_size - expanded.width) // 2,
                expanded.y,
                min_size,
                expanded.height
            )
        if expanded.height < min_size:
            expanded = BoundingBox(
                expanded.x,
                expanded.y - (min_size - expanded.height) // 2,
                expanded.width,
                min_size
            )
        
        # Clip to image bounds
        h, w = image.shape[:2]
        clipped = expanded.clip(w, h)
        
        # Extract
        x, y, cw, ch = clipped.to_xywh()
        crop_image = image[y:y+ch, x:x+cw].copy()
        crop_mask = mask[y:y+ch, x:x+cw].copy()
        
        return crop_image, crop_mask, (x, y, cw, ch)
    
    def stitch(
        self,
        original: np.ndarray,
        processed_crop: np.ndarray,
        crop_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        original_crop_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Stitch processed crop back into original image.
        
        Args:
            original: Original full image
            processed_crop: Processed crop (possibly at different resolution)
            crop_mask: Mask in crop coordinates
            bbox: Crop position in original (x, y, w, h)
            original_crop_size: If processed at different res, original size
            
        Returns:
            Stitched image
        """
        import cv2
        
        x, y, w, h = bbox
        
        # Resize if processed at different resolution
        if original_crop_size is not None:
            target_h, target_w = original_crop_size
            if processed_crop.shape[:2] != (target_h, target_w):
                interp = cv2.INTER_LANCZOS4 if self.downscale_method == "lanczos" else cv2.INTER_LINEAR
                processed_crop = cv2.resize(processed_crop, (target_w, target_h), interpolation=interp)
                crop_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Get original region
        original_region = original[y:y+h, x:x+w]
        
        # Create feathered stitch mask
        edge_mask = create_edge_feather_mask(h, w, self.feather_pixels)
        content_mask = create_differential_mask(crop_mask, self.feather_pixels // 2)
        
        # Combined mask: content + edge feathering
        stitch_mask = content_mask * edge_mask
        stitch_mask = stitch_mask[:, :, np.newaxis] if stitch_mask.ndim == 2 else stitch_mask
        
        # Blend
        blended = original_region * (1 - stitch_mask) + processed_crop * stitch_mask
        
        # Place in result
        result = original.copy()
        result[y:y+h, x:x+w] = blended.astype(original.dtype)
        
        return result


# Convenience function for quick harmonization
def harmonize(
    background: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    method: str = "hybrid",
    feather: int = 32
) -> np.ndarray:
    """
    Quick harmonization with sensible defaults.
    
    Args:
        background: Original background
        generated: Generated/modified image
        mask: Mask of modified region
        method: "differential", "poisson", or "hybrid"
        feather: Feather pixels for blending
        
    Returns:
        Harmonized image
    """
    config = HarmonizationConfig(
        blend_method=method,
        feather_pixels=feather
    )
    harmonizer = EdgeHarmonizer(config)
    return harmonizer.harmonize(background, generated, mask)

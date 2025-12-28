"""
Differential Blending Module

Creates smooth gradient masks for seamless stitching - full strength at center,
gradual falloff to edges to prevent visible seams.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from typing import Tuple, Optional
import cv2


def create_differential_mask(
    binary_mask: np.ndarray,
    feather_pixels: int = 32,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> np.ndarray:
    """
    Create a differential blending mask from a binary mask.
    
    The mask has value 1.0 at the center (far from edges) and gradually
    falls off to 0.0 at the mask boundary.
    
    Args:
        binary_mask: Binary mask (H, W) with 1s inside region of interest
        feather_pixels: Number of pixels for the falloff gradient
        min_value: Minimum mask value at edges
        max_value: Maximum mask value at center
        
    Returns:
        Soft mask with gradient falloff at edges
    """
    # Ensure binary
    binary_mask = (binary_mask > 0.5).astype(np.float32)
    
    # Distance from mask edge (inward)
    dist_inside = distance_transform_edt(binary_mask)
    
    # Normalize: center = max_value, edge = min_value
    if feather_pixels > 0:
        soft_mask = np.clip(dist_inside / feather_pixels, 0, 1)
    else:
        soft_mask = binary_mask
    
    # Scale to desired range
    soft_mask = soft_mask * (max_value - min_value) + min_value
    
    return soft_mask.astype(np.float32)


def create_edge_feather_mask(
    height: int,
    width: int,
    feather_pixels: int = 32
) -> np.ndarray:
    """
    Create a mask that feathers from edges of a rectangular region.
    
    Useful for crop/stitch workflows where you want to blend the entire
    crop region smoothly with the background.
    
    Args:
        height: Height of the region
        width: Width of the region
        feather_pixels: Pixels of falloff from edges
        
    Returns:
        Mask with 1.0 at center, 0.0 at edges
    """
    # Distance from each edge
    y_dist = np.minimum(
        np.arange(height)[:, None],
        np.arange(height - 1, -1, -1)[:, None]
    )
    x_dist = np.minimum(
        np.arange(width)[None, :],
        np.arange(width - 1, -1, -1)[None, :]
    )
    
    # Minimum distance to any edge
    edge_dist = np.minimum(
        np.broadcast_to(y_dist, (height, width)),
        np.broadcast_to(x_dist, (height, width))
    )
    
    # Normalize by feather distance
    mask = np.clip(edge_dist / max(feather_pixels, 1), 0, 1)
    
    return mask.astype(np.float32)


def create_dual_mask(
    binary_mask: np.ndarray,
    inner_feather: int = 8,
    outer_feather: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dual masks for two-pass blending.
    
    - Inner mask: Sharp edges for the generated content (vehicle)
    - Outer mask: Soft edges for blending with background
    
    Args:
        binary_mask: Original binary mask
        inner_feather: Feather for inner (content) mask
        outer_feather: Feather for outer (blend) mask
        
    Returns:
        Tuple of (inner_mask, outer_mask)
    """
    inner_mask = create_differential_mask(binary_mask, inner_feather)
    outer_mask = create_differential_mask(binary_mask, outer_feather)
    
    return inner_mask, outer_mask


def apply_differential_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    soft_mask: Optional[np.ndarray] = None,
    feather_pixels: int = 32
) -> np.ndarray:
    """
    Blend foreground into background using differential mask.
    
    Args:
        background: Background image (H, W, C)
        foreground: Foreground image to blend in (H, W, C)
        mask: Binary mask indicating foreground region
        soft_mask: Pre-computed soft mask (optional)
        feather_pixels: Feather amount if soft_mask not provided
        
    Returns:
        Blended image
    """
    if soft_mask is None:
        soft_mask = create_differential_mask(mask, feather_pixels)
    
    # Ensure mask has channel dimension for broadcasting
    if soft_mask.ndim == 2:
        soft_mask = soft_mask[:, :, np.newaxis]
    
    # Blend
    result = background * (1 - soft_mask) + foreground * soft_mask
    
    return result.astype(background.dtype)


def create_radial_gradient_mask(
    height: int,
    width: int,
    center: Optional[Tuple[int, int]] = None,
    radius: Optional[float] = None,
    falloff: str = 'linear'
) -> np.ndarray:
    """
    Create a radial gradient mask centered on a point.
    
    Args:
        height: Mask height
        width: Mask width
        center: Center point (x, y), defaults to image center
        radius: Radius of full-strength region, defaults to min(h,w)/4
        falloff: 'linear', 'quadratic', or 'gaussian'
        
    Returns:
        Radial gradient mask
    """
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(height, width) / 4
    
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Normalize distance
    norm_dist = dist / (radius + 1e-8)
    
    if falloff == 'linear':
        mask = np.clip(1 - norm_dist, 0, 1)
    elif falloff == 'quadratic':
        mask = np.clip(1 - norm_dist**2, 0, 1)
    elif falloff == 'gaussian':
        mask = np.exp(-0.5 * norm_dist**2)
    else:
        mask = np.clip(1 - norm_dist, 0, 1)
    
    return mask.astype(np.float32)


class DifferentialBlender:
    """
    High-level class for differential blending operations.
    """
    
    def __init__(
        self,
        feather_pixels: int = 32,
        use_dual_mask: bool = False,
        inner_feather: int = 8
    ):
        self.feather_pixels = feather_pixels
        self.use_dual_mask = use_dual_mask
        self.inner_feather = inner_feather
    
    def blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Blend foreground into background with differential mask.
        """
        if self.use_dual_mask:
            # Two-pass blending
            inner_mask, outer_mask = create_dual_mask(
                mask, self.inner_feather, self.feather_pixels
            )
            
            # First pass: sharp content
            temp = apply_differential_blend(
                background, foreground, mask, inner_mask
            )
            
            # Second pass: soft edges
            result = apply_differential_blend(
                background, temp, mask, outer_mask
            )
        else:
            # Single-pass blending
            result = apply_differential_blend(
                background, foreground, mask,
                feather_pixels=self.feather_pixels
            )
        
        return result
    
    def create_stitch_mask(
        self,
        crop_shape: Tuple[int, int],
        inner_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create mask for stitching a crop back into the original image.
        
        Args:
            crop_shape: (height, width) of the crop region
            inner_mask: Optional mask within the crop (e.g., vehicle mask)
            
        Returns:
            Stitch mask with feathered edges
        """
        h, w = crop_shape
        
        # Base edge feather mask
        edge_mask = create_edge_feather_mask(h, w, self.feather_pixels)
        
        if inner_mask is not None:
            # Combine with inner content mask
            content_mask = create_differential_mask(inner_mask, self.inner_feather)
            
            # Use maximum - preserve both inner content and edge blending
            stitch_mask = np.maximum(edge_mask, content_mask)
        else:
            stitch_mask = edge_mask
        
        return stitch_mask


# Convenience functions
def quick_blend(bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, feather: int = 32) -> np.ndarray:
    """Quick differential blend with default settings."""
    return apply_differential_blend(bg, fg, mask, feather_pixels=feather)


def create_soft_mask(mask: np.ndarray, feather: int = 32) -> np.ndarray:
    """Create soft mask from binary mask."""
    return create_differential_mask(mask, feather)

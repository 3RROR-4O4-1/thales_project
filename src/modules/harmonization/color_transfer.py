"""
Color Transfer Module

Matches color statistics between the inserted vehicle and surrounding scene
to ensure realistic color harmony.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.ndimage import binary_dilation


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # OpenCV uses BGR, so convert if needed
    if image.shape[-1] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)
    
    return lab.astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB image to RGB."""
    lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
    return rgb


def compute_masked_stats(
    image: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of image within masked region.
    
    Args:
        image: Input image (H, W, C)
        mask: Binary mask (H, W)
        
    Returns:
        Tuple of (mean, std) arrays with shape (C,)
    """
    # Flatten spatial dimensions
    pixels = image[mask > 0.5]
    
    if len(pixels) == 0:
        return np.zeros(image.shape[-1]), np.ones(image.shape[-1])
    
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0) + 1e-6  # Avoid division by zero
    
    return mean, std


def get_context_region(
    mask: np.ndarray,
    dilation_pixels: int = 20
) -> np.ndarray:
    """
    Get the context region around a mask (for sampling background colors).
    
    Args:
        mask: Binary mask of the object
        dilation_pixels: How many pixels to dilate
        
    Returns:
        Binary mask of the context region (dilated - original)
    """
    # Dilate mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1)
    )
    dilated = cv2.dilate(
        (mask > 0.5).astype(np.uint8),
        kernel
    )
    
    # Context is dilated region minus original
    context = dilated.astype(bool) & ~(mask > 0.5)
    
    return context.astype(np.float32)


def lab_color_transfer(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
    preserve_luminance: bool = False
) -> np.ndarray:
    """
    Transfer color statistics from target to source using LAB color space.
    
    This is the classic Reinhard color transfer algorithm.
    
    Args:
        source: Source image to be color-corrected (H, W, C)
        target: Target image to match colors to (H, W, C)
        source_mask: Optional mask for source region
        target_mask: Optional mask for target region
        preserve_luminance: If True, only transfer chrominance (a, b channels)
        
    Returns:
        Color-corrected source image
    """
    # Convert to LAB
    source_lab = rgb_to_lab(source)
    target_lab = rgb_to_lab(target)
    
    # Create default masks if not provided
    if source_mask is None:
        source_mask = np.ones(source.shape[:2], dtype=np.float32)
    if target_mask is None:
        target_mask = np.ones(target.shape[:2], dtype=np.float32)
    
    # Compute statistics
    src_mean, src_std = compute_masked_stats(source_lab, source_mask)
    tgt_mean, tgt_std = compute_masked_stats(target_lab, target_mask)
    
    # Transfer
    result_lab = source_lab.copy()
    
    if preserve_luminance:
        # Only transfer a and b channels
        for i in [1, 2]:
            result_lab[:, :, i] = (source_lab[:, :, i] - src_mean[i]) / src_std[i]
            result_lab[:, :, i] = result_lab[:, :, i] * tgt_std[i] + tgt_mean[i]
    else:
        # Transfer all channels
        for i in range(3):
            result_lab[:, :, i] = (source_lab[:, :, i] - src_mean[i]) / src_std[i]
            result_lab[:, :, i] = result_lab[:, :, i] * tgt_std[i] + tgt_mean[i]
    
    # Clip and convert back
    result_lab = np.clip(result_lab, 0, 255)
    result = lab_to_rgb(result_lab)
    
    return result


def histogram_matching(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Match histogram of source to target.
    
    Args:
        source: Source image
        target: Target image
        source_mask: Optional mask for source
        target_mask: Optional mask for target
        
    Returns:
        Histogram-matched source image
    """
    from skimage.exposure import match_histograms
    
    # If masks provided, extract regions
    if source_mask is not None and target_mask is not None:
        # Match within masked regions only
        src_pixels = source[source_mask > 0.5]
        tgt_pixels = target[target_mask > 0.5]
        
        # This is a simplified version - full implementation would
        # properly handle masked histogram matching
        matched = match_histograms(source, target, channel_axis=-1)
    else:
        matched = match_histograms(source, target, channel_axis=-1)
    
    return matched.astype(source.dtype)


def harmonize_colors(
    generated: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    context_dilation: int = 20,
    method: str = "lab_transfer",
    strength: float = 1.0
) -> np.ndarray:
    """
    Harmonize colors of generated region with surrounding background.
    
    Args:
        generated: Generated/inpainted image (full size)
        background: Original background image
        mask: Mask of generated region
        context_dilation: Pixels to sample around mask for target colors
        method: "lab_transfer" or "histogram_matching"
        strength: Blending strength (0-1), 1 = full color transfer
        
    Returns:
        Color-harmonized image
    """
    # Get context region from background
    context_mask = get_context_region(mask, context_dilation)
    
    # Extract generated region
    gen_region = generated.copy()
    
    if method == "lab_transfer":
        # Transfer colors from context to generated region
        corrected = lab_color_transfer(
            gen_region,
            background,
            source_mask=mask,
            target_mask=context_mask
        )
    elif method == "histogram_matching":
        corrected = histogram_matching(
            gen_region,
            background,
            source_mask=mask,
            target_mask=context_mask
        )
    else:
        corrected = gen_region
    
    # Blend based on strength
    if strength < 1.0:
        corrected = gen_region * (1 - strength) + corrected * strength
    
    # Only apply to masked region
    result = background.copy()
    mask_3ch = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
    result = result * (1 - mask_3ch) + corrected * mask_3ch
    
    return result.astype(background.dtype)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Adjust brightness and contrast of image.
    
    Args:
        image: Input image (0-1 or 0-255)
        brightness: Brightness adjustment (-1 to 1)
        contrast: Contrast multiplier (0.5 to 2.0 typical)
        mask: Optional mask to apply adjustment only to masked region
        
    Returns:
        Adjusted image
    """
    # Normalize to 0-1 if needed
    if image.max() > 1.0:
        img = image.astype(np.float32) / 255.0
        was_uint8 = True
    else:
        img = image.astype(np.float32)
        was_uint8 = False
    
    # Apply adjustments
    adjusted = (img - 0.5) * contrast + 0.5 + brightness
    adjusted = np.clip(adjusted, 0, 1)
    
    # Apply mask if provided
    if mask is not None:
        mask_3ch = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
        adjusted = img * (1 - mask_3ch) + adjusted * mask_3ch
    
    if was_uint8:
        adjusted = (adjusted * 255).astype(np.uint8)
    
    return adjusted


def match_lighting(
    generated: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    context_dilation: int = 30
) -> np.ndarray:
    """
    Match lighting (luminance) of generated region to background context.
    
    This is useful when the generated vehicle has different lighting
    than the surrounding scene.
    
    Args:
        generated: Generated image
        background: Background image
        mask: Mask of generated region
        context_dilation: Pixels to sample around mask
        
    Returns:
        Lighting-matched image
    """
    # Convert to LAB
    gen_lab = rgb_to_lab(generated)
    bg_lab = rgb_to_lab(background)
    
    # Get context region
    context_mask = get_context_region(mask, context_dilation)
    
    # Match only L channel (luminance)
    gen_L = gen_lab[:, :, 0]
    bg_L = bg_lab[:, :, 0]
    
    # Compute stats
    gen_mean = np.mean(gen_L[mask > 0.5])
    gen_std = np.std(gen_L[mask > 0.5]) + 1e-6
    bg_mean = np.mean(bg_L[context_mask > 0.5])
    bg_std = np.std(bg_L[context_mask > 0.5]) + 1e-6
    
    # Transfer luminance
    result_lab = gen_lab.copy()
    result_lab[:, :, 0] = (gen_L - gen_mean) / gen_std * bg_std + bg_mean
    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
    
    # Convert back
    result = lab_to_rgb(result_lab)
    
    # Apply only to masked region
    mask_3ch = mask[:, :, np.newaxis]
    final = background.copy()
    
    if background.dtype == np.float32:
        result = result.astype(np.float32) / 255.0
        
    final = final * (1 - mask_3ch) + result * mask_3ch
    
    return final.astype(background.dtype)


class ColorHarmonizer:
    """
    High-level class for color harmonization operations.
    """
    
    def __init__(
        self,
        method: str = "lab_transfer",
        context_dilation: int = 20,
        strength: float = 0.8,
        match_lighting: bool = True
    ):
        self.method = method
        self.context_dilation = context_dilation
        self.strength = strength
        self.do_match_lighting = match_lighting
    
    def harmonize(
        self,
        generated: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Full color harmonization pipeline.
        """
        result = generated.copy()
        
        # Match lighting first
        if self.do_match_lighting:
            result = match_lighting(result, background, mask, self.context_dilation)
        
        # Then harmonize colors
        result = harmonize_colors(
            result,
            background,
            mask,
            context_dilation=self.context_dilation,
            method=self.method,
            strength=self.strength
        )
        
        return result

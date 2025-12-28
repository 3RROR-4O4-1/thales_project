"""
Shadow Generation Module

Generates realistic contact shadows and cast shadows for inserted vehicles.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter, shift


def create_contact_shadow(
    vehicle_mask: np.ndarray,
    shadow_intensity: float = 0.4,
    blur_sigma: float = 15.0,
    vertical_offset: int = 5,
    scale_y: float = 0.3
) -> np.ndarray:
    """
    Create a simple contact shadow beneath the vehicle.
    
    The contact shadow is created by:
    1. Taking the bottom portion of the vehicle mask
    2. Scaling it vertically (squashing)
    3. Offsetting it downward
    4. Blurring it
    
    Args:
        vehicle_mask: Binary mask of the vehicle (H, W)
        shadow_intensity: Darkness of shadow (0-1)
        blur_sigma: Gaussian blur sigma for shadow softness
        vertical_offset: Pixels to shift shadow down
        scale_y: Vertical scale factor (< 1 squashes shadow)
        
    Returns:
        Shadow mask (H, W) with values 0-shadow_intensity
    """
    h, w = vehicle_mask.shape[:2]
    
    # Find bottom edge of vehicle
    rows = np.any(vehicle_mask > 0.5, axis=1)
    if not rows.any():
        return np.zeros_like(vehicle_mask, dtype=np.float32)
    
    bottom_row = np.where(rows)[0][-1]
    
    # Create shadow base from bottom portion of mask
    shadow_base = vehicle_mask.copy().astype(np.float32)
    
    # Scale vertically (squash shadow)
    if scale_y != 1.0:
        # Calculate new height
        center_y = bottom_row
        coords_y = np.arange(h)
        
        # Scale coordinates around bottom edge
        scaled_coords = center_y + (coords_y - center_y) * scale_y
        scaled_coords = np.clip(scaled_coords, 0, h - 1).astype(int)
        
        # Resample
        shadow_scaled = np.zeros_like(shadow_base)
        for y in range(h):
            src_y = scaled_coords[y]
            if 0 <= src_y < h:
                shadow_scaled[y] = shadow_base[src_y]
        shadow_base = shadow_scaled
    
    # Shift downward
    shadow_shifted = np.zeros_like(shadow_base)
    if vertical_offset > 0:
        shadow_shifted[vertical_offset:] = shadow_base[:-vertical_offset]
    else:
        shadow_shifted = shadow_base
    
    # Remove overlap with vehicle (shadow should be behind)
    shadow_shifted = shadow_shifted * (1 - vehicle_mask.astype(np.float32))
    
    # Blur for soft edges
    shadow_blurred = gaussian_filter(shadow_shifted, sigma=blur_sigma)
    
    # Apply intensity
    shadow_final = shadow_blurred * shadow_intensity
    
    return shadow_final.astype(np.float32)


def create_cast_shadow(
    vehicle_mask: np.ndarray,
    light_direction: Tuple[float, float] = (0.3, -0.8),
    shadow_length: float = 50.0,
    shadow_intensity: float = 0.3,
    blur_sigma: float = 20.0,
    fade_with_distance: bool = True
) -> np.ndarray:
    """
    Create a cast shadow based on light direction.
    
    Args:
        vehicle_mask: Binary mask of the vehicle
        light_direction: (x, y) normalized direction light is coming FROM
                        Positive y means light from above
        shadow_length: Maximum shadow length in pixels
        shadow_intensity: Darkness of shadow
        blur_sigma: Blur sigma for softness
        fade_with_distance: If True, shadow fades as it gets further from object
        
    Returns:
        Cast shadow mask
    """
    h, w = vehicle_mask.shape[:2]
    
    # Normalize light direction
    lx, ly = light_direction
    length = np.sqrt(lx**2 + ly**2) + 1e-8
    lx, ly = lx / length, ly / length
    
    # Shadow direction is opposite to light direction
    shadow_dx = -lx * shadow_length
    shadow_dy = -ly * shadow_length
    
    # Create shadow by shifting mask
    shadow = np.zeros_like(vehicle_mask, dtype=np.float32)
    
    # Multiple shifts to create elongated shadow
    num_steps = max(int(shadow_length / 2), 10)
    
    for i in range(num_steps):
        t = i / num_steps
        offset_x = int(shadow_dx * t)
        offset_y = int(shadow_dy * t)
        
        # Shift mask
        shifted = shift(
            vehicle_mask.astype(np.float32),
            [offset_y, offset_x],
            mode='constant',
            cval=0
        )
        
        # Fade with distance if requested
        if fade_with_distance:
            weight = 1.0 - t * 0.7  # Fade to 30% at full distance
        else:
            weight = 1.0
        
        shadow = np.maximum(shadow, shifted * weight)
    
    # Remove overlap with vehicle
    shadow = shadow * (1 - vehicle_mask.astype(np.float32))
    
    # Blur
    shadow = gaussian_filter(shadow, sigma=blur_sigma)
    
    # Apply intensity
    shadow = shadow * shadow_intensity
    
    return shadow.astype(np.float32)


def create_ambient_occlusion_shadow(
    vehicle_mask: np.ndarray,
    ao_map: Optional[np.ndarray] = None,
    intensity: float = 0.3,
    blur_sigma: float = 10.0
) -> np.ndarray:
    """
    Create shadow from ambient occlusion map.
    
    If no AO map provided, generates a simple approximation based on
    the vehicle mask edge.
    
    Args:
        vehicle_mask: Binary mask of the vehicle
        ao_map: Pre-rendered ambient occlusion map (optional)
        intensity: Shadow intensity
        blur_sigma: Blur sigma
        
    Returns:
        AO shadow mask
    """
    if ao_map is not None:
        # Use provided AO map
        shadow = ao_map.astype(np.float32)
        if shadow.max() > 1.0:
            shadow = shadow / 255.0
        shadow = shadow * intensity
    else:
        # Generate simple AO approximation from mask edges
        # Inner shadow along mask boundary
        from scipy.ndimage import distance_transform_edt
        
        # Distance from outside the mask
        dist_outside = distance_transform_edt(vehicle_mask > 0.5)
        
        # Create shadow that's dark near edges, fades toward center
        max_dist = min(20, dist_outside.max())
        shadow = np.clip((max_dist - dist_outside) / max_dist, 0, 1)
        
        # Only keep shadow at bottom (ground contact)
        rows = np.any(vehicle_mask > 0.5, axis=1)
        if rows.any():
            bottom = np.where(rows)[0][-1]
            # Fade shadow based on distance from bottom
            y_coords = np.arange(vehicle_mask.shape[0])
            dist_from_bottom = np.abs(y_coords - bottom)[:, np.newaxis]
            bottom_weight = np.exp(-dist_from_bottom / 30.0)
            shadow = shadow * bottom_weight
        
        shadow = gaussian_filter(shadow, sigma=blur_sigma)
        shadow = shadow * intensity
    
    return shadow.astype(np.float32)


def apply_shadow_to_image(
    image: np.ndarray,
    shadow_mask: np.ndarray,
    shadow_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    blend_mode: str = "multiply"
) -> np.ndarray:
    """
    Apply shadow mask to image.
    
    Args:
        image: Input image (H, W, C)
        shadow_mask: Shadow intensity mask (H, W), 0-1
        shadow_color: RGB color of shadow (usually dark)
        blend_mode: "multiply" or "overlay"
        
    Returns:
        Image with shadow applied
    """
    # Ensure float
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
        was_uint8 = True
    else:
        img = image.astype(np.float32)
        was_uint8 = False
    
    # Expand shadow mask to 3 channels
    shadow_3ch = shadow_mask[:, :, np.newaxis]
    
    if blend_mode == "multiply":
        # Multiply blend: image * (1 - shadow * (1 - shadow_color))
        shadow_layer = 1 - shadow_3ch * (1 - np.array(shadow_color))
        result = img * shadow_layer
    else:  # overlay
        # Simple overlay
        shadow_rgb = np.array(shadow_color)[np.newaxis, np.newaxis, :]
        result = img * (1 - shadow_3ch) + shadow_rgb * shadow_3ch
    
    result = np.clip(result, 0, 1)
    
    if was_uint8:
        result = (result * 255).astype(np.uint8)
    
    return result


def estimate_light_direction(
    image: np.ndarray,
    depth_map: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Estimate dominant light direction from image.
    
    Simple heuristic based on image brightness gradients.
    
    Args:
        image: Input image
        depth_map: Optional depth map for better estimation
        
    Returns:
        (x, y) light direction vector
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image if image.dtype == np.uint8 else (image * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    
    # Average gradient direction (weighted by magnitude)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    weights = magnitude / (magnitude.sum() + 1e-8)
    
    avg_gx = np.sum(grad_x * weights)
    avg_gy = np.sum(grad_y * weights)
    
    # Normalize
    length = np.sqrt(avg_gx**2 + avg_gy**2) + 1e-8
    light_dir = (avg_gx / length, avg_gy / length)
    
    return light_dir


class ShadowGenerator:
    """
    High-level class for shadow generation.
    """
    
    def __init__(
        self,
        contact_shadow: bool = True,
        cast_shadow: bool = True,
        ao_shadow: bool = False,
        contact_intensity: float = 0.4,
        cast_intensity: float = 0.3,
        ao_intensity: float = 0.2,
        blur_sigma: float = 15.0
    ):
        self.contact_shadow = contact_shadow
        self.cast_shadow = cast_shadow
        self.ao_shadow = ao_shadow
        self.contact_intensity = contact_intensity
        self.cast_intensity = cast_intensity
        self.ao_intensity = ao_intensity
        self.blur_sigma = blur_sigma
    
    def generate(
        self,
        vehicle_mask: np.ndarray,
        light_direction: Optional[Tuple[float, float]] = None,
        ao_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate combined shadow mask.
        """
        combined_shadow = np.zeros_like(vehicle_mask, dtype=np.float32)
        
        if self.contact_shadow:
            contact = create_contact_shadow(
                vehicle_mask,
                shadow_intensity=self.contact_intensity,
                blur_sigma=self.blur_sigma
            )
            combined_shadow = np.maximum(combined_shadow, contact)
        
        if self.cast_shadow and light_direction is not None:
            cast = create_cast_shadow(
                vehicle_mask,
                light_direction=light_direction,
                shadow_intensity=self.cast_intensity,
                blur_sigma=self.blur_sigma * 1.5
            )
            combined_shadow = np.maximum(combined_shadow, cast)
        
        if self.ao_shadow:
            ao = create_ambient_occlusion_shadow(
                vehicle_mask,
                ao_map=ao_map,
                intensity=self.ao_intensity,
                blur_sigma=self.blur_sigma * 0.5
            )
            combined_shadow = np.maximum(combined_shadow, ao)
        
        return combined_shadow
    
    def apply(
        self,
        image: np.ndarray,
        vehicle_mask: np.ndarray,
        light_direction: Optional[Tuple[float, float]] = None,
        ao_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate and apply shadow to image.
        """
        shadow_mask = self.generate(vehicle_mask, light_direction, ao_map)
        return apply_shadow_to_image(image, shadow_mask)

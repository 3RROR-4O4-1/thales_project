"""
Inpainting Module

Scale-aware inpainting for small object insertion with ComfyUI integration.
"""

from .scale_aware import (
    ScaleAwareInpainter,
    ScaleAwareConfig,
    CropRegion,
    scale_aware_inpaint,
    compute_optimal_processing_size
)

from .comfyui_api import (
    ComfyUIClient,
    ComfyUIConfig,
    FluxInpaintWorkflow,
    create_inpaint_function
)

__all__ = [
    # Scale-aware inpainting
    'ScaleAwareInpainter',
    'ScaleAwareConfig',
    'CropRegion',
    'scale_aware_inpaint',
    'compute_optimal_processing_size',
    
    # ComfyUI integration
    'ComfyUIClient',
    'ComfyUIConfig',
    'FluxInpaintWorkflow',
    'create_inpaint_function'
]

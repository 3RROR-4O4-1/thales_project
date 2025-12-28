"""
3D Rendering Module

Blender-based multi-view rendering with depth, normal, and mask export.
"""

from .blender_multiview import (
    MultiViewRenderer,
    setup_simple_scene
)

__all__ = [
    'MultiViewRenderer',
    'setup_simple_scene'
]

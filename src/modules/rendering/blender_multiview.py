#!/usr/bin/env python3
"""
Multi-View Renderer for Blender 5.0+ (Auto-Camera Edition)

Automatically calculates optimal camera distance based on object size.
Optionally normalizes model scale to real-world dimensions.

Renders:
- RGB renders (with transparency)
- Alpha masks

Usage:
    blender --background scene.blend --python blender_multiview_auto.py -- \
        --output ./renders \
        --target Tank \
        --resolution 1024 \
        --normalize-scale 10.0   # Scale to 10 meters length
"""

import bpy
import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import mathutils


class MultiViewRenderer:
    """Render object from multiple viewpoints with auto camera setup."""
    
    def __init__(
        self,
        output_dir: str,
        resolution: Tuple[int, int] = (1024, 1024),
        samples: int = 128
    ):
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.samples = samples
        
        # Create output directories
        for subdir in ['rgb', 'mask']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self._setup_render_settings()
    
    def _setup_render_settings(self):
        """Configure render settings."""
        scene = bpy.context.scene
        
        # Use Cycles for quality
        scene.render.engine = 'CYCLES'
        
        # Try to use GPU
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            
            # Try CUDA first, then HIP, then OPTIX
            for compute_type in ['CUDA', 'OPTIX', 'HIP', 'METAL']:
                try:
                    prefs.compute_device_type = compute_type
                    prefs.get_devices()
                    
                    # Enable all available devices
                    devices_found = False
                    for device in prefs.devices:
                        if device.type != 'CPU':
                            device.use = True
                            devices_found = True
                    
                    if devices_found:
                        scene.cycles.device = 'GPU'
                        print(f"Using GPU rendering ({compute_type})")
                        break
                except:
                    continue
            else:
                scene.cycles.device = 'CPU'
                print("Using CPU rendering")
        except Exception as e:
            scene.cycles.device = 'CPU'
            print(f"Using CPU rendering (error: {e})")
        
        # Quality settings
        scene.cycles.samples = self.samples
        scene.cycles.use_denoising = True
        
        # Resolution
        scene.render.resolution_x = self.resolution[0]
        scene.render.resolution_y = self.resolution[1]
        scene.render.resolution_percentage = 100
        
        # Transparent background
        scene.render.film_transparent = True
        
        # Output format
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '16'
    
    def get_object_bounds(self, obj: bpy.types.Object) -> Dict:
        """
        Get object bounding box info in world space.
        
        Returns:
            Dict with center, dimensions, radius, min/max coordinates
        """
        # Get world-space bounding box corners
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        
        # Calculate bounds
        xs = [v.x for v in bbox_corners]
        ys = [v.y for v in bbox_corners]
        zs = [v.z for v in bbox_corners]
        
        min_coord = mathutils.Vector((min(xs), min(ys), min(zs)))
        max_coord = mathutils.Vector((max(xs), max(ys), max(zs)))
        
        center = (min_coord + max_coord) / 2
        dimensions = max_coord - min_coord
        
        # Bounding sphere radius
        radius = max(dimensions) / 2
        
        return {
            'center': center,
            'dimensions': dimensions,
            'radius': radius,
            'min': min_coord,
            'max': max_coord,
            'width': dimensions.x,   # X
            'depth': dimensions.y,   # Y  
            'height': dimensions.z   # Z
        }
    
    def normalize_object_scale(
        self,
        obj: bpy.types.Object,
        target_length: float = 10.0
    ) -> float:
        """
        Scale object so its longest dimension equals target_length.
        
        Args:
            obj: Object to scale
            target_length: Desired length in meters
            
        Returns:
            Scale factor applied
        """
        bounds = self.get_object_bounds(obj)
        current_max = max(bounds['dimensions'])
        
        if current_max == 0:
            print("Warning: Object has zero size!")
            return 1.0
        
        scale_factor = target_length / current_max
        
        print(f"Normalizing scale:")
        print(f"  Current size: {bounds['width']:.2f} x {bounds['depth']:.2f} x {bounds['height']:.2f}")
        print(f"  Scale factor: {scale_factor:.4f}")
        
        # Apply scale
        obj.scale = (scale_factor, scale_factor, scale_factor)
        
        # Apply the scale transform
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        
        # Reposition on ground (Z=0)
        bpy.context.view_layer.update()
        new_bounds = self.get_object_bounds(obj)
        obj.location.z -= new_bounds['min'].z
        
        # Center XY
        obj.location.x -= new_bounds['center'].x
        obj.location.y -= new_bounds['center'].y
        
        bpy.context.view_layer.update()
        final_bounds = self.get_object_bounds(obj)
        print(f"  New size: {final_bounds['width']:.2f} x {final_bounds['depth']:.2f} x {final_bounds['height']:.2f}")
        print(f"  Position: {obj.location}")
        
        return scale_factor
    
    def calculate_camera_distance(
        self,
        obj: bpy.types.Object,
        camera: bpy.types.Object,
        fill_factor: float = 0.7,
        elevation: float = 30.0
    ) -> float:
        """
        Calculate camera distance to frame object properly.
        
        Args:
            obj: Target object
            camera: Camera object
            fill_factor: How much of frame object should fill (0.5-0.9)
            elevation: Camera elevation angle in degrees
            
        Returns:
            Optimal camera distance
        """
        bounds = self.get_object_bounds(obj)
        
        # Get camera FOV
        cam_data = camera.data
        
        # Calculate FOV in radians
        if cam_data.type == 'PERSP':
            # Get horizontal FOV based on sensor
            if cam_data.sensor_fit == 'HORIZONTAL':
                fov = cam_data.angle
            else:
                # Calculate from vertical FOV and aspect ratio
                aspect = self.resolution[0] / self.resolution[1]
                fov = 2 * math.atan(aspect * math.tan(cam_data.angle / 2))
        else:
            # Orthographic - use a default
            fov = math.radians(50)
        
        # Object's bounding sphere radius
        radius = bounds['radius']
        
        # Account for elevation (object appears smaller when viewed from above)
        el_rad = math.radians(elevation)
        effective_radius = radius * max(math.cos(el_rad), 0.5)
        
        # Distance to fit object in frame
        # d = r / sin(fov/2) * (1/fill_factor)
        distance = (effective_radius / math.sin(fov / 2)) * (1.0 / fill_factor)
        
        # Add some padding
        distance *= 1.2
        
        # Minimum distance to avoid clipping
        min_distance = radius * 2
        distance = max(distance, min_distance)
        
        return distance
    
    def _position_camera(
        self,
        camera: bpy.types.Object,
        target_center: mathutils.Vector,
        azimuth: float,
        elevation: float,
        distance: float
    ):
        """
        Position camera to look at target from spherical coordinates.
        
        Args:
            camera: Camera object
            target_center: Point to look at
            azimuth: Horizontal angle in degrees (0 = +Y axis, 90 = +X axis)
            elevation: Vertical angle in degrees (0 = level, 90 = top-down)
            distance: Distance from target center
        """
        # Convert to radians
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        
        # Spherical to Cartesian
        # Azimuth 0 = looking from +Y direction (front of vehicle)
        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = -distance * math.cos(el_rad) * math.cos(az_rad)
        z = distance * math.sin(el_rad)
        
        # Position camera
        camera.location = (
            target_center.x + x,
            target_center.y + y,
            target_center.z + z
        )
        
        # Point at target
        direction = target_center - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
    
    def _extract_mask_from_render(self, view_name: str) -> bool:
        """Extract alpha channel as mask from the rendered image."""
        rgb_path = self.output_dir / 'rgb' / f'{view_name}.png'
        mask_path = self.output_dir / 'mask' / f'{view_name}.png'
        
        try:
            # Load with Blender
            img = bpy.data.images.load(str(rgb_path))
            width, height = img.size
            
            # Get pixels (flat RGBA array)
            pixels = list(img.pixels)
            
            # Extract alpha and create grayscale
            mask_pixels = []
            for i in range(0, len(pixels), 4):
                alpha = pixels[i + 3]
                mask_pixels.extend([alpha, alpha, alpha, 1.0])
            
            # Create new image for mask
            mask_img = bpy.data.images.new(f"mask_{view_name}", width, height)
            mask_img.pixels = mask_pixels
            mask_img.filepath_raw = str(mask_path)
            mask_img.file_format = 'PNG'
            mask_img.save()
            
            # Cleanup
            bpy.data.images.remove(img)
            bpy.data.images.remove(mask_img)
            
            return True
        except Exception as e:
            print(f"  Warning: Could not extract mask: {e}")
            return False
    
    def render_views(
        self,
        target_object: str,
        camera_name: str = "Camera",
        azimuth_list: List[float] = None,
        elevation_list: List[float] = None,
        distance_override: float = None,
        fill_factor: float = 0.7,
        normalize_scale: float = None
    ) -> List[Dict]:
        """
        Render from multiple viewpoints with automatic camera setup.
        
        Args:
            target_object: Name of object to render
            camera_name: Name of camera to use
            azimuth_list: List of azimuth angles (degrees), default 0-330 step 30
            elevation_list: List of elevation angles (degrees), default [15, 30, 45]
            distance_override: Manual camera distance (None = auto-calculate)
            fill_factor: How much of frame object should fill (0.5-0.9)
            normalize_scale: If set, scale object to this length (meters)
            
        Returns:
            List of render metadata dicts
        """
        # Defaults
        if azimuth_list is None:
            azimuth_list = list(range(0, 360, 30))
        if elevation_list is None:
            elevation_list = [15, 30, 45]
        
        # Get target object
        target = bpy.data.objects.get(target_object)
        if target is None:
            available = [o.name for o in bpy.data.objects if o.type == 'MESH']
            raise KeyError(f"Target '{target_object}' not found. Available meshes: {available}")
        
        # Get or create camera
        camera = bpy.data.objects.get(camera_name)
        if camera is None or camera.type != 'CAMERA':
            bpy.ops.object.camera_add()
            camera = bpy.context.active_object
            camera.name = camera_name
            camera.data.lens = 50  # 50mm lens
        
        bpy.context.scene.camera = camera
        
        # Normalize scale if requested
        if normalize_scale is not None:
            self.normalize_object_scale(target, normalize_scale)
        
        # Get object bounds
        bounds = self.get_object_bounds(target)
        target_center = bounds['center'].copy()
        # Look at center of object, slightly above ground
        target_center.z = bounds['height'] * 0.4
        
        print(f"\nObject info:")
        print(f"  Dimensions: {bounds['width']:.2f} x {bounds['depth']:.2f} x {bounds['height']:.2f}")
        print(f"  Center: {bounds['center']}")
        print(f"  Look-at point: {target_center}")
        
        # Calculate camera distances for each elevation
        if distance_override:
            distances = {el: distance_override for el in elevation_list}
            print(f"\nUsing fixed camera distance: {distance_override:.2f}")
        else:
            distances = {}
            print(f"\nAuto-calculated camera distances (fill={fill_factor}):")
            for el in elevation_list:
                dist = self.calculate_camera_distance(target, camera, fill_factor, el)
                distances[el] = dist
                print(f"  Elevation {el}°: {dist:.2f}m")
        
        # Total renders
        total = len(azimuth_list) * len(elevation_list)
        print(f"\nRendering {total} views...")
        print(f"  Azimuths: {azimuth_list}")
        print(f"  Elevations: {elevation_list}")
        print()
        
        results = []
        count = 0
        
        for elevation in elevation_list:
            distance = distances[elevation]
            
            for azimuth in azimuth_list:
                count += 1
                view_name = f"view_az{int(azimuth):03d}_el{int(elevation):02d}"
                
                print(f"[{count}/{total}] {view_name} (d={distance:.1f}m)")
                
                # Position camera
                self._position_camera(camera, target_center, azimuth, elevation, distance)
                bpy.context.view_layer.update()
                
                # Set output path
                rgb_path = self.output_dir / 'rgb' / f'{view_name}.png'
                bpy.context.scene.render.filepath = str(rgb_path)
                
                # Render
                bpy.ops.render.render(write_still=True)
                
                # Extract mask
                self._extract_mask_from_render(view_name)
                
                # Store metadata
                results.append({
                    'view_name': view_name,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'distance': distance,
                    'camera_location': list(camera.location),
                    'camera_rotation': [math.degrees(r) for r in camera.rotation_euler],
                    'rgb_path': f'rgb/{view_name}.png',
                    'mask_path': f'mask/{view_name}.png'
                })
        
        # Save metadata
        metadata = {
            'target_object': target_object,
            'resolution': list(self.resolution),
            'samples': self.samples,
            'object_dimensions': {
                'width': bounds['width'],
                'depth': bounds['depth'],
                'height': bounds['height']
            },
            'fill_factor': fill_factor,
            'views': results
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Done! Rendered {len(results)} views")
        print(f"Output: {self.output_dir}")
        print(f"Metadata: {metadata_path}")
        print(f"{'='*60}")
        
        return results


def main():
    """Command line entry point."""
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(
        description='Multi-view renderer with auto camera setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic render with auto camera
  blender --background scene.blend --python %(prog)s -- --target Tank
  
  # Normalize scale to 10m and render
  blender --background scene.blend --python %(prog)s -- --target Tank --normalize-scale 10
  
  # Custom angles and fixed distance
  blender --background scene.blend --python %(prog)s -- --target Tank --distance 15 --elevations 20 40 60
        """
    )
    
    parser.add_argument('--output', '-o', default='./renders', 
                        help='Output directory (default: ./renders)')
    parser.add_argument('--target', '-t', default='Tank', 
                        help='Target object name (default: Tank)')
    parser.add_argument('--camera', '-c', default='Camera', 
                        help='Camera name (default: Camera)')
    parser.add_argument('--resolution', '-r', type=int, default=1024, 
                        help='Render resolution (default: 1024)')
    parser.add_argument('--samples', '-s', type=int, default=128, 
                        help='Render samples (default: 128)')
    parser.add_argument('--azimuth-step', type=int, default=30, 
                        help='Azimuth step in degrees (default: 30)')
    parser.add_argument('--elevations', nargs='+', type=float, default=[15, 30, 45], 
                        help='Elevation angles (default: 15 30 45)')
    parser.add_argument('--distance', '-d', type=float, default=None, 
                        help='Fixed camera distance (default: auto-calculate)')
    parser.add_argument('--fill', '-f', type=float, default=0.7, 
                        help='Frame fill factor 0.5-0.9 (default: 0.7)')
    parser.add_argument('--normalize-scale', type=float, default=None,
                        help='Normalize object to this length in meters (e.g., 10 for a tank)')
    parser.add_argument('--save-blend', action='store_true',
                        help='Save the modified blend file')
    
    args = parser.parse_args(argv)
    
    # Create renderer
    renderer = MultiViewRenderer(
        output_dir=args.output,
        resolution=(args.resolution, args.resolution),
        samples=args.samples
    )
    
    # Generate azimuth list
    azimuth_list = list(range(0, 360, args.azimuth_step))
    
    # Render
    renderer.render_views(
        target_object=args.target,
        camera_name=args.camera,
        azimuth_list=azimuth_list,
        elevation_list=args.elevations,
        distance_override=args.distance,
        fill_factor=args.fill,
        normalize_scale=args.normalize_scale
    )
    
    # Save blend file if requested
    if args.save_blend:
        bpy.ops.wm.save_mainfile()
        print(f"Saved blend file")


if __name__ == '__main__':
    main()
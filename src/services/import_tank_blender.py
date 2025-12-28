#!/usr/bin/env python3
"""
Blender Tank Import Script - Auto-Fix Edition

Automatically handles:
- Broken Windows paths in MTL files
- DDS to PNG conversion
- Texture assignment by filename matching
- Multi-material objects

Usage:
    blender --background --python import_tank_auto.py -- \
        --asset-dir /path/to/tank \
        --output /path/to/output.blend \
        --setup-scene
"""

import bpy
import os
import sys
import re
import shutil
from pathlib import Path


def find_obj_file(asset_dir: Path) -> Path:
    """Find the OBJ file in the asset directory."""
    for search_dir in [asset_dir / "source", asset_dir]:
        if search_dir.exists():
            obj_files = list(search_dir.glob("*.obj"))
            if obj_files:
                return obj_files[0]
    
    obj_files = list(asset_dir.rglob("*.obj"))
    if obj_files:
        return obj_files[0]
    
    raise FileNotFoundError(f"No OBJ file found in {asset_dir}")


def find_mtl_file(obj_path: Path) -> Path:
    """Find MTL file referenced by OBJ or in same directory."""
    # Check OBJ for mtllib reference
    with open(obj_path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('mtllib'):
                mtl_name = line.split()[1].strip()
                mtl_path = obj_path.parent / mtl_name
                if mtl_path.exists():
                    return mtl_path
    
    # Fallback: find any MTL in same directory
    mtl_files = list(obj_path.parent.glob("*.mtl"))
    if mtl_files:
        return mtl_files[0]
    
    return None


def extract_texture_names_from_mtl(mtl_path: Path) -> list:
    """
    Extract texture filenames from MTL file, ignoring broken paths.
    
    Handles:
    - map_Kd (diffuse)
    - map_Ka (ambient)
    - map_Ks (specular)
    - map_Ns (shininess)
    - map_bump / bump
    - map_d (alpha)
    """
    texture_names = []
    
    if not mtl_path or not mtl_path.exists():
        return texture_names
    
    # Patterns for texture maps
    map_patterns = ['map_Kd', 'map_Ka', 'map_Ks', 'map_Ns', 'map_bump', 'bump', 'map_d', 'map_Ke']
    
    with open(mtl_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            for pattern in map_patterns:
                if line.startswith(pattern):
                    # Extract path after the keyword
                    parts = line.split(maxsplit=1)
                    if len(parts) > 1:
                        tex_path = parts[1].strip()
                        # Get just the filename (handles both Windows and Unix paths)
                        filename = Path(tex_path.replace('\\', '/')).name
                        texture_names.append(filename)
    
    return list(set(texture_names))  # Remove duplicates


def find_local_textures(asset_dir: Path, texture_names: list) -> dict:
    """
    Find local texture files matching the names from MTL.
    
    Returns dict mapping original name -> local path
    """
    found = {}
    
    # Collect all image files in asset directory
    search_dirs = [
        asset_dir / "source",
        asset_dir / "textures",
        asset_dir
    ]
    
    local_images = {}
    for search_dir in search_dirs:
        if search_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tga', '*.dds', '*.PNG', '*.DDS']:
                for img_path in search_dir.glob(ext):
                    # Store by stem (without extension) for flexible matching
                    local_images[img_path.stem.lower()] = img_path
                    local_images[img_path.name.lower()] = img_path
    
    # Match MTL texture names to local files
    for tex_name in texture_names:
        tex_stem = Path(tex_name).stem.lower()
        tex_name_lower = tex_name.lower()
        
        # Try exact match first
        if tex_name_lower in local_images:
            found[tex_name] = local_images[tex_name_lower]
        # Try stem match (different extension)
        elif tex_stem in local_images:
            found[tex_name] = local_images[tex_stem]
        # Try partial match
        else:
            for local_name, local_path in local_images.items():
                if tex_stem in local_name or local_name in tex_stem:
                    found[tex_name] = local_path
                    break
    
    return found


def convert_dds_if_needed(texture_path: Path) -> Path:
    """Convert DDS to PNG if needed. Returns path to usable image."""
    if texture_path.suffix.lower() != '.dds':
        return texture_path
    
    png_path = texture_path.with_suffix('.png')
    
    # Check if PNG already exists
    if png_path.exists():
        return png_path
    
    # Try conversion with Pillow
    try:
        from PIL import Image
        img = Image.open(texture_path)
        img.save(png_path, 'PNG')
        print(f"  Converted: {texture_path.name} -> {png_path.name}")
        return png_path
    except ImportError:
        print(f"  Warning: Pillow not installed, cannot convert DDS")
        print(f"  Install with: pip install Pillow")
    except Exception as e:
        print(f"  Warning: Failed to convert {texture_path.name}: {e}")
    
    # Try with ImageMagick as fallback
    try:
        import subprocess
        result = subprocess.run(
            ['convert', str(texture_path), str(png_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Converted (ImageMagick): {texture_path.name} -> {png_path.name}")
            return png_path
    except FileNotFoundError:
        pass  # ImageMagick not installed
    except Exception as e:
        print(f"  ImageMagick conversion failed: {e}")
    
    return texture_path  # Return original if conversion fails


def categorize_textures(texture_paths: list) -> dict:
    """
    Categorize textures by type based on filename patterns.
    
    Returns dict with keys: diffuse, normal, roughness, metallic, ao, specular
    """
    categories = {
        'diffuse': None,
        'normal': None,
        'roughness': None,
        'metallic': None,
        'ao': None,
        'specular': None,
        'emissive': None
    }
    
    patterns = {
        'diffuse': ['diffuse', 'albedo', 'base_color', 'basecolor', 'color', 'diff', '_d'],
        'normal': ['normal', 'norm', 'nrm', '_n', 'bump'],
        'roughness': ['roughness', 'rough', '_r', 'gloss'],
        'metallic': ['metallic', 'metal', '_m', 'metalness'],
        'ao': ['ao', 'ambient', 'occlusion'],
        'specular': ['specular', 'spec', '_s'],
        'emissive': ['emissive', 'emission', 'emit', 'glow']
    }
    
    uncategorized = []
    
    for tex_path in texture_paths:
        name_lower = tex_path.stem.lower()
        matched = False
        
        for category, keywords in patterns.items():
            if categories[category] is None:
                for keyword in keywords:
                    if keyword in name_lower:
                        categories[category] = tex_path
                        matched = True
                        break
            if matched:
                break
        
        if not matched:
            uncategorized.append(tex_path)
    
    # If no diffuse found, use the largest uncategorized texture
    if categories['diffuse'] is None and uncategorized:
        categories['diffuse'] = max(uncategorized, key=lambda p: p.stat().st_size)
        uncategorized.remove(categories['diffuse'])
    
    # If still no diffuse, use largest from all textures
    if categories['diffuse'] is None and texture_paths:
        categories['diffuse'] = max(texture_paths, key=lambda p: p.stat().st_size)
    
    return categories


def create_pbr_material(name: str, textures: dict) -> bpy.types.Material:
    """Create a PBR material with the given textures."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create output and BSDF
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    x_pos = -400
    y_pos = 400
    
    # Diffuse/Albedo
    if textures.get('diffuse'):
        tex_path = convert_dds_if_needed(textures['diffuse'])
        try:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (x_pos, y_pos)
            tex_node.image = bpy.data.images.load(str(tex_path))
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            # Also use alpha if present
            if tex_node.image.channels == 4:
                links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
                mat.blend_method = 'HASHED'
            y_pos -= 300
            print(f"    Diffuse: {tex_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load diffuse: {e}")
    
    # Normal map
    if textures.get('normal'):
        tex_path = convert_dds_if_needed(textures['normal'])
        try:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (x_pos, y_pos)
            tex_node.image = bpy.data.images.load(str(tex_path))
            tex_node.image.colorspace_settings.name = 'Non-Color'
            
            normal_map = nodes.new('ShaderNodeNormalMap')
            normal_map.location = (x_pos + 300, y_pos)
            
            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
            y_pos -= 300
            print(f"    Normal: {tex_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load normal: {e}")
    
    # Roughness
    if textures.get('roughness'):
        tex_path = convert_dds_if_needed(textures['roughness'])
        try:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (x_pos, y_pos)
            tex_node.image = bpy.data.images.load(str(tex_path))
            tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(tex_node.outputs['Color'], bsdf.inputs['Roughness'])
            y_pos -= 300
            print(f"    Roughness: {tex_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load roughness: {e}")
    
    # Metallic
    if textures.get('metallic'):
        tex_path = convert_dds_if_needed(textures['metallic'])
        try:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (x_pos, y_pos)
            tex_node.image = bpy.data.images.load(str(tex_path))
            tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(tex_node.outputs['Color'], bsdf.inputs['Metallic'])
            y_pos -= 300
            print(f"    Metallic: {tex_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load metallic: {e}")
    
    # Specular -> use as roughness inverse or specular tint
    if textures.get('specular') and not textures.get('roughness'):
        tex_path = convert_dds_if_needed(textures['specular'])
        try:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (x_pos, y_pos)
            tex_node.image = bpy.data.images.load(str(tex_path))
            tex_node.image.colorspace_settings.name = 'Non-Color'
            
            # Invert for roughness (specular = 1 - roughness approximately)
            invert = nodes.new('ShaderNodeInvert')
            invert.location = (x_pos + 200, y_pos)
            links.new(tex_node.outputs['Color'], invert.inputs['Color'])
            links.new(invert.outputs['Color'], bsdf.inputs['Roughness'])
            print(f"    Specular (as roughness): {tex_path.name}")
        except Exception as e:
            print(f"    Warning: Could not load specular: {e}")
    
    return mat


def import_tank_auto(asset_dir: str, center: bool = True, scale: float = 1.0):
    """
    Import tank model with automatic texture fixing.
    """
    asset_path = Path(asset_dir)
    
    print(f"\n{'='*60}")
    print(f"Importing tank from: {asset_path}")
    print(f"{'='*60}\n")
    
    # Step 1: Find OBJ file
    obj_path = find_obj_file(asset_path)
    print(f"[1/6] Found OBJ: {obj_path.name}")
    
    # Step 2: Find and parse MTL file
    mtl_path = find_mtl_file(obj_path)
    if mtl_path:
        print(f"[2/6] Found MTL: {mtl_path.name}")
        texture_names = extract_texture_names_from_mtl(mtl_path)
        print(f"      Referenced textures: {texture_names}")
    else:
        print(f"[2/6] No MTL file found")
        texture_names = []
    
    # Step 3: Find local texture files
    print(f"[3/6] Searching for local textures...")
    
    # First try matching MTL references
    found_textures = find_local_textures(asset_path, texture_names)
    
    # Also collect ALL local images as fallback
    all_local_images = []
    for search_dir in [asset_path / "source", asset_path / "textures", asset_path]:
        if search_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tga', '*.dds']:
                all_local_images.extend(search_dir.glob(ext))
    
    all_local_images = list(set(all_local_images))  # Remove duplicates
    
    if found_textures:
        print(f"      Matched from MTL:")
        for mtl_name, local_path in found_textures.items():
            print(f"        {mtl_name} -> {local_path.name}")
        texture_paths = list(found_textures.values())
    else:
        print(f"      No MTL matches, using all local images")
        texture_paths = all_local_images
    
    print(f"      Total textures found: {len(texture_paths)}")
    for tp in texture_paths:
        print(f"        - {tp.name}")
    
    # Step 4: Categorize textures
    print(f"[4/6] Categorizing textures...")
    categorized = categorize_textures(texture_paths)
    for cat, path in categorized.items():
        if path:
            print(f"      {cat}: {path.name}")
    
    # Step 5: Import OBJ (ignoring broken MTL)
    print(f"[5/6] Importing OBJ...")
    
    # Import OBJ
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    
    # Get imported objects
    imported = bpy.context.selected_objects
    if not imported:
        raise RuntimeError("No objects imported from OBJ")
    
    print(f"      Imported {len(imported)} object(s)")
    
    # Join if multiple objects
    if len(imported) > 1:
        bpy.context.view_layer.objects.active = imported[0]
        bpy.ops.object.join()
    
    tank = bpy.context.active_object
    tank.name = "Tank"
    
    # Step 6: Create and assign material
    print(f"[6/6] Creating material with textures...")
    
    # Remove existing materials
    tank.data.materials.clear()
    
    # Create new PBR material
    mat = create_pbr_material("TankMaterial", categorized)
    tank.data.materials.append(mat)
    
    # Center and position
    if center:
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
        tank.location = (0, 0, 0)
        
        # Put bottom at Z=0
        import mathutils
        bpy.context.view_layer.update()
        bbox = [tank.matrix_world @ mathutils.Vector(corner) for corner in tank.bound_box]
        min_z = min(v.z for v in bbox)
        tank.location.z -= min_z
    
    # Scale
    if scale != 1.0:
        tank.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(scale=True)
    
    print(f"\n{'='*60}")
    print(f"Tank imported successfully!")
    print(f"  Object: {tank.name}")
    print(f"  Vertices: {len(tank.data.vertices)}")
    print(f"  Materials: {len(tank.data.materials)}")
    print(f"{'='*60}\n")
    
    return tank


def setup_render_scene(tank_obj):
    """Setup scene for rendering."""
    
    # World/Environment
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputWorld')
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (0.7, 0.75, 0.8, 1.0)
    background.inputs['Strength'].default_value = 0.5
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
    # Sun light
    bpy.ops.object.light_add(type='SUN', location=(10, 10, 15))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.2, 0.5)
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 8))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = 150
    fill.data.size = 5
    
    # Camera
    bpy.ops.object.camera_add(location=(8, -8, 5))
    camera = bpy.context.active_object
    camera.name = "Camera"
    
    # Point camera at tank
    direction = tank_obj.location - camera.location
    import mathutils
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    bpy.context.scene.camera = camera
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 128
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = True
    
    # Ground plane (optional, for shadows)
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    
    # Ground material (shadow catcher)
    ground_mat = bpy.data.materials.new("GroundMaterial")
    ground_mat.use_nodes = True
    ground_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1)
    ground.data.materials.append(ground_mat)
    ground.is_shadow_catcher = True
    
    print("Render scene setup complete")


def main():
    """Main entry point."""
    import argparse
    
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description='Import tank with auto texture fixing')
    parser.add_argument('--asset-dir', '-a', required=True, help='Path to tank asset directory')
    parser.add_argument('--output', '-o', help='Output .blend file path')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--setup-scene', action='store_true', help='Setup render scene')
    parser.add_argument('--render', '-r', help='Render to this path')
    
    args = parser.parse_args(argv)
    
    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import tank
    tank = import_tank_auto(args.asset_dir, scale=args.scale)
    
    # Setup scene
    if args.setup_scene or args.render:
        setup_render_scene(tank)
    
    # Save
    if args.output:
        bpy.ops.wm.save_as_mainfile(filepath=args.output)
        print(f"Saved: {args.output}")
    
    # Render
    if args.render:
        bpy.context.scene.render.filepath = args.render
        bpy.ops.render.render(write_still=True)
        print(f"Rendered: {args.render}")


if __name__ == "__main__":
    main()
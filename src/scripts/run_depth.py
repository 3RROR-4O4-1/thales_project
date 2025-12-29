"""Run depth estimation on a test background image.

Usage:
    python src/scripts/run_depth_on_test.py
    python src/scripts/run_depth_on_test.py --input assets/backgrounds/test.jpg --output-dir outputs

This script loads an image, runs `estimate_depth` from the project's
depth estimation module, and writes a .npy depth map plus a colored PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run depth estimation on a test image")
    parser.add_argument("--input", "-i", default="assets/backgrounds/test.jpg",
                        help="Input image path (relative to src)")
    parser.add_argument("--output-dir", "-o", default="outputs",
                        help="Output directory to write depth maps and visualizations")
    parser.add_argument("--model", default="depth-anything-v2-large",
                        help="Depth model to use (depth-anything-v2-large|zoedepth|midas)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    args = parser.parse_args(argv)

    # Ensure project root is in sys.path so imports work when running from src
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import depth utilities from project
    try:
        from modules.scene_analysis.depth_estimation import estimate_depth, depth_to_colormap
    except Exception as e:
        print(f"Failed to import depth estimation module: {e}")
        return 2

    input_path = repo_root / args.input
    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        return 3

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image (PIL) and convert to RGB numpy
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    print(f"Running depth estimation on: {input_path}")
    print(f"Model: {args.model}, Device: {args.device}")

    try:
        depth = estimate_depth(img_np, model=args.model, device=args.device)
    except Exception as e:
        print(f"Depth estimation failed: {e}")
        return 4

    # Save raw depth (normalized float32 0-1)
    depth_path = output_dir / (input_path.stem + "_depth.npy")
    np.save(depth_path, depth)
    print(f"Saved depth numpy to: {depth_path}")

    # Save colored visualization
    try:
        colored = depth_to_colormap(depth, colormap="turbo", normalize=False)
        colored_img = Image.fromarray(colored)
        vis_path = output_dir / (input_path.stem + "_depth.png")
        colored_img.save(vis_path)
        print(f"Saved depth visualization to: {vis_path}")
    except Exception as e:
        print(f"Failed to create/save colored depth: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run SAM2 segmentation on a test background image.

Usage:
    python src/scripts/run_sam2_on_test.py
    python src/scripts/run_sam2_on_test.py --input assets/backgrounds/test.jpg --output-dir outputs

This script loads an image, runs the project's SAM-based segmenter,
and writes mask numpy files, mask PNGs, and an overlay visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SAM2 segmentation on a test image")
    parser.add_argument("--input", "-i", default="assets/backgrounds/test.jpg",
                        help="Input image path (relative to src)")
    parser.add_argument("--output-dir", "-o", default="outputs",
                        help="Output directory to write masks and visualizations")
    parser.add_argument("--model", default="sam2",
                        help="Segmentation model to use (sam2|grounded-dino|semantic)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--max-masks", type=int, default=20,
                        help="Maximum number of masks to save (largest first)")
    args = parser.parse_args(argv)

    # Ensure project root is in sys.path so imports work when running from src
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import segmentation utilities from project
    try:
        from modules.scene_analysis.segmentation import SceneSegmenter, SegmentationConfig
    except Exception as e:
        print(f"Failed to import segmentation module: {e}")
        return 2

    input_path = repo_root / args.input
    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        return 3

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    # Configure and run segmenter
    config = SegmentationConfig(model=args.model, device=args.device)
    seg = SceneSegmenter(config)

    print(f"Running segmentation on: {input_path}")
    print(f"Model: {args.model}, Device: {args.device}")

    try:
        # Use the lower-level SAM segmenter to get masks
        masks = seg.segmenter.segment_automatic(img_np)
    except Exception as e:
        print(f"Segmentation failed: {e}")
        return 4

    if not masks:
        print("No masks returned by segmenter")
        return 0

    # Sort masks by area (already sorted in segment_automatic, but ensure)
    masks.sort(key=lambda m: m.area, reverse=True)

    # Save masks and create overlay
    overlay = img_np.copy()
    h, w = img_np.shape[:2]

    rng = np.random.default_rng(42)
    max_save = min(args.max_masks, len(masks))

    for i, m in enumerate(masks[:max_save]):
        mask_np = (m.mask > 0).astype(np.uint8)

        mask_npy_path = output_dir / f"{input_path.stem}_mask_{i}.npy"
        np.save(mask_npy_path, mask_np)

        mask_png = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_png_path = output_dir / f"{input_path.stem}_mask_{i}.png"
        mask_png.save(mask_png_path)

        # Color overlay
        color = (rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256))
        colored_layer = np.zeros_like(overlay, dtype=np.uint8)
        for c in range(3):
            colored_layer[:, :, c] = (mask_np * color[c]).astype(np.uint8)

        alpha = 0.5 * (mask_np[..., None])
        overlay = (overlay * (1 - alpha) + colored_layer * alpha).astype(np.uint8)

    # Save overlay visualization
    try:
        overlay_img = Image.fromarray(overlay)
        overlay_path = output_dir / f"{input_path.stem}_masks_overlay.png"
        overlay_img.save(overlay_path)
        print(f"Saved overlay visualization to: {overlay_path}")
    except Exception as e:
        print(f"Failed to save overlay: {e}")

    print(f"Saved {max_save} masks to: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

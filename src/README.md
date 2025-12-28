# Vehicle Inpainting Pipeline

A comprehensive pipeline for generating synthetic training datasets by inserting 3D-rendered military vehicles into background images using generative AI (Flux 2 / Stable Diffusion).

## Overview

This pipeline automates the process of:
1. **Scene Analysis** - Detecting valid insertion zones using depth estimation and segmentation
2. **3D Rendering** - Multi-view vehicle rendering with Blender (RGB, depth, normal, mask)
3. **Scale-Aware Inpainting** - Crop-upscale-process-downscale-stitch workflow for small objects
4. **Edge Harmonization** - Differential blending, Poisson compositing, and shadow generation
5. **Quality Assurance** - CLIP scoring, artifact detection, and automatic filtering
6. **Annotation Export** - COCO and YOLO format dataset generation

## Project Structure

```
vehicle-inpainting-pipeline/
├── config/
│   └── pipeline_config.yaml      # Main configuration
├── modules/
│   ├── scene_analysis/           # Depth estimation, segmentation, zone detection
│   ├── rendering/                # Blender multi-view rendering
│   ├── inpainting/              # Scale-aware inpainting, ComfyUI API
│   ├── harmonization/           # Blending, color transfer, shadows
│   ├── quality/                 # Metrics, artifact detection, filtering
│   ├── annotation/              # COCO and YOLO export
│   └── utils.py                 # Common utilities
├── workflows/                    # ComfyUI workflow JSON files
├── assets/
│   ├── 3d_models/               # Source 3D vehicle models
│   ├── backgrounds/             # Background images
│   └── renders/                 # Pre-rendered vehicle views
├── output/
│   ├── generated/               # Generated images
│   ├── rejected/                # Failed QA images
│   └── dataset/                 # Exported datasets
├── orchestrator.py              # Main pipeline coordinator
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Blender 3.6+ (for 3D rendering)
- ComfyUI (for inpainting)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd vehicle-inpainting-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install segment-anything  # For SAM segmentation
pip install git+https://github.com/openai/CLIP.git  # For CLIP scoring
```

### Blender Setup

For 3D rendering, ensure Blender is installed and accessible:

```bash
# Linux
export PATH="/path/to/blender:$PATH"

# Verify
blender --version
```

### ComfyUI Setup

1. Install ComfyUI following their documentation
2. Install required custom nodes:
   - ComfyUI-Manager
   - ComfyUI-Inpaint-CropAndStitch
   - ComfyUI-ControlNet-Aux
3. Download models:
   - Flux 2 Dev FP8
   - ControlNet Depth
4. Copy workflow files from `workflows/` to ComfyUI

## Quick Start

### 1. Prepare Assets

```bash
# Place background images
cp /path/to/backgrounds/*.jpg assets/backgrounds/

# Place 3D models (OBJ, FBX, or Blend files)
cp /path/to/models/*.obj assets/3d_models/
```

### 2. Render Vehicle Views

```bash
# Run Blender rendering
blender --background \
    /home/lockin/Projects/vehicle_insertion/src/assets/3d_models/t-90/t90_scene.blend \
    --python blender_multiview.py -- \
    --output /home/lockin/Projects/vehicle_insertion/src/assets/renders/t90 \
    --target Tank \
    --normalize-scale 10 \
    --samples 64 \
    --save-blend
```

### 3. Run Pipeline

```python
from orchestrator import PipelineOrchestrator, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    backgrounds_dir="assets/backgrounds",
    renders_dir="assets/renders",
    output_dir="output/generated",
    num_insertions_per_background=3
)

# Initialize
pipeline = PipelineOrchestrator(config)

# Load vehicle renders
renders = pipeline.load_vehicle_renders()

# Process backgrounds
backgrounds = ["assets/backgrounds/scene1.jpg", "assets/backgrounds/scene2.jpg"]
results, stats = pipeline.run_batch(backgrounds, renders)

# Export dataset
pipeline.export_dataset(results, "my_dataset")
```

### 4. Command Line Usage

```bash
python orchestrator.py \
    --backgrounds assets/backgrounds \
    --renders assets/renders \
    --output output \
```

## Configuration

Edit `config/pipeline_config.yaml` to customize:

```yaml
scene_analysis:
  depth:
    model: "depth-anything-v2-large"
  segmentation:
    model: "sam2-hiera-large"

inpainting:
  model: "flux2-dev-fp8"
  scale_aware:
    processing_resolution: 1024
    context_expand_factor: 2.0

harmonization:
  blending:
    method: "hybrid"  # differential, poisson, or hybrid
  shadow:
    enabled: true
    intensity: 0.4

quality_assurance:
  metrics:
    clip_score:
      threshold: 0.25
    artifact_detection:
      threshold: 0.15
```

## Module Usage

### Scene Analysis

```python
from modules.scene_analysis import SceneAnalyzer

analyzer = SceneAnalyzer()
scene_info = analyzer.analyze(image)

# Access results
depth_map = scene_info['depth']
valid_zones = scene_info['valid_zones']
```

### Scale-Aware Inpainting

```python
from modules.inpainting import ScaleAwareInpainter, ScaleAwareConfig

config = ScaleAwareConfig(
    processing_resolution=1024,
    context_expand_factor=2.0,
    mask_blur_pixels=28
)

inpainter = ScaleAwareInpainter(config, inpaint_fn=your_inpaint_function)
result = inpainter.process(image, mask, vehicle_render=vehicle_rgb)
```

### Edge Harmonization

```python
from modules.harmonization import EdgeHarmonizer, HarmonizationConfig

config = HarmonizationConfig(
    blend_method="hybrid",
    feather_pixels=32,
    shadow_generation=True
)

harmonizer = EdgeHarmonizer(config)
result = harmonizer.harmonize(background, generated, mask)
```

### Quality Assurance

```python
from modules.quality import QualityAssurance, QAConfig

config = QAConfig(
    clip_threshold=0.25,
    artifact_threshold=0.15
)

qa = QualityAssurance(config)
result = qa.evaluate(generated, background, mask)

if result.passed:
    print("Image passed QA")
else:
    print(f"Failed: {result.failure_reasons}")
```

### Annotation Export

```python
from modules.annotation import COCOExporter, YOLOExporter

# COCO format
coco = COCOExporter("output/coco_dataset")
coco.add_image(image, [{"mask": mask, "category_id": 1}])
coco.save()

# YOLO format
yolo = YOLOExporter("output/yolo_dataset")
yolo.add_image(image, [{"mask": mask, "class_id": 0}])
yolo.save()
```

## Key Features

### Differential Blending

Prevents visible seams by using gradient masks:
- Full strength at object center
- Gradual falloff to edges
- Configurable feather radius

### Poisson Blending

Photoshop-quality seamless compositing:
- Gradient domain editing
- Preserves edge detail
- Automatic color harmonization

### Scale-Aware Processing

Handles tiny objects effectively:
1. Crop with expanded context
2. Upscale to processing resolution
3. Inpaint at high resolution
4. Downscale with antialiasing
5. Stitch with feathered blending

### Quality Assurance

Multi-metric filtering:
- **CLIP Score**: Image-text alignment
- **Artifact Detection**: Blur, distortion, anomalies
- **Edge Consistency**: Seam visibility
- **Color Harmony**: Distribution matching

## Troubleshooting

### Edge Artifacts

If you see visible seams:
1. Increase `feather_pixels` (try 48-64)
2. Enable Poisson blending fallback
3. Increase `context_expand_factor`

### Blurry Small Objects

For tiny vehicle insertions:
1. Increase `processing_resolution` to 1024+
2. Ensure `context_expand_factor` >= 2.0
3. Use Lanczos downscaling

### Color Mismatch

If inserted vehicles look out of place:
1. Enable color harmonization
2. Increase `context_dilation` for color sampling
3. Enable lighting matching

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- Anthropic for AI assistance
- ComfyUI community for inpainting workflows
- Depth Anything, SAM, and CLIP teams for models

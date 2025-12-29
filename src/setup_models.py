#!/usr/bin/env python3
"""
Model Setup Script - Downloads all required models (LARGE/BF16 variants)

Downloads:
    - Depth Anything V2 Large (~1.3GB)
    - SAM2 Hiera Large (~900MB)  
    - CLIP ViT-L/14 (~890MB)
    - FLUX.2-dev BF16 (~64GB) - Full precision diffusion model
    - FLUX.2 Mistral Text Encoder BF16 (~48GB)
    - FLUX.2 VAE (~336MB)

Total: ~115GB+ of models (BF16 full precision)

For lower VRAM, use --fp8 flag to download FP8 quantized versions instead (~30GB total)

Usage:
    python setup_models.py                    # Download all models (BF16 LARGE)
    python setup_models.py --fp8              # Download FP8 quantized versions
    python setup_models.py --check            # Check what's installed
    python setup_models.py --list             # List all models
    python setup_models.py --component sam2   # Download specific component
    python setup_models.py --comfyui-path /path/to/ComfyUI  # Set ComfyUI path

Requirements:
    pip install torch torchvision transformers huggingface_hub
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# ============================================================================
# Model Configuration
# ============================================================================

CACHE_DIR = Path.home() / ".cache" / "vehicle_inpainting"

# FLUX.2 Model URLs (from Comfy-Org and Black Forest Labs)
FLUX2_MODELS = {
    # BF16 Full Precision (LARGE)
    "bf16": {
        "diffusion_model": {
            "url": "https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/flux2-dev.safetensors",
            "filename": "flux2-dev.safetensors",
            "size": "64.4GB",
            "repo": "black-forest-labs/FLUX.2-dev",
        },
        "text_encoder": {
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
            "filename": "mistral_3_small_flux2_bf16.safetensors",
            "size": "~48GB",
        },
        "vae": {
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
            "filename": "flux2-vae.safetensors",
            "size": "336MB",
        },
    },
    # FP8 Quantized (for lower VRAM)
    "fp8": {
        "diffusion_model": {
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
            "filename": "flux2_dev_fp8mixed.safetensors",
            "size": "~32GB",
        },
        "text_encoder": {
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
            "filename": "mistral_3_small_flux2_fp8.safetensors",
            "size": "~24GB",
        },
        "vae": {
            "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
            "filename": "flux2-vae.safetensors",
            "size": "336MB",
        },
    },
}

MODELS = {
    # -------------------------------------------------------------------------
    # Depth Anything V2 - Large
    # -------------------------------------------------------------------------
    "depth_anything_v2": {
        "name": "Depth Anything V2 Large",
        "description": "State-of-the-art monocular depth estimation",
        "source": "huggingface",
        "model_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "size": "~1.3GB",
        "vram": "~8GB",
        "auto_download": True,
    },
    
    # -------------------------------------------------------------------------
    # SAM2 - Hiera Large
    # -------------------------------------------------------------------------
    "sam2": {
        "name": "SAM2 Hiera Large",
        "description": "Segment Anything Model 2 - improved segmentation",
        "source": "github",
        "repo": "https://github.com/facebookresearch/sam2",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "checkpoint_name": "sam2.1_hiera_large.pt",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        "config_name": "sam2.1_hiera_l.yaml",
        "size": "~900MB",
        "vram": "~8GB",
        "cache_subdir": "sam2",
    },
    
    # -------------------------------------------------------------------------
    # CLIP - ViT-L/14 (Largest)
    # -------------------------------------------------------------------------
    "clip": {
        "name": "CLIP ViT-L/14",
        "description": "OpenAI CLIP for image-text alignment scoring",
        "source": "openai",
        "model_variant": "ViT-L/14",
        "size": "~890MB",
        "vram": "~4GB",
        "pip_package": "git+https://github.com/openai/CLIP.git",
        "auto_download": True,
    },
    
    # -------------------------------------------------------------------------
    # FLUX.2-dev - 32B Flow Matching Transformer (November 2025)
    # -------------------------------------------------------------------------
    "flux2": {
        "name": "FLUX.2-dev",
        "description": "32B rectified flow transformer for image generation & editing",
        "source": "huggingface",
        "model_id": "black-forest-labs/FLUX.2-dev",
        "size_bf16": "~115GB total (BF16)",
        "size_fp8": "~57GB total (FP8)",
        "vram_bf16": "~90GB (or use weight streaming)",
        "vram_fp8": "~24GB with weight streaming",
        "requires_login": True,
        "license_note": "FLUX.2-dev Non-Commercial License",
        "features": [
            "Multi-reference conditioning (up to 10 images)",
            "4MP resolution generation/editing",
            "Mistral Small 3.1 24B text encoder",
            "Improved text rendering, lighting, hands",
        ],
        "comfyui_paths": {
            "diffusion_models": "models/diffusion_models",
            "text_encoders": "models/text_encoders", 
            "vae": "models/vae",
        },
    },
}


# ============================================================================
# Utility Functions
# ============================================================================

def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def download_file(url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download a file with progress."""
    import urllib.request
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return True
    
    print(f"  Downloading: {url.split('/')[-1]}")
    print(f"  To: {dest}")
    
    try:
        if show_progress:
            if shutil.which("wget"):
                subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
            elif shutil.which("curl"):
                subprocess.run(["curl", "-L", "-#", "-o", str(dest), url], check=True)
            else:
                urllib.request.urlretrieve(url, dest)
        else:
            urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        if dest.exists():
            dest.unlink()  # Remove partial download
        return False


def check_pip_package(package: str) -> bool:
    """Check if a pip package is installed."""
    try:
        pkg_name = package.split('[')[0].replace('-', '_').split('/')[-1]
        result = subprocess.run(
            [sys.executable, "-c", f"import {pkg_name}"],
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False


def install_pip_package(package: str) -> bool:
    """Install a pip package."""
    print(f"  Installing: {package}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ ERROR: {e.stderr}")
        return False


def get_comfyui_path() -> Optional[Path]:
    """Try to find ComfyUI installation."""
    possible_paths = [
        Path.home() / "ComfyUI",
        Path.home() / "comfyui", 
        Path("/opt/ComfyUI"),
        Path("./ComfyUI"),
        Path("../ComfyUI"),
    ]
    
    if "COMFYUI_PATH" in os.environ:
        return Path(os.environ["COMFYUI_PATH"])
    
    for p in possible_paths:
        if p.exists() and (p / "main.py").exists():
            return p
    
    return None


def check_hf_login() -> bool:
    """Check if logged into HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"  ✓ Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"  ✗ Not logged in: {e}")
        return False


def hf_download(repo_id: str, filename: str, local_dir: Path, token: bool = True) -> bool:
    """Download file from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"  Downloading {filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# ============================================================================
# Model Setup Functions
# ============================================================================

def setup_depth_anything_v2(force: bool = False) -> bool:
    """Setup Depth Anything V2 Large."""
    print("\n" + "=" * 70)
    print("  DEPTH ANYTHING V2 LARGE")
    print("=" * 70)
    
    config = MODELS["depth_anything_v2"]
    print(f"  Size: {config['size']} | VRAM: {config['vram']}")
    
    if not check_pip_package("transformers"):
        install_pip_package("transformers")
    
    print("  Pre-downloading model from HuggingFace...")
    try:
        from transformers import pipeline
        pipe = pipeline(
            "depth-estimation",
            model=config["model_id"],
            device="cpu"
        )
        del pipe
        print("  ✓ Depth Anything V2 Large ready!")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def setup_sam2(force: bool = False) -> bool:
    """Setup SAM2 Hiera Large."""
    print("\n" + "=" * 70)
    print("  SAM2 HIERA LARGE")
    print("=" * 70)
    
    config = MODELS["sam2"]
    print(f"  Size: {config['size']} | VRAM: {config['vram']}")
    
    cache_dir = CACHE_DIR / config["cache_subdir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Install sam2 package
    print("  Installing SAM2 package...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sam2", "-q"],
            check=True, capture_output=True
        )
    except:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", 
                 "git+https://github.com/facebookresearch/sam2.git", "-q"],
                check=True, capture_output=True
            )
        except Exception as e:
            print(f"  ⚠ Could not install sam2 package: {e}")
    
    # Download checkpoint
    checkpoint_path = cache_dir / config["checkpoint_name"]
    if not download_file(config["checkpoint_url"], checkpoint_path):
        return False
    
    # Download config
    config_path = cache_dir / config["config_name"]
    download_file(config["config_url"], config_path)
    
    # Create symlink in standard location
    standard_path = Path.home() / ".cache" / "sam2"
    standard_path.mkdir(parents=True, exist_ok=True)
    
    standard_checkpoint = standard_path / config["checkpoint_name"]
    if not standard_checkpoint.exists():
        try:
            standard_checkpoint.symlink_to(checkpoint_path)
        except:
            shutil.copy2(checkpoint_path, standard_checkpoint)
    
    print(f"  ✓ SAM2 Hiera Large ready!")
    return True


def setup_clip(force: bool = False) -> bool:
    """Setup CLIP ViT-L/14."""
    print("\n" + "=" * 70)
    print("  CLIP ViT-L/14")
    print("=" * 70)
    
    config = MODELS["clip"]
    print(f"  Size: {config['size']} | VRAM: {config['vram']}")
    
    if not check_pip_package("clip"):
        print("  Installing CLIP from GitHub...")
        install_pip_package("git+https://github.com/openai/CLIP.git")
    
    print("  Pre-downloading CLIP ViT-L/14...")
    try:
        import clip
        model, preprocess = clip.load(config["model_variant"], device="cpu")
        del model, preprocess
        print("  ✓ CLIP ViT-L/14 ready!")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def setup_flux2(comfyui_path: Path, precision: str = "bf16", force: bool = False) -> bool:
    """Setup FLUX.2-dev for ComfyUI."""
    print("\n" + "=" * 70)
    print(f"  FLUX.2-dev ({precision.upper()})")
    print("=" * 70)
    
    config = MODELS["flux2"]
    flux_config = FLUX2_MODELS[precision]
    
    total_size = config[f"size_{precision}"]
    vram = config[f"vram_{precision}"]
    print(f"  Total Size: {total_size} | VRAM: {vram}")
    print(f"  Features: {', '.join(config['features'][:2])}")
    
    if not comfyui_path or not comfyui_path.exists():
        print("  ✗ ComfyUI path not found!")
        print("    Specify with --comfyui-path or set COMFYUI_PATH env var")
        return False
    
    print(f"  ComfyUI: {comfyui_path}")
    
    # Check HuggingFace login for BF16 (requires license acceptance)
    if precision == "bf16":
        print("\n  Checking HuggingFace login (required for BF16)...")
        if not check_hf_login():
            print("  ✗ Please login first: huggingface-cli login")
            print("  ✗ And accept license at: https://huggingface.co/black-forest-labs/FLUX.2-dev")
            return False
    
    success = True
    
    # -------------------------------------------------------------------------
    # 1. Diffusion Model
    # -------------------------------------------------------------------------
    print(f"\n  [1/3] Diffusion Model ({flux_config['diffusion_model']['size']})")
    dm_dir = comfyui_path / "models" / "diffusion_models"
    dm_dir.mkdir(parents=True, exist_ok=True)
    
    dm_file = dm_dir / flux_config['diffusion_model']['filename']
    if dm_file.exists() and not force:
        print(f"    ✓ Already exists: {dm_file.name}")
    else:
        if precision == "bf16" and "repo" in flux_config['diffusion_model']:
            # Use HF hub for BF16 (requires auth)
            if not hf_download(
                flux_config['diffusion_model']['repo'],
                flux_config['diffusion_model']['filename'],
                dm_dir
            ):
                success = False
        else:
            # Direct download for FP8
            if not download_file(flux_config['diffusion_model']['url'], dm_file):
                success = False
    
    # -------------------------------------------------------------------------
    # 2. Text Encoder (Mistral Small 3.1)
    # -------------------------------------------------------------------------
    print(f"\n  [2/3] Text Encoder - Mistral 3.1 ({flux_config['text_encoder']['size']})")
    te_dir = comfyui_path / "models" / "text_encoders"
    te_dir.mkdir(parents=True, exist_ok=True)
    
    te_file = te_dir / flux_config['text_encoder']['filename']
    if te_file.exists() and not force:
        print(f"    ✓ Already exists: {te_file.name}")
    else:
        if not download_file(flux_config['text_encoder']['url'], te_file):
            success = False
    
    # -------------------------------------------------------------------------
    # 3. VAE
    # -------------------------------------------------------------------------
    print(f"\n  [3/3] VAE ({flux_config['vae']['size']})")
    vae_dir = comfyui_path / "models" / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    
    vae_file = vae_dir / flux_config['vae']['filename']
    if vae_file.exists() and not force:
        print(f"    ✓ Already exists: {vae_file.name}")
    else:
        if not download_file(flux_config['vae']['url'], vae_file):
            success = False
    
    if success:
        print(f"\n  ✓ FLUX.2-dev ({precision.upper()}) ready!")
    
    return success


# ============================================================================
# Check Functions
# ============================================================================

def check_installation(comfyui_path: Optional[Path] = None) -> Dict[str, bool]:
    """Check what's installed."""
    print("\n" + "=" * 70)
    print("  INSTALLATION STATUS")
    print("=" * 70)
    
    status = {}
    
    # Check Depth Anything V2
    print("\n[Depth Anything V2 Large]")
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(
            "depth-anything/Depth-Anything-V2-Large-hf",
            "model.safetensors"
        )
        if cached:
            print("  ✓ Downloaded and cached")
            status["depth_anything_v2"] = True
        else:
            print("  ✗ Not downloaded")
            status["depth_anything_v2"] = False
    except Exception as e:
        print(f"  ? Unknown: {e}")
        status["depth_anything_v2"] = False
    
    # Check SAM2
    print("\n[SAM2 Hiera Large]")
    sam2_paths = [
        CACHE_DIR / "sam2" / "sam2.1_hiera_large.pt",
        Path.home() / ".cache" / "sam2" / "sam2.1_hiera_large.pt",
    ]
    sam2_found = any(p.exists() for p in sam2_paths)
    if sam2_found:
        print("  ✓ Checkpoint found")
        status["sam2"] = True
    else:
        print("  ✗ Checkpoint not found")
        status["sam2"] = False
    
    # Check CLIP
    print("\n[CLIP ViT-L/14]")
    try:
        import clip
        clip_cache = Path.home() / ".cache" / "clip"
        if clip_cache.exists() and any(clip_cache.glob("*.pt")):
            print("  ✓ Downloaded and cached")
            status["clip"] = True
        else:
            print("  ? Package installed, model may need download")
            status["clip"] = True
    except ImportError:
        print("  ✗ CLIP package not installed")
        status["clip"] = False
    
    # Check FLUX.2 models
    print("\n[FLUX.2-dev (ComfyUI)]")
    if comfyui_path and comfyui_path.exists():
        print(f"  ComfyUI: {comfyui_path}")
        
        flux_files = {
            "BF16": [
                ("diffusion_models/flux2-dev.safetensors", "Diffusion Model (BF16)"),
                ("text_encoders/mistral_3_small_flux2_bf16.safetensors", "Text Encoder (BF16)"),
            ],
            "FP8": [
                ("diffusion_models/flux2_dev_fp8mixed.safetensors", "Diffusion Model (FP8)"),
                ("text_encoders/mistral_3_small_flux2_fp8.safetensors", "Text Encoder (FP8)"),
            ],
            "Common": [
                ("vae/flux2-vae.safetensors", "VAE"),
            ],
        }
        
        flux_status = {"bf16": True, "fp8": True, "vae": True}
        
        for category, files in flux_files.items():
            for filepath, name in files:
                full_path = comfyui_path / "models" / filepath
                if full_path.exists():
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {name}: {size_mb:.0f}MB")
                else:
                    print(f"  ✗ {name}: not found")
                    if "BF16" in category:
                        flux_status["bf16"] = False
                    elif "FP8" in category:
                        flux_status["fp8"] = False
                    else:
                        flux_status["vae"] = False
        
        status["flux2_bf16"] = flux_status["bf16"] and flux_status["vae"]
        status["flux2_fp8"] = flux_status["fp8"] and flux_status["vae"]
        status["flux2"] = status["flux2_bf16"] or status["flux2_fp8"]
    else:
        print("  ? ComfyUI not found")
        status["flux2"] = False
    
    return status


def print_model_list():
    """Print list of all models."""
    print("\n" + "=" * 70)
    print("  AVAILABLE MODELS")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  DEPTH ANYTHING V2 LARGE                                            │
├─────────────────────────────────────────────────────────────────────┤
│  Purpose:    Monocular depth estimation for scene analysis          │
│  Size:       ~1.3GB                                                 │
│  VRAM:       ~8GB                                                   │
│  Source:     HuggingFace (auto-download)                            │
│  Model ID:   depth-anything/Depth-Anything-V2-Large-hf              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SAM2 HIERA LARGE                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Purpose:    Segment Anything Model 2 for object segmentation       │
│  Size:       ~900MB                                                 │
│  VRAM:       ~8GB                                                   │
│  Source:     Facebook Research (manual download)                    │
│  Checkpoint: sam2.1_hiera_large.pt                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CLIP ViT-L/14                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Purpose:    Image-text alignment for quality assessment            │
│  Size:       ~890MB                                                 │
│  VRAM:       ~4GB                                                   │
│  Source:     OpenAI (auto-download)                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  FLUX.2-dev (November 2025)                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Purpose:    32B flow transformer for image generation & editing    │
│                                                                     │
│  BF16 (Full Precision - LARGE):                                     │
│    • Diffusion Model:  flux2-dev.safetensors (~64GB)                │
│    • Text Encoder:     mistral_3_small_flux2_bf16.safetensors       │
│    • VAE:              flux2-vae.safetensors (336MB)                │
│    • Total:            ~115GB                                       │
│    • VRAM:             ~90GB (or weight streaming)                  │
│                                                                     │
│  FP8 (Quantized - for consumer GPUs):                               │
│    • Diffusion Model:  flux2_dev_fp8mixed.safetensors (~32GB)       │
│    • Text Encoder:     mistral_3_small_flux2_fp8.safetensors        │
│    • VAE:              flux2-vae.safetensors (336MB)                │
│    • Total:            ~57GB                                        │
│    • VRAM:             ~24GB with weight streaming                  │
│                                                                     │
│  Features:                                                          │
│    • Multi-reference conditioning (up to 10 images)                 │
│    • 4MP resolution generation/editing                              │
│    • Mistral Small 3.1 24B text encoder                             │
│    • Improved text rendering, lighting, hands                       │
│                                                                     │
│  License:    FLUX.2-dev Non-Commercial License                      │
│  Source:     black-forest-labs/FLUX.2-dev (requires HF login)       │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("-" * 70)
    print("  TOTAL DOWNLOAD SIZE:")
    print("    BF16 (LARGE):  ~118GB")
    print("    FP8:           ~60GB")
    print("-" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup models for Vehicle Inpainting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_models.py                          # Download all (BF16 LARGE)
    python setup_models.py --fp8                    # Download FP8 quantized
    python setup_models.py --check                  # Check installation status
    python setup_models.py --list                   # List all models
    python setup_models.py --component sam2 clip    # Download specific models
    python setup_models.py --comfyui-path ~/ComfyUI # Specify ComfyUI location
    python setup_models.py --skip-flux              # Skip Flux models

ComfyUI Model Paths:
    models/diffusion_models/  - flux2-dev.safetensors (or fp8mixed)
    models/text_encoders/     - mistral_3_small_flux2_bf16.safetensors
    models/vae/               - flux2-vae.safetensors
        """
    )
    
    parser.add_argument(
        "--check", action="store_true",
        help="Check installation status"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all models"
    )
    parser.add_argument(
        "--component", "-c", nargs="+",
        choices=["depth", "sam2", "clip", "flux2"],
        help="Install specific components"
    )
    parser.add_argument(
        "--comfyui-path", type=Path,
        help="Path to ComfyUI installation"
    )
    parser.add_argument(
        "--fp8", action="store_true",
        help="Use FP8 quantized models instead of BF16"
    )
    parser.add_argument(
        "--skip-flux", action="store_true",
        help="Skip Flux model downloads"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-download"
    )
    
    args = parser.parse_args()
    
    # Determine precision
    precision = "fp8" if args.fp8 else "bf16"
    
    # Handle ComfyUI path
    comfyui_path = args.comfyui_path or get_comfyui_path()
    if comfyui_path:
        os.environ["COMFYUI_PATH"] = str(comfyui_path)
    
    # List models
    if args.list:
        print_model_list()
        return 0
    
    # Check installation
    if args.check:
        status = check_installation(comfyui_path)
        all_good = all(status.values())
        return 0 if all_good else 1
    
    # Determine what to install
    components = args.component or ["depth", "sam2", "clip"]
    if not args.skip_flux and "flux2" not in components and not args.component:
        components.append("flux2")
    
    print("=" * 70)
    print("  VEHICLE INPAINTING PIPELINE - MODEL SETUP")
    print("=" * 70)
    print(f"\n  Components: {', '.join(components)}")
    print(f"  Precision:  {precision.upper()}")
    print(f"  ComfyUI:    {comfyui_path or 'Not found'}")
    
    if "flux2" in components:
        if precision == "bf16":
            print(f"\n  ⚠️  BF16 models are LARGE (~115GB total)")
            print(f"     Use --fp8 for quantized versions (~57GB)")
        else:
            print(f"\n  Using FP8 quantized models (~57GB total)")
    
    # Confirm
    response = input("\n  Continue? [y/N] ")
    if response.lower() != 'y':
        print("  Aborted.")
        return 0
    
    # Install components
    results = {}
    
    if "depth" in components:
        results["depth"] = setup_depth_anything_v2(args.force)
    
    if "sam2" in components:
        results["sam2"] = setup_sam2(args.force)
    
    if "clip" in components:
        results["clip"] = setup_clip(args.force)
    
    if "flux2" in components:
        if not comfyui_path:
            print("\n  ⚠️  Skipping FLUX.2: ComfyUI path not found")
            print("     Specify with --comfyui-path or set COMFYUI_PATH")
            results["flux2"] = False
        else:
            results["flux2"] = setup_flux2(comfyui_path, precision, args.force)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SETUP SUMMARY")
    print("=" * 70)
    
    all_success = True
    for component, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {component}")
        if not success:
            all_success = False
    
    if all_success:
        print("\n  ✓ All models installed successfully!")
    else:
        print("\n  ⚠️  Some models failed. Check errors above.")
    
    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print(f"""
  1. If FLUX.2 download failed, ensure HuggingFace login:
     huggingface-cli login
     
  2. Accept the FLUX.2 license:
     https://huggingface.co/black-forest-labs/FLUX.2-dev

  3. Start ComfyUI:
     cd {comfyui_path or '/path/to/ComfyUI'} && python main.py

  4. Load the FLUX.2 workflow from your JSON file

  5. Run the vehicle inpainting pipeline:
     cd src && python orchestrator.py --help
""")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())


"""
huggingface-cli login

# 2. Accept the FLUX.2 license (open this URL in browser)
# https://huggingface.co/black-forest-labs/FLUX.2-dev
# Click "Agree" to accept the license terms
"""

# python setup_models.py --fp8 --comfyui-path /home/lockin/Documents/GitHub/ComfyUI
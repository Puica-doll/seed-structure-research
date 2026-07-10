"""
00_test_setup.py  —  Quick environment verification
Paper: Towards Reproducible AI-Generated Scientific Visualizations

Generates 5 test images (1 per output type, seed=0) to verify:
  - CUDA is available
  - SD 1.5 loads and runs
  - Fidelity scoring works
  - Disk write is OK

Expected run time: ~2-3 minutes on RTX 3070
"""

import sys
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Single .safetensors file — no symlinks, no config mismatches
MODEL_FILE     = Path(__file__).parent / "sd15_model" / "v1-5-pruned-emaonly.safetensors"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32
BASE_DIR       = Path(__file__).parent
TEST_DIR       = BASE_DIR / "experiment" / "test_run"

TEST_PROMPTS = {
    "bar_chart":       "a clean simple bar chart with exactly 5 vertical bars in strictly descending height order from left to right, plain white background, no text, no labels, minimal style",
    "line_chart":      "a clean simple line graph with a single line strictly decreasing from left to right, plain white background, no text, no labels, minimal style",
    "scatter_plot":    "a clean simple scatter plot with data points clustered tightly in the top-right corner, plain white background, no text, no labels, minimal style",
    "network_diagram": "a clean simple network diagram with exactly 6 circular nodes connected by 8 directed arrows, plain white background, no text, no labels, minimal style",
    "flowchart":       "a clean simple flowchart with exactly 6 rectangular process boxes and 1 diamond decision shape with arrows flowing downward, plain white background, no text, no labels",
}

# ── Basic fidelity check (pixel coverage proxy) ─────────────────────────────

def basic_fidelity_check(img_path):
    """
    Lightweight check: measures how much non-white content the image has.
    Returns a rough score in [0, 1]. A chart/diagram typically covers
    10-60% of the canvas; a blank or fully-saturated image scores low.
    """
    img  = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coverage = np.sum(binary > 0) / binary.size
    # Penalize extremes: < 5% (almost blank) or > 80% (no white background)
    if coverage < 0.05 or coverage > 0.80:
        return round(coverage, 4), "LOW (image may not be a clean chart)"
    return round(coverage, 4), "OK"

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment Environment Verification")
    print("=" * 60)

    # 1. CUDA check
    print(f"\n[1] PyTorch  : {torch.__version__}")
    print(f"    CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU      : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    VRAM     : {mem_gb:.1f} GB")
    else:
        print("    WARNING: CUDA not found — running on CPU (will be very slow)")

    # 2. Load model
    print(f"\n[2] Loading {MODEL_FILE.name} ...")
    t0 = time.time()
    if not MODEL_FILE.exists():
        print(f"\nERROR: Model file not found at:\n  {MODEL_FILE}")
        print("Run this first (venv active, as Administrator):")
        print('  huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 v1-5-pruned-emaonly.safetensors --local-dir "C:\\Users\\Ticudean Diana\\CLAUDE\\Research\\04_Seed-Structure-Paper\\sd15_model"')
        sys.exit(1)

    pipe = StableDiffusionPipeline.from_single_file(
        str(MODEL_FILE),
        torch_dtype=DTYPE,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    print(f"    Loaded in {time.time()-t0:.1f}s")

    # 3. Generate test images
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[3] Generating 5 test images → {TEST_DIR}")
    results = []
    for output_type, prompt in TEST_PROMPTS.items():
        t1 = time.time()
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        image = pipe(
            prompt,
            num_inference_steps=20,    # fewer steps for speed in test
            guidance_scale=7.5,
            generator=generator,
            height=512, width=512,
        ).images[0]
        img_path = TEST_DIR / f"{output_type}_seed0.png"
        image.save(img_path)
        gen_time = time.time() - t1

        # 4. Score
        coverage, status = basic_fidelity_check(img_path)
        results.append((output_type, coverage, status, gen_time))
        print(f"    {output_type:<20} coverage={coverage:.3f}  {status}  ({gen_time:.1f}s)")

    # 5. Summary
    print("\n" + "=" * 60)
    avg_time = sum(r[3] for r in results) / len(results)
    eta_hours = avg_time * 50_000 / 3600
    print(f"Average generation time : {avg_time:.1f}s / image")
    print(f"Estimated full run time : {eta_hours:.0f} hours for 50,000 images")
    print(f"Test images saved to    : {TEST_DIR}")
    print("\nIf everything looks OK, run 01_generate_and_score.py to start the experiment.")
    print("=" * 60)

if __name__ == "__main__":
    main()

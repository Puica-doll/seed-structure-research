# Seed-Structure Mapping for AI-Generated Scientific Visualizations

**Author:** Diana Ticudean  
**Affiliation:** Technical University of Cluj-Napoca  
**Status:** Active experiment — data collection in progress (July 2026)

---

## Overview

This repository contains the code and working paper for an ongoing research project investigating whether the random seed used in diffusion-based image generation models deterministically influences the structural fidelity of the output — specifically for scientific visualizations such as charts, diagrams, and flowcharts.

The core hypothesis: because Stable Diffusion with a DDIM scheduler is fully deterministic, the same seed always produces the same output. This means seeds can be characterized in advance as "structurally favorable" or "structurally unfavorable" for a given output type — without generating the image first.

This work builds directly on:
> Ticudean, D. (2026). *The PRNG Trap in Browser Simulations*. Zenodo. https://doi.org/10.5281/zenodo.21235031

---

## Repository Structure

```
├── 00_test_setup.py               # Environment verification (generates 5 test images)
├── 01_generate_and_score.py       # Main experiment: 50,000 image generation + CV scoring
├── build_paper_docx.js            # Builds the working paper as a .docx file
├── seed_structure_paper_working_draft.docx   # Working paper (Sections 1–4 complete)
├── experiment/
│   ├── results.csv                # Fidelity scores per seed/prompt (grows as experiment runs)
│   └── test_run/                  # 5 sample images from environment verification
└── README.md
```

> **Note:** The `experiment/images/` folder (~10 GB, 50,000 PNG files) is excluded from this repository via `.gitignore`. The `results.csv` file contains all scoring data needed to reproduce the analysis.

---

## Experiment Design

| Parameter | Value |
|---|---|
| Model | Stable Diffusion 1.5 (`v1-5-pruned-emaonly.safetensors`) |
| Scheduler | DDIM (deterministic) |
| Inference steps | 20 |
| Guidance scale | 7.5 |
| Image size | 512 × 512 |
| Seeds | 0 – 999 (1,000 per prompt) |
| Output types | 5 (bar chart, line chart, scatter plot, network diagram, flowchart) |
| Prompts per type | 10 |
| Total images | 50,000 |
| Hardware | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM) |

### Output Types and Structural Fidelity Scoring

Each generated image is automatically scored using a computer vision pipeline (OpenCV) for structural fidelity — measuring how closely the image matches the specified structural specification (e.g., correct number of bars, correct ordering, correct node count).

Scores range from 0.0 to 1.0. Images scoring ≥ 0.70 are classified as "accepted."

---

## Running the Experiment

### Prerequisites

```bash
python -m venv venv
venv\Scripts\activate
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate opencv-python scikit-learn xgboost pandas
```

Download the model weights (required, not included in repo):
```bash
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 v1-5-pruned-emaonly.safetensors --local-dir sd15_model
```

### Verify environment
```bash
python 00_test_setup.py
```

### Run experiment (resumable)
```bash
python 01_generate_and_score.py
```

The experiment is fully resumable — if interrupted, re-run the same command and it continues from where it left off.

---

## Paper

The working paper is available in this repository as `seed_structure_paper_working_draft.docx`.

**Title:** Towards Reproducible AI-Generated Scientific Visualizations: A Seed-Structure Mapping Approach for Structured Output Optimization in Diffusion-Based Models

Sections 1–4 are complete. Sections 5–8 (Discussion, Conclusion, Future Work) will be written after experimental results are available.

---

## License

Code: MIT  
Paper: CC BY 4.0

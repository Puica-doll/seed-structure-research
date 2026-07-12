# Seed-Structure Mapping for AI-Generated Scientific Visualizations

**Author:** Diana Ticudean  
**Affiliation:** Technical University of Cluj-Napoca, Faculty of Industrial Engineering, Robotics and Management of Production  
**Status:** Complete — paper finalized July 2026

---

## Overview

This repository contains the full code, data, and paper for a research project investigating whether the random seed used in diffusion-based image generation models deterministically influences the structural fidelity of the output — specifically for scientific visualizations such as charts, diagrams, and flowcharts.

**Core finding:** A lightweight XGBoost mapping model trained on 50,000 seed-prompt-fidelity triples improves first-pass acceptance rate from 21.7% (random selection) to 64.0% — an absolute gain of 42.3 percentage points. Feature importance analysis reveals that output type accounts for 94.2% of predictive weight, establishing that structural achievability is primarily type-determined rather than seed-determined within SD 1.5.

This work builds directly on:
> Ticudean, D. (2026). *The PRNG Trap in Browser Simulations*. Zenodo. https://doi.org/10.5281/zenodo.21235031

---

## Paper

**Title:** Towards Reproducible AI-Generated Scientific Visualizations: A Seed-Structure Mapping Approach for Structured Output Optimization in Diffusion-Based Models

The final paper is available in this repository as `Towards Reproducible AI-Generated Scientific Visualizations_Final.docx`

---

## Repository Structure

```
├── 00_test_setup.py                    # Environment verification (5 test images)
├── 01_generate_and_score.py            # Main experiment: 50,000 images + CV scoring
├── 02_train_mapping_model.py           # Train XGBoost, MLP, LR; compute AUC-ROC
├── 03_evaluate.py                      # Generate figures and evaluation report
├── Towards Reproducible AI-Generated Scientific Visualizations_Final.docx
├── experiment/
│   ├── results.csv                     # 50,000 rows: seed, output_type, fidelity_score
│   ├── figures/                        # All paper figures (PNG)
│   └── model/                          # Trained model + all numeric outputs
└── README.md
```

> `experiment/images/` (~10 GB, 50,000 PNGs) and `sd15_model/` are excluded via `.gitignore`.  
> `results.csv` contains all data needed to reproduce the analysis without regenerating images.

---

## Key Results

| Metric | Value |
|---|---|
| Total images | 50,000 |
| Baseline FPAR (random seed) | 21.7% |
| Model-guided FPAR | 64.0% |
| Improvement | +42.3 percentage points |
| XGBoost AUC-ROC | 0.849 |
| Top feature | type_scatter_plot = 0.716 importance |

### Per-type FPAR

| Output Type | Baseline | Model-Guided | Improvement |
|---|---|---|---|
| Bar Chart | 7.2% | 39.0% | +31.7 pp |
| Line Chart | 71.6% | 71.6% | 0.0 pp |
| Scatter Plot | 9.9% | — | not achievable in SD 1.5 |
| Network Diagram | 1.4% | — | not achievable in SD 1.5 |
| Flowchart | 18.5% | 28.8% | +10.3 pp |

---

## Reproducing the Results

### Prerequisites

```bash
python -m venv venv
venv\Scripts\activate
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate opencv-python scikit-learn xgboost pandas matplotlib seaborn
```

Download model weights (not included in repo):
```bash
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 v1-5-pruned-emaonly.safetensors --local-dir sd15_model
```

### Run pipeline

```bash
python 00_test_setup.py           # Verify environment
python 01_generate_and_score.py   # Generate + score 50,000 images (resumable)
python 02_train_mapping_model.py  # Train and compare models
python 03_evaluate.py             # Generate figures and report
```

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Model | Stable Diffusion 1.5 |
| Scheduler | DDIM (deterministic) |
| Denoising steps | 50 |
| Guidance scale | 7.5 |
| Image resolution | 512 × 512 px |
| Seeds | 0 – 999 |
| Prompts per output type | 10 |
| Total images | 50,000 |
| Hardware | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM) |
| Framework | HuggingFace Diffusers 0.27 / PyTorch 2.1 / CUDA 12.1 |

---

## License

Code: MIT | Paper: CC BY 4.0

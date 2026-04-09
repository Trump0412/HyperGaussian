# HyperGaussian

**HyperGaussian** is a 4D dynamic scene reconstruction framework built on top of [4DGaussians](https://github.com/hustvl/4DGaussians), introducing a **spatiotemporal tube primitive** that couples each Gaussian's spatial covariance with an explicit temporal support region. This enables more faithful reconstruction of dynamic scenes compared to the vanilla deformation-field baseline.

On top of reconstruction, the codebase provides a full **semantic query pipeline** — leveraging Qwen-VL and Grounded SAM2 — for training-free, query-conditioned 4D entity localization and rendering.

---

## Key Features

- **StellarTube primitive**: per-Gaussian temporal extent (support interval) with covariance mixing along the tube axis
- **Temporal warp**: learnable monotone reparameterization of the scene timeline
- **Entity bank**: structured export of per-Gaussian temporal parameters and semantic clustering
- **Semantic query pipeline**: Qwen-VL query planning → Grounded SAM2 tracking → entity selection → query-conditioned rendering
- **Two benchmark protocols**: our 12-scene HyperNeRF benchmark and the 4DLangSplat-anno benchmark

---

## Results

### HyperNeRF — Our 12-Scene Benchmark (Reconstruction)

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Train Time |
|-------|--------|--------|---------|------------|
| keyboard | 28.41 | 0.8867 | 0.2072 | ~17 min |
| cut_lemon | 29.76 | 0.7545 | 0.3616 | ~17 min |
| torchchocolate | 27.27 | 0.8747 | 0.2486 | ~19 min |
| split_cookie | 31.10 | 0.9014 | 0.1802 | ~20 min |

### HyperNeRF — 4DLangSplat-anno Benchmark

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| americano | 28.78 | 0.8696 | 0.2253 |
| espresso | 24.96 | 0.8835 | 0.2222 |

### DyNeRF

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| coffee_martini | 10.82 | 0.329 | 0.649 |
| flame_steak | 14.38 | 0.460 | 0.609 |
| cook_spinach | 13.28 | 0.440 | 0.589 |
| sear_steak | 12.94 | 0.536 | 0.549 |
| cut_roasted_beef | 14.22 | 0.426 | 0.588 |
| flame_salmon | 8.20 | 0.281 | 0.627 |

---

## Repository Layout

```
HyperGaussian/
├── gaussian_stellar/          # Core library
│   ├── temporal/              # StellarTube temporal primitive
│   ├── entitybank/            # Entity bank export and clustering
│   └── semantics/             # Qwen-VL planner, Grounded-SAM2 backend
├── external/
│   ├── 4DGaussians/           # Vendored 4DGS base (train.py, render.py, ...)
│   ├── Grounded-SAM-2/        # Grounded SAM2 for query detection
│   └── gsplat/                # Gaussian rasterization
├── scripts/                   # Training, rendering, eval, pipeline scripts
├── configs/                   # Dataset argument files
├── patches/                   # Patches applied to external code
├── docs/                      # Design documents
├── data/                      # (ignored) datasets go here
├── runs/                      # (ignored) experiment outputs go here
└── reports/                   # (ignored) eval reports go here
```

---

## Environment Setup

**Requirements**: CUDA 12.1, Python 3.10

```bash
# 1. Create conda environment
bash scripts/setup_baseline_env.sh cuda121

# The environment is created at:
# /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310/

# 2. Verify installation
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
    python scripts/check_install.py
```

Key packages: `torch==2.1.2+cu121`, `diff_gaussian_rasterization`, `transformers==4.53.3`

For the semantic pipeline, a second environment for Grounded SAM2 is required:
```bash
bash scripts/setup_gsam2_env.sh
# Creates: /root/autodl-tmp/.conda-envs/grounded-sam2-py310/
```

For Qwen-VL inference:
```bash
# Model is expected at:
# /root/autodl-tmp/models/Qwen3-VL-8B-Instruct/
```

---

## Data Preparation

### HyperNeRF

Download the [HyperNeRF dataset](https://github.com/google/hypernerf) and place scenes under:
```
data/hypernerf/misc/      # misc split (americano, espresso, keyboard, ...)
data/hypernerf/interp/    # interp split (torchocolate, cut-lemon1, split-cookie, ...)
```

### DyNeRF (Neural 3D Video)

Download from [Neural 3D Video Synthesis](https://github.com/facebookresearch/Neural_3D_Video) and place under:
```
data/dynerf/coffee_martini/
data/dynerf/flame_steak/
...
```

---

## Reproducing Best Results

All best results use the **StellarTube** configuration. There are two hyperparameter presets:

**Preset A** — `bench12_best` (HyperNeRF keyboard, cut_lemon, split_cookie, espresso):
```
temporal_tube_span=0.40, temporal_tube_sigma=0.34, temporal_tube_covariance_mix=0.05
temporal_lr_init=0.00012, temporal_lr_final=0.000012
```

**Preset B** — `covp1_ready7` (HyperNeRF americano, torchchocolate; all DyNeRF):
```
temporal_tube_span=0.42, temporal_tube_sigma=0.30, temporal_tube_covariance_mix=0.06
temporal_lr_init=0.00012, temporal_lr_final=0.000012
```

### Training

```bash
# Set environment variables for the run
export GS_ROOT=/path/to/HyperGaussian
export PYTHONPATH=${GS_ROOT}:${GS_ROOT}/external/4DGaussians

# HyperNeRF — Preset A (keyboard, cut_lemon, espresso)
GS_RUN_NAMESPACE=my_run \
TEMPORAL_TUBE_SPAN=0.40 TEMPORAL_TUBE_SIGMA=0.34 TEMPORAL_TUBE_COV_MIX=0.05 \
TEMPORAL_LR_INIT=0.00012 TEMPORAL_LR_FINAL=0.000012 \
bash scripts/train_stellar_tube.sh hypernerf misc/keyboard

# HyperNeRF — Preset B (americano, torchchocolate)
GS_RUN_NAMESPACE=my_run \
TEMPORAL_TUBE_SPAN=0.42 TEMPORAL_TUBE_SIGMA=0.30 TEMPORAL_TUBE_COV_MIX=0.06 \
TEMPORAL_LR_INIT=0.00012 TEMPORAL_LR_FINAL=0.000012 \
bash scripts/train_stellar_tube.sh hypernerf misc/americano

# DyNeRF — Preset B
GS_RUN_NAMESPACE=my_run \
TEMPORAL_TUBE_SPAN=0.42 TEMPORAL_TUBE_SIGMA=0.30 TEMPORAL_TUBE_COV_MIX=0.06 \
TEMPORAL_LR_INIT=0.00012 TEMPORAL_LR_FINAL=0.000012 \
bash scripts/train_stellar_tube.sh dynerf coffee_martini
```

### Rendering & Evaluation

```bash
# Render test views
bash scripts/render_stellar_tube.sh hypernerf misc/keyboard

# Collect metrics (PSNR / SSIM / LPIPS)
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
    python scripts/collect_metrics.py \
    --run-dir runs/my_run/hypernerf/keyboard \
    --write-summary
```

Metrics are written to `runs/<run>/hypernerf/<scene>/metrics.json`.

---

## Semantic Query Pipeline

After training, the full semantic pipeline can be run on any scene:

```bash
# Step 1: Export entity bank from trained checkpoint
bash scripts/export_entitybank.sh hypernerf misc/keyboard --run-namespace my_run

# Step 2–7: Run full query pipeline (query planning → SAM2 → entity selection → render)
# Requires: Qwen3-VL-8B-Instruct + Grounded-SAM2 environments
bash scripts/run_query_guided_full.sh \
    --run-dir runs/my_run/hypernerf/keyboard \
    --query "正在键盘上打字的左手"

# Evaluate against ground truth annotations
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
    python scripts/eval_query_guided.py \
    --benchmark data/benchmarks/Ours_benchmark.json \
    --results-dir reports/my_eval/
```

### Benchmark Annotations

- **Our 12-scene benchmark**: `data/benchmarks/Ours_benchmark.json`
- **4DLangSplat-anno**: `external/benchmarks/4dlangsplat/HyperNeRF-Annotation/`

---

## 4DGS Baseline

To reproduce the 4DGS baseline:

```bash
bash scripts/train_baseline.sh hypernerf misc/keyboard
bash scripts/eval_baseline.sh hypernerf misc/keyboard
```

---

## Citation

```bibtex
@article{hypergaussian2026,
  title     = {HyperGaussian: ...},
  author    = {...},
  journal   = {...},
  year      = {2026}
}
```

---

## Acknowledgements

This codebase builds on [4DGaussians](https://github.com/hustvl/4DGaussians), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), and [gsplat](https://github.com/nerfstudio-project/gsplat).

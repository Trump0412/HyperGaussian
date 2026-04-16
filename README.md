# HyperGaussian: Referring 4D Gaussian Splatting

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](#environment-setup)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-orange.svg)](#environment-setup)

[Project Page](https://trump0412.github.io/HyperGaussian/) | [Paper](#citation) | [Code](https://github.com/Trump0412/HyperGaussian)

</div>

HyperGaussian is a unified framework for dynamic 4D Gaussian reconstruction and training-free referring understanding in complex dynamic scenes. It targets **Referring 4D Gaussian Splatting (R4DGS)**, where natural-language queries may be temporally varying, multi-target, or involve distractors and zero-target conditions.

## Abstract

Dynamic 4D Gaussian representations have shown strong capability in reconstruction and rendering, yet they remain insufficient for complex natural-language referring understanding in dynamic scenes. In this paper, we study Referring 4D Gaussian Splatting (R4DGS), a task that targets realistic 4D referring understanding under temporally varying queries, multi-target and reasoning-intensive queries, and zero-target or distractor queries. We introduce R4D-Bench-QA, a benchmark with structured query annotations, and present HyperGaussian, a unified framework that couples generalized dynamic Gaussian reconstruction, entity-centric scene structuring, and training-free referring inference. HyperGaussian augments the dynamic Gaussian representation with explicit temporal support, organizes reconstructed Gaussians into a shared entity bank, uses Multimodal Large Language Models (MLLMs) to decompose complex queries into trackable object phrases and referring constraints, and performs query-conditioned grounding without retraining the underlying scene representation.

## Environment Setup

**Requirements:** CUDA 12.1, Miniconda

```bash
git clone https://github.com/Trump0412/HyperGaussian.git --recursive
cd HyperGaussian

# Main training/rendering/evaluation environment
bash scripts/setup_baseline_env.sh cuda121

# Verify installation
bash scripts/check_install.py
```

For the semantic pipeline (Grounded-SAM2):

```bash
bash scripts/setup_grounded_sam2.sh
```

Environments are installed under `/root/autodl-tmp/.conda-envs/` by default. Override with `GS4D_ENV_ROOT`.

## Dataset Setup

### D-NeRF

```bash
# Download a single scene (default: bouncingballs)
bash scripts/prepare_dnerf.sh --scene mutant

# Download all 8 scenes
bash scripts/prepare_dnerf.sh --all
```

Expected layout: `data/dnerf/<scene>/`

### HyperNeRF

```bash
# Download a scene (default: broom2 from virg split)
HYPERNERF_GROUP=misc HYPERNERF_SCENE=keyboard HYPERNERF_ASSET=misc_keyboard.zip \
  bash scripts/prepare_hypernerf.sh
```

Expected layout: `data/hypernerf/<group>/<scene>/`

To register a locally available HyperNeRF scene:

```bash
bash scripts/prepare_local_hypernerf_scene.sh /path/to/scene misc keyboard
```

### 4DLangSplat Annotations (for referring evaluation)

```bash
bash scripts/download_4dlangsplat_annotations.sh rpzhou/HyperNeRF-Annotation \
  data/benchmarks/4dlangsplat/HyperNeRF-Annotation
```

## Training

```bash
# Train on a D-NeRF scene
bash scripts/train.sh dnerf mutant

# Train on a HyperNeRF scene
bash scripts/train.sh hypernerf misc/keyboard
```

Key environment variables (all have sensible defaults):

| Variable | Default | Description |
|---|---|---|
| `GS_RUN_NAMESPACE` | `hypergaussian` | Output directory prefix under `runs/` |
| `GS_PORT` | `6021` | Port for the training viewer |
| `TEMPORAL_TUBE_SAMPLES` | `5` | Temporal tube integration samples |
| `TEMPORAL_TUBE_SPAN` | `1.0` | Temporal tube span |
| `TEMPORAL_TUBE_SIGMA` | `0.75` | Temporal tube bandwidth |

## Evaluation

```bash
# Render and compute metrics for a trained scene
bash scripts/eval.sh dnerf mutant
bash scripts/eval.sh hypernerf misc/keyboard
```

Metrics (PSNR, SSIM, LPIPS) are written to `runs/<namespace>/<dataset>/<scene>/metrics.log` and summarized in `summary.json`.

## Referring Evaluation

### 4DLangSplat Public Protocol

```bash
bash scripts/run_public_query_protocol.sh \
  data/benchmarks/4dlangsplat/HyperNeRF-Annotation/protocol.json \
  runs/hypergaussian \
  data/hypernerf

python scripts/evaluate_public_query_protocol.py \
  --protocol-json data/benchmarks/4dlangsplat/HyperNeRF-Annotation/protocol.json \
  --annotation-dir data/benchmarks/4dlangsplat/HyperNeRF-Annotation \
  --dataset-dir data/hypernerf \
  --query-root runs/hypergaussian \
  --output-json reports/public_eval.json \
  --output-md reports/public_eval.md
```

### R4D-Bench-QA

```bash
python scripts/evaluate_ours_benchmark.py \
  --benchmark data/benchmarks/Ours_benchmark.json \
  --query-root-map configs/query_root_map.json \
  --dataset-dir-map configs/dataset_dir_map.json \
  --output-json reports/r4d_bench_eval.json \
  --output-md reports/r4d_bench_eval.md
```

## Reproducing the Keyboard Result

The following configuration reproduces the reported reconstruction result on the HyperNeRF `keyboard` scene:

```bash
export GS_RUN_NAMESPACE=hypergaussian_keyboard_repro
export TEMPORAL_TUBE_SAMPLES=3
export TEMPORAL_TUBE_SPAN=0.40
export TEMPORAL_TUBE_SIGMA=0.34
export TEMPORAL_TUBE_WEIGHT_POWER=1.0
export TEMPORAL_TUBE_COVARIANCE_MIX=0.05
export TEMPORAL_DRIFT_SCALE=1.0
export TEMPORAL_GATE_MIX=1.0
export TEMPORAL_DRIFT_MIX=1.0
export TEMPORAL_ACCELERATION_ENABLED=0
export TEMPORAL_LR_INIT=0.00012
export TEMPORAL_LR_FINAL=0.000012

bash scripts/train.sh hypernerf misc/keyboard
bash scripts/eval.sh hypernerf misc/keyboard
```

## Repository Layout

```
HyperGaussian/
├── hypergaussian/         # core library
│   ├── temporal/          # temporal warp modules
│   ├── entitybank/        # entity-centric scene organization
│   └── semantics/         # query planning, grounding, and scoring
├── scripts/               # training, evaluation, and data preparation
├── configs/               # scene and benchmark configurations
├── external/              # 4DGaussians submodule
├── data/                  # datasets (not tracked)
├── runs/                  # experiment outputs (not tracked)
└── docs/                  # project page
```

## Citation

```bibtex
@article{hypergaussian2026,
  title   = {HyperGaussian: Referring 4D Gaussian Splatting},
  author  = {Anonymous},
  journal = {arXiv},
  year    = {2026}
}
```

## Acknowledgements

This project builds on [4DGaussians](https://github.com/hustvl/4DGaussians), [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2), and related dynamic scene understanding toolchains.

# HyperGaussian: Referring 4D Gaussian Splatting

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-orange.svg)](#installation)

</div>

HyperGaussian is a unified framework for **dynamic 4D Gaussian reconstruction** and **training-free referring understanding** in complex dynamic scenes.

It targets Referring 4D Gaussian Splatting (R4DGS), where queries may be:
- temporally varying,
- multi-target and reasoning-intensive,
- or distractor/zero-target.

## Paper

**Title:** *HyperGaussian: Referring 4D Gaussian Splatting*  
**Contact:** `2286793492@qq.com`

### Abstract

Dynamic 4D Gaussian representations have shown strong capability in reconstruction and rendering, yet they remain insufficient for complex natural-language referring understanding in dynamic scenes. Existing 4D Gaussian methods are primarily designed for rendering-oriented dynamic modeling, while current semantic extensions often rely on scene-specific semantic optimization or are evaluated on relatively simple query settings. In this paper, we study Referring 4D Gaussian Splatting (R4DGS), a task that targets realistic 4D referring understanding under temporally varying queries, multi-target and reasoning-intensive queries, and zero-target or distractor queries. To support this task, we introduce R4D-Bench-QA, a benchmark with structured query annotations. We further present HyperGaussian, a unified framework that couples generalized dynamic Gaussian reconstruction, entity-centric scene structuring, and training-free referring inference. HyperGaussian augments the dynamic Gaussian representation with explicit temporal support, organizes reconstructed Gaussians into a shared entitybank, uses Multimodal Large Language Models (MLLMs) to decompose complex queries into trackable object phrases and referring constraints, and performs query-conditioned grounding without retraining the underlying scene representation. Experiments on R4D-Bench-QA and public benchmarks show that HyperGaussian preserves competitive reconstruction quality relative to a 4DGS baseline and improves grounding under complex spatiotemporal expressions.

## Highlights

- 🎯 **Referring-first 4DGS formulation** for realistic dynamic query understanding.
- 🧠 **Entity-centric scene organization** with a shared entity bank.
- 🧪 **Training-free referring inference** on top of reconstructed dynamic scenes.
- 📊 **Benchmark-ready evaluation** for both in-house R4D-Bench-QA and public protocols.

## Quick Results

### Reconstruction (Keyboard Example)

- PSNR: **28.4051**
- SSIM: **0.8867**
- LPIPS-vgg: **0.2072**

### Referring Benchmarks

- **R4D-Bench-QA (snapshot):**
  - Acc: **97.4990%**
  - vIoU: **8.1029%**
  - tIoU: **97.4908%**
- **4DLangSplat (Americano):**
  - Acc: **97.72%**
  - vIoU: **69.35%**
  - temporal tIoU: **94.34%**

## Installation

### 1) Main environment (training / rendering / evaluation)

```bash
bash scripts/setup_baseline_env.sh cuda121

conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python scripts/check_install.py
```

### 2) Grounded-SAM2 environment (semantic pipeline)

```bash
bash scripts/setup_grounded_sam2.sh
```

## Data Preparation

Expected layout:

```text
data/hypernerf/misc/keyboard
data/hypernerf/misc/americano
data/hypernerf/interp/cut-lemon1
data/dynerf/<scene>
```

To register a local HyperNeRF scene into this repository:

```bash
bash scripts/prepare_local_hypernerf_scene.sh /abs/path/to/keyboard misc keyboard
```

### Public 4DLangSplat Annotations

```bash
bash scripts/download_4dlangsplat_annotations.sh rpzhou/HyperNeRF-Annotation \
  data/benchmarks/4dlangsplat/HyperNeRF-Annotation
```

### R4D-Bench-QA (planned release)

- Hugging Face placeholder:
  `https://huggingface.co/datasets/<ORG>/HyperGaussian-R4D-Bench-QA`

## Reproduce Keyboard (PSNR 28.4051)

```bash
export GS_RUN_NAMESPACE=stellar_tube_bench12_best04034_lrlow_20260402_keyboard
export TEMPORAL_TUBE_SAMPLES=3
export TEMPORAL_TUBE_SPAN=0.40
export TEMPORAL_TUBE_SIGMA=0.34
export TEMPORAL_TUBE_WEIGHT_POWER=1.0
export TEMPORAL_TUBE_COVARIANCE_MIX=0.05
export TEMPORAL_DRIFT_SCALE=1.0
export TEMPORAL_GATE_MIX=1.0
export TEMPORAL_DRIFT_MIX=1.0
export TEMPORAL_ACCELERATION_ENABLED=0
export TEMPORAL_VELOCITY_REG_WEIGHT=0.0
export TEMPORAL_ACCELERATION_REG_WEIGHT=0.0
export TEMPORAL_LR_INIT=0.00012
export TEMPORAL_LR_FINAL=0.000012
export TEMPORAL_LR_DELAY_MULT=0.01

bash scripts/train_stellar_tube.sh hypernerf misc/keyboard \
  --iterations 14000 \
  --coarse_iterations 3000 \
  --test_iterations 3000 7000 14000 \
  --save_iterations 7000 14000 \
  --checkpoint_iterations 7000 14000 \
  --seed 3407

bash scripts/eval_stellar_tube.sh hypernerf misc/keyboard
```

## Evaluate Referring Benchmarks

### R4D-Bench-QA

```bash
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python scripts/evaluate_ours_benchmark.py \
    --benchmark data/benchmarks/Ours_benchmark.json \
    --query-root-map /abs/path/to/query_root_map.json \
    --dataset-dir-map /abs/path/to/dataset_dir_map.json \
    --output-json reports/ours_benchmark_eval.json \
    --output-md reports/ours_benchmark_eval.md \
    --skip-missing
```

### 4DLangSplat Public Protocol

```bash
bash scripts/run_public_query_protocol.sh \
  /abs/path/to/protocol.json \
  /abs/path/to/run_dir \
  /abs/path/to/dataset_dir

conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python scripts/evaluate_public_query_protocol.py \
    --protocol-json /abs/path/to/protocol.json \
    --annotation-dir /abs/path/to/annotation_dir \
    --dataset-dir /abs/path/to/dataset_dir \
    --query-root /abs/path/to/query_root \
    --output-json reports/public_eval.json \
    --output-md reports/public_eval.md
```

## Repository Layout

```text
HyperGaussian/
├── hypergaussian/             # core package (temporal, entity bank, semantics)
├── scripts/                   # training, rendering, evaluation, benchmark tools
├── configs/                   # model/benchmark configurations
├── docs/                      # project page, reproducibility, release docs
├── patches/                   # dependency patch notes
├── data/                      # datasets (ignored)
├── runs/                      # experiment outputs (ignored)
└── reports/                   # evaluation outputs (ignored)
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

This project builds on excellent open-source foundations including 4DGaussians, Grounded-SAM2, and related dynamic scene understanding toolchains.

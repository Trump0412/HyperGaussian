# HyperGaussian: Referring 4D Gaussian Splatting

HyperGaussian（内部代号 **GaussianStellar**）是一个面向动态场景重建与语义检索的 4D Gaussian 框架：

- 重建侧：在 4DGaussians 上引入时空管状建模（StellarTube / WorldTube）。
- 语义侧：支持 query-guided 的实体选择与时序渲染评测。
- Benchmark：支持我们自有 benchmark 与 4DLangSplat 公共协议（含 `americano`）。

## Paper

- Title: `HyperGaussian: Referring 4D Gaussian Splatting`
- Contact: `2286793492@qq.com`

### Abstract

Dynamic 4D Gaussian representations have shown strong capability in reconstruction and rendering, yet they remain insufficient for complex natural-language referring understanding in dynamic scenes. Existing 4D Gaussian methods are primarily designed for rendering-oriented dynamic modeling, while current semantic extensions often rely on scene-specific semantic optimization or are evaluated on relatively simple query settings. In this paper, we study Referring 4D Gaussian Splatting (R4DGS), a task that targets realistic 4D referring understanding under temporally varying queries, multi-target and reasoning-intensive queries, and zero-target or distractor queries. To support this task, we introduce R4D-Bench-QA, a benchmark with structured query annotations. We further present HyperGaussian, a unified framework that couples generalized dynamic Gaussian reconstruction, entity-centric scene structuring, and training-free referring inference. HyperGaussian augments the dynamic Gaussian representation with explicit temporal support, organizes reconstructed Gaussians into a shared entitybank, uses Multimodal Large Language Models (MLLMs) to decompose complex queries into trackable object phrases and referring constraints, and performs query-conditioned grounding without retraining the underlying scene representation. Experiments on R4D-Bench-QA and public benchmarks show that HyperGaussian preserves competitive reconstruction quality relative to a 4DGS baseline and improves grounding under complex spatiotemporal expressions.

## 当前开源状态（2026-04-16）

- 代码主干与核心脚本可用。
- 已完成一轮复现核验（见 `docs/reproducibility_20260416.md`）。
- `keyboard` 重建示例已核验到 `PSNR=28.4050827`（四舍五入为 **28.4051**）。

发布前仍建议确认两件事：

1. 许可证（`LICENSE`）最终版本。
2. 清理/归档当前仓库中的历史实验草稿脚本（避免把一次性脚本作为对外接口）。

## 目录结构

```text
HyperGaussian/
├── gaussian_stellar/          # 核心库
├── external/                  # 第三方依赖（4DGaussians / Grounded-SAM-2 / gsplat 等）
├── scripts/                   # 训练、评测、语义 pipeline 脚本
├── configs/                   # 参数与 benchmark 配置
├── docs/                      # 设计文档、复现记录、GitHub Pages 首页
├── data/                      # 数据集（默认不入库）
├── runs/                      # 训练输出（默认不入库）
├── reports/                   # 评测输出（默认不入库）
└── patches/                   # 外部依赖补丁说明
```

## 1. 环境准备

推荐 CUDA 12.1 + Python 3.10。

```bash
# 主环境（训练/渲染/重建评测）
bash scripts/setup_baseline_env.sh cuda121

# 检查安装
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python scripts/check_install.py

# 语义环境（Grounded-SAM-2 / Qwen 相关流程）
bash scripts/setup_grounded_sam2.sh
```

## 2. 数据与模型下载

### 2.1 HyperNeRF / DyNeRF 数据

将数据放到如下目录（可用软链）：

```text
data/hypernerf/misc/keyboard
data/hypernerf/misc/americano
data/hypernerf/interp/cut-lemon1
data/dynerf/...
```

可用脚本把本地已有 HyperNeRF 场景链接进仓库：

```bash
bash scripts/prepare_local_hypernerf_scene.sh /abs/path/to/keyboard misc keyboard
```

### 2.2 4DLangSplat 注释

```bash
bash scripts/download_4dlangsplat_annotations.sh rpzhou/HyperNeRF-Annotation \
  data/benchmarks/4dlangsplat/HyperNeRF-Annotation
```

### 2.3 我们自有 benchmark（占位链接）

`README` 先预留对外地址：

- HuggingFace dataset（待公开）：`https://huggingface.co/datasets/<ORG>/GaussianStellar-OursBench`

默认约定路径：

```text
data/benchmarks/Ours_benchmark.json
```

### 2.4 语义模型

- Qwen3-VL-8B-Instruct（示例路径）：`/root/autodl-tmp/models/Qwen3-VL-8B-Instruct`
- Grounded DINO / SAM2 会在 `setup_grounded_sam2.sh` 与相关脚本中自动拉取。

## 3. 固定参数复现：keyboard 重建（目标 PSNR=28.4051）

### 3.1 固定训练配置

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
```

训练 + 评测：

```bash
bash scripts/train_stellar_tube.sh hypernerf misc/keyboard \
  --iterations 14000 \
  --coarse_iterations 3000 \
  --test_iterations 3000 7000 14000 \
  --save_iterations 7000 14000 \
  --checkpoint_iterations 7000 14000 \
  --seed 3407

bash scripts/eval_stellar_tube.sh hypernerf misc/keyboard

conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python scripts/collect_metrics.py \
    --run-dir runs/${GS_RUN_NAMESPACE}/hypernerf/keyboard \
    --write-summary
```

目标结果（本机复现记录）：

- `PSNR=28.4050827`（四舍五入 `28.4051`）
- `SSIM=0.8866783`
- `LPIPS-vgg=0.2071507`

## 4. 语义 Benchmark 复现

### 4.1 我们自有 benchmark（OursBench）

先准备 query 输出目录映射（`query_id -> query_root`）与数据目录映射（`query_id -> dataset_dir`），然后运行：

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

发布版历史结果（snapshot）示例：

- `Acc=97.4990%`
- `vIoU=8.1029%`
- `tIoU=97.4908%`

### 4.2 4DLangSplat（Americano）

```bash
# 先跑 query pipeline（如果还没生成 query_root）
bash scripts/run_public_query_protocol.sh \
  /abs/path/to/protocol_splits/americano.json \
  /abs/path/to/run_dir \
  /abs/path/to/dataset_dir

# 再评测
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python scripts/evaluate_public_query_protocol.py \
    --protocol-json /abs/path/to/protocol_splits/americano.json \
    --annotation-dir /abs/path/to/HyperNeRF-Annotation/americano \
    --dataset-dir /abs/path/to/hypernerf/misc/americano \
    --query-root /abs/path/to/query_root \
    --output-json reports/4dlangsplat_americano_eval.json \
    --output-md reports/4dlangsplat_americano_eval.md
```

本机最新复现结果（2026-04-16）：

- `Acc=97.72%`
- `vIoU=69.35%`
- `temporal tIoU=94.34%`

## 5. 复现记录与证据

- 复现报告：`docs/reproducibility_20260416.md`
- 开源就绪审计：`docs/open_source_readiness_20260416.md`

## 6. GitHub Pages 首页模板

已提供一个可直接改链接的首页模板：

- `docs/index.html`
- `docs/assets/githubio.css`

用于在首页展示：

- GitHub 代码地址
- HG（项目主页/演示）地址
- arXiv 论文地址

## 7. Citation（占位）

```bibtex
@article{hypergaussian2026,
  title   = {HyperGaussian: Referring 4D Gaussian Splatting},
  author  = {Anonymous},
  journal = {arXiv},
  year    = {2026}
}
```

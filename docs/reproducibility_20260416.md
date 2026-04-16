# Reproducibility Record (2026-04-16)

## 执行环境

- Host: `myautodl`
- Project: `/root/autodl-tmp/HyperGaussian`
- Main env: `/root/autodl-tmp/.conda-envs/gs4d-cuda121-py310`
- Repro output dir: `/root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416`

## A. keyboard 重建指标复核（目标 PSNR 28.4051）

使用已完成训练的固定配置 run：

- Run dir:
  `/root/autodl-tmp/GaussianStellar/runs/stellar_tube_bench12_best04034_lrlow_20260402_keyboard/hypernerf/keyboard`

执行：

```bash
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python /root/autodl-tmp/HyperGaussian/scripts/collect_metrics.py \
  --run-dir /root/autodl-tmp/GaussianStellar/runs/stellar_tube_bench12_best04034_lrlow_20260402_keyboard/hypernerf/keyboard \
  --write-summary
```

结果：

- PSNR: `28.40508270263672`（四舍五入 `28.4051`）
- SSIM: `0.8866782784461975`
- LPIPS-vgg: `0.20715069770812988`

产物：

- `reports/reproducibility_20260416/keyboard_metrics.json`
- `reports/reproducibility_20260416/keyboard_summary.md`

## B. OursBench（语义 benchmark）

### B1. 发布历史 snapshot（建议作为论文/主页口径）

来源文件：

- `/root/autodl-tmp/GaussianStellar/reports/ours_benchmark_eval/results_latest.json`
- `/root/autodl-tmp/GaussianStellar/reports/ours_benchmark_eval/results_latest.md`

摘要：

- Valid queries: `18 / 36`
- Acc: `97.4990%`
- vIoU: `8.1029%`
- tIoU: `97.4908%`

### B2. 使用当前评测脚本重跑（2026-04-16 口径）

执行：

```bash
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python /root/autodl-tmp/HyperGaussian/scripts/evaluate_ours_benchmark.py \
  --benchmark /root/autodl-tmp/data/Ours_benchmark.json \
  --query-root-map /root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416/query_root_map_from_results_latest.json \
  --dataset-dir-map /root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416/dataset_dir_map_from_results_latest.json \
  --output-json /root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416/ours_benchmark_eval_reproduced_from_latest.json \
  --output-md /root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416/ours_benchmark_eval_reproduced_from_latest.md \
  --skip-missing
```

当前脚本口径下结果：

- Valid queries: `15 / 36`
- Acc: `98.0296%`
- vIoU: `5.0668%`
- tIoU: `48.7067%`

说明：历史 snapshot 与当前脚本口径存在差异，建议发布时固定一套官方口径。

## C. 4DLangSplat (Americano) 复现

执行：

```bash
conda run -p /root/autodl-tmp/.conda-envs/gs4d-cuda121-py310 \
  python /root/autodl-tmp/HyperGaussian/scripts/evaluate_public_query_protocol.py \
  --protocol-json /root/autodl-tmp/GaussianStellar/reports/4dlangsplat_compare/protocol_splits/americano.json \
  --annotation-dir /root/autodl-tmp/GaussianStellar/data/benchmarks/4dlangsplat/HyperNeRF-Annotation/americano \
  --dataset-dir /root/autodl-tmp/GaussianStellar/data/hypernerf/misc/americano \
  --query-root /root/autodl-tmp/GaussianStellar/runs/stellar_tube_4dlangsplat_refresh_20260328_americano/hypernerf/americano/entitybank/query_guided \
  --output-json /root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416/4dlangsplat_americano_public_eval_reproduced.json \
  --output-md /root/autodl-tmp/HyperGaussian/reports/reproducibility_20260416/4dlangsplat_americano_public_eval_reproduced.md
```

结果：

- Queries: `3`
- Acc: `97.72%`
- vIoU: `69.35%`
- temporal tIoU: `94.34%`

## D. 备注

- `reports/` 默认在 `.gitignore` 中，不会随代码仓库提交。
- 若要把本次复现实验摘要随代码发布，建议把关键结论同步到 `docs/`（本文件已同步）。


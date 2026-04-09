# Runbook And Pipeline

## 1. 统一 pipeline

项目的标准流程是：

`dataset -> train -> eval -> export_entitybank -> export semantic artifacts -> query pipeline -> application bundle`

每个阶段的核心输出如下。

| 阶段 | 主要输出 |
| --- | --- |
| 训练 | `run_dir/config.yaml`、`point_cloud/iteration_*`、`temporal_warp/` |
| 评测 | `test/`、指标汇总、render log |
| entitybank 导出 | `trajectory_samples.npz`、`tube_bank.json`、`cluster_stats.json`、`entities.json` |
| 语义导出 | `semantic_slots.json`、`semantic_tracks.json`、`semantic_priors.json`、`native_semantic_assignments.json` |
| query pipeline | `query_plan.json`、`grounded_sam2_query_tracks.json`、`proposal_dir/`、`query_entitybank/`、`selected_query_qwen.json`、`final_query_render_sourcebg/` |
| 应用 bundle | subset run、filtered ply、removal bundle、诊断视频/图像 |

## 2. 重建主线运行

### 2.1 baseline 4DGS

```bash
bash scripts/train_baseline.sh dnerf bouncingballs
bash scripts/eval_baseline.sh dnerf bouncingballs
```

### 2.2 weak tube 分支

如果当前准备复现实验中“weaktube 优于 4DGS”的重建主结果，建议从这里开始：

```bash
bash scripts/train_stellar_tube.sh dnerf mutant
bash scripts/eval_stellar_tube.sh dnerf mutant
```

### 2.3 explicit worldtube 分支

```bash
bash scripts/train_stellar_worldtube.sh dnerf mutant
bash scripts/eval_stellar_worldtube.sh dnerf mutant
```

### 2.4 chrono / earlier branches

```bash
bash scripts/train_chrono.sh dnerf mutant
bash scripts/eval_chrono.sh dnerf mutant
```

## 3. entitybank 与语义导出

训练完成后，统一先导出 entitybank：

```bash
python scripts/export_entitybank.py --run-dir runs/stellar_tube/.../scene_name
```

然后继续导出语义接口：

```bash
python scripts/export_semantic_slots.py --run-dir runs/.../scene_name
python scripts/export_semantic_tracks.py --run-dir runs/.../scene_name
python scripts/export_semantic_priors.py --run-dir runs/.../scene_name
python scripts/export_native_semantics.py --run-dir runs/.../scene_name
```

这一步不重新训练 scene representation，只是消费 `run_dir` 中已有的重建结果。

## 4. Query-specific grounding pipeline

单个 query 的统一脚本入口是：

```bash
bash scripts/run_query_specific_worldtube_pipeline.sh \
  <run_dir> \
  <dataset_dir> \
  "<query_text>" \
  <query_name>
```

这个脚本内部会按顺序执行：

1. `plan_query_entities.py`
2. `run_grounded_sam2_query.py`
3. `build_query_proposal_dir.py`
4. `export_entitybank.py --proposal-dir ...`
5. `export_semantic_slots.py`
6. `export_semantic_tracks.py`
7. `export_semantic_priors.py`
8. `export_native_semantics.py`
9. `export_qwen_semantics.py`
10. `select_qwen_query_entities.py`
11. `render_query_video.py`

关键点是：

- 底层场景表示不重训。
- query-specific 阶段只是从已有 worldtubes 里做重组和筛选。

## 5. 公共 query benchmark

如果要批量跑 protocol：

```bash
bash scripts/run_public_query_protocol.sh <protocol_json> <run_dir> <dataset_dir>
```

如果要跑仓库内的 GR4D curated 语义 benchmark：

```bash
bash scripts/run_gr4d_curated_semantic_benchmark.sh
```

## 6. 应用侧实体剔除

应用侧当前最直接的入口是：

```bash
bash scripts/run_scene_deepfill_removal_experiment.sh split-cookie
```

或：

```bash
bash scripts/run_scene_deepfill_removal_experiment.sh cut-lemon1
```

这条线依赖：

- 已有 `query_root`
- `selected_query_qwen.json` 或 proposal alias
- subset Gaussian filtering
- removal bundle 导出

本质上它仍然是沿用同一个 query-conditioned entity selection。

## 7. 结果检查顺序

对于一个完整 query run，建议按下面顺序排查：

1. `query_plan.json`
2. `grounded_sam2/grounded_sam2_query_tracks.json`
3. `proposal_dir/query_proposal_summary.json`
4. `query_entitybank/entities.json`
5. `query_worldtube_run/entitybank/semantic_assignments_qwen.json`
6. `query_worldtube_run/entitybank/selected_query_qwen.json`
7. `final_query_render_sourcebg/validation.json`
8. `diagnostics/`

## 8. 当前建议的统一实验口径

如果是准备论文与开源说明，建议把任务口径固定成下面三条：

- 重建主线
  - `stellar_tube` 作为当前强重建分支。
- 语义 grounding 主线
  - `entitybank + semantic priors + query-specific reassignment`。
- 应用主线
  - 在 query-specific entity selection 上做实体剔除/编辑。

这样既保留现在的 strongest result，也不破坏整个系统的统一性。


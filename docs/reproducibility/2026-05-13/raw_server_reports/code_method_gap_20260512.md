# ReferGaussian Repro Audit: Method-Code Gaps (2026-05-12)

## Scope
- Codebase: `/root/autodl-tmp/HyperGaussian` (latest pushed commit `351686f`)
- Reference paper: `/home/chenbp/RefGaussian (6).pdf`
- Server assets: code/data/runs under `/root/autodl-tmp/GaussianStellar` and `/root/autodl-tmp/data`

## Confirmed Gaps / Risks

1. **Benchmark scale mismatch vs paper claim**
- Paper states R4D-Bench-QA has **266 sentence-level queries** (12 scenes).
- Server-available benchmark file (`/root/autodl-tmp/data/Ours_benchmark_v2.json`) contains **36 queries** only.
- This blocks exact 266-query full reproduction on current server snapshot.

2. **R4D benchmark download path inconsistency (fixed in code)**
- README points to HuggingFace dataset `LiYacheng/r4d-bench-qa`.
- Original `scripts/download_r4d_bench_qa.sh` defaulted to another repo id.
- Fixed to default to `LiYacheng/r4d-bench-qa` and support `R4D_BENCH_REPO_ID` override.

3. **Pipeline benchmark path propagation bug (fixed in code)**
- `scripts/run_ours_benchmark_query_pipeline.sh` called `prepare_ours_benchmark_queries.py` without passing chosen benchmark path.
- If default `data/benchmarks/r4d_bench_qa/benchmark.json` is absent, pipeline fails even when benchmark path is provided via CLI.
- Fixed by passing `OURS_BENCHMARK_JSON` and `OURS_BENCHMARK_REPORT_DIR` to prep script.

4. **Asset-root coupling bug for split workspaces (fixed in code)**
- Pipeline assumed runs/data under `GS_ROOT`.
- On this server, code is in `HyperGaussian`, while run/data assets are in `GaussianStellar`.
- Added fallback root detection to auto-switch to `../GaussianStellar/{runs,data}` when needed.

5. **Query prep compatibility issue (fixed in code)**
- `prepare_ours_benchmark_queries.py` previously failed/under-filled on non-list query payloads and alternate file names.
- Added support for dict payload formats and additional query keys/files (`curated_queries.json`, `question`, `query`).

6. **Cook-spinach query-text missing in benchmark artifact (data issue)**
- `cook-spinach_q1/q4/q5` had empty `question` in `Ours_benchmark_v2.json`.
- Recovered text from `/root/autodl-tmp/GR4D-Bench/data/scenes/cook_spinach/cook-spinach_queries.json`.

7. **Semantic export dependence on `run_dir/test` renders (fixed in code; requires rerun)**
- Original logic in `refergaussian/semantics/query_render.py::_find_render_dir` failed when `query_worldtube_run/test` is a broken symlink (common for DyNeRF runs without pre-rendered test frames).
- Added fallback logic:
  - remove broken `test` symlink if needed
  - fallback to `source_path/cam*/images` via `test/ours_fallback_source/renders`
- This removes the prior hard failure in `export_qwen_semantics.py` and allows semantic export to proceed.

8. **Flame_salmon_q5 output layout inconsistency (partially patched via mapping)**
- For `flame_salmon_q5`, valid `validation.json` exists under `entity_library_qwen_sourcebg/.../rendered_source/` but not under canonical `final_query_render_sourcebg/validation.json` in the mapped root.
- Evaluator map was patched to this query root and `validation.json` symlinked for evaluation.

## Current Repro Status (R4D 36-query artifact)
- Best current eval file: `reports/repro_check_20260512/r4d_bench_eval_recheck_36_autofix_plus_flameq5_mapped.json`
- Valid queries: **33 / 36**
- Missing: `cook-spinach_q1`, `cook-spinach_q4`, `cook-spinach_q5`

## Current Repro Status (4D LangSplat public protocol)
- Reproduced scene-level reports stored in:
  - `reports/repro_check_20260512/public_protocol/`

## Reconstruction Metrics
- 12-scene summary:
  - `reports/repro_check_20260512/recon_metrics/recon_metrics_summary.json`

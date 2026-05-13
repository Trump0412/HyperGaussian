# README Repro Audit (2026-05-12)

## Scope
- Repo: `/root/autodl-tmp/HyperGaussian`
- Checked against current runtime assets in `/root/autodl-tmp/GaussianStellar`
- Dataset source used in this run: `LiYacheng/r4d-bench-qa`

## Key Findings

1. **README benchmark path mismatch (fixed)**
- README originally used:
  - `data/benchmarks/r4d_bench_qa/benchmark.json`
- HF dataset snapshot actually stores dense-GT files under:
  - `scripts/new_predictions_ground_truth_final.json` (36)
  - `scripts/new_predictions_ground_truth_all_queries.json` (89)

2. **Download script lacked canonical aliases (fixed)**
- `scripts/download_r4d_bench_qa.sh` now creates:
  - `benchmark.json` -> dense GT 36 entry
  - `benchmark_all_queries.json` -> dense GT 89 entry
- Manifest now records resolved benchmark entries.

3. **Pipeline script benchmark resolution was brittle (fixed)**
- `scripts/run_ours_benchmark_query_pipeline.sh` now accepts:
  - benchmark JSON path, or
  - benchmark directory root
- It auto-resolves known benchmark entry files.

4. **Cook-spinach scene prefix compatibility (fixed)**
- Added both `cook-spinach_*` and `cook_spinach_*` support in scene-key parser.

5. **README updated with explicit 36/89 commands (fixed)**
- Added separate command blocks for 36-query and 89-query evaluation.
- Added HF mirror usage note.

## Verification

- Syntax checks passed:
  - `bash -n scripts/download_r4d_bench_qa.sh`
  - `bash -n scripts/run_ours_benchmark_query_pipeline.sh`
- Path-resolution dry run passed:
  - command used directory input: `/root/autodl-tmp/r4d-bench-qa-mirror-meta`
  - resolved benchmark logged at:
    - `/tmp/readme_repro_pathcheck.log`
  - key lines:
    - `BENCHMARK_JSON=/root/autodl-tmp/r4d-bench-qa-mirror-meta/scripts/new_predictions_ground_truth_final.json`
    - `RUNS_ROOT=/root/autodl-tmp/GaussianStellar/runs`
    - `DATA_ROOT=/root/autodl-tmp/GaussianStellar/data`

## Dataset Tier Clarification (important)

Current `LiYacheng/r4d-bench-qa` provides:
- Dense mask GT: **36** and **89** tiers
- Supplementary language-only queries: **246** (no dense GT)

Therefore:
- `Acc / vIoU / tIoU` can be computed for **36/89** dense-GT tiers.
- Supplementary 246 cannot produce strict mask metrics without additional dense GT alignment.

Reference: dataset README inside snapshot.

## Current Repro Status Snapshot

- 36-query recheck (with empty-set vIoU=100 rule):
  - file: `reports/repro_check_20260512/r4d_bench_eval_recheck_36_cookfix_empty100.json`
  - summary:
    - `Valid queries: 36/36`
    - `Acc: 73.7257%`
    - `vIoU: 30.7753%`
    - `tIoU: 43.7100%`

- 89-query incremental rerun:
  - runner: `reports/repro_check_20260512/run_r4d89_remaining_parallel.sh`
  - mode: single-process (`R4D89_MAX_PARALLEL=1`) to avoid Qwen OOM
  - progress log: `reports/repro_check_20260512/r4d89_remaining_parallel_master.log`


# Reproducibility Result Analysis (2026-05-13)

## 0. Evaluation scope and metric rules
- R4D-Bench-QA currently reproducible with dense GT: 36-query and 89-query tiers.
- Empty-set rule: when GT and prediction are both empty (temporal union = 0), vIoU and tIoU are counted as 1.0 (100%).
- This report uses `r4d_bench_eval_89_live_afterfix6_pathfix_empty100.json` as the final 89-query result.

## 1. 12-scene reconstruction metrics (R4D-Bench scenes)
- Ours per-scene PSNR/SSIM/LPIPS: see `reconstruction_12scene_ours_metrics.csv`.
- Ours 12-scene mean: PSNR=20.3591, SSIM=0.6370, LPIPS-vgg=0.4212.
- 4DGS baseline for all 12 scenes is not found in synchronized raw logs; only keyboard baseline numbers are available in README/paper table.

## 2. R4D-Bench-QA Acc/vIoU
- 36-query: Acc=73.7257%, vIoU=30.7753%, tIoU=43.7100%.
- 89-query (final): Acc=71.2249%, vIoU=26.3361%, tIoU=41.3758%.
- Query-level details: `r4d_bench_89_query_detail_afterfix6_empty100.csv`.

## 3. 4D LangSplat HyperNeRF annotation split Acc/vIoU
- Reproduced public-protocol scenes in logs: americano / espresso / split-cookie / chickchicken (13 queries total).
- Weighted mean on these 4 scenes: Acc=87.9363%, vIoU=54.2140%, tIoU=74.6666%.
- Per-scene table: `public_protocol_4scene_summary.csv`.
- Full HyperNeRF annotation split aggregate (paper table 91.62/66.48) was not fully re-run in this cycle; treat as paper-reported value unless full split rerun is scheduled.

## 4. Extended reconstruction comparison on keyboard (Time/FPS/Storage)
- Reproduced keyboard (ours) from logs:
  - PSNR=28.40508270263672, SSIM=0.8866782784461975, LPIPS-vgg=0.20715069770812988
  - Train time=1023 s, render stage time=58 s, storage=628822641 bytes
- 4DGS vs Ours full comparison row (including FPS) currently available from README/paper table, not from synchronized raw benchmark scripts.

## 5. Keyboard case study (A-class, 3 queries)
- Assumed A-class = `keyboard_q1/q2/q3` (three semantic queries in keyboard scene).
  - keyboard_q1: Acc=98.0769%, vIoU=1.3484%, tIoU=24.9186%
  - keyboard_q2: Acc=98.0769%, vIoU=1.6532%, tIoU=24.9186%
  - keyboard_q3: Acc=98.0769%, vIoU=0.8640%, tIoU=24.9186%
- CSV: `keyboard_case_study_A_queries.csv`.

## 6. Americano public case study (4D LangSplat annotation split)
- Scene summary: Acc=97.7186%, vIoU=69.3452%, tIoU=94.3410%.
- Query-level details: `americano_public_case_study_queries.csv`.

## 7. Ablation data
- No standalone ablation JSON/log artifact was found in synchronized server outputs.
- Current repository README keeps the paper ablation table as reference values.

## 8. vIoU alignment/path-fix patch impact
- Patched `scripts/evaluate_ours_benchmark.py` to resolve relative `binary_masks` paths and decode polygon masks without explicit size metadata.
- 89-query vIoU improved: 25.8114% -> 26.3361% (delta +0.5248 pts).
- Changed queries:
  - cut_lemon_q1: 0.0000% -> 1.9361%
  - espresso_q4: 0.0000% -> 5.2634%
  - americano_q5: 0.0000% -> 39.5069%

## 9. Gap report
- Dense-GT full 266-query evaluation is not currently possible from synchronized benchmark artifact; available dense tiers are 36 and 89.
- 12-scene 4DGS baseline reconstruction logs are missing in synchronized files (only keyboard baseline numbers available from paper/README).
- Full HyperNeRF annotation split aggregate was not fully rerun in this cycle (only 4 public-protocol scenes logged here).

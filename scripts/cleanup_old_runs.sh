#!/usr/bin/env bash
# cleanup_old_runs.sh
# 清理不再需要的旧runs目录，释放磁盘空间供Neu3D场景下载使用
# 只删除明确不需要的ablation/comparison/非benchmark场景runs
set -uo pipefail

RUNS_DIR="/root/autodl-tmp/GaussianStellar/runs"
LOG="/root/autodl-tmp/GaussianStellar/reports/ours_benchmark_eval/cleanup.log"
mkdir -p "$(dirname "$LOG")"

log_msg() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
TOTAL_FREED=0

safe_rm() {
  local dir="$1"
  local reason="$2"
  if [[ -d "${RUNS_DIR}/${dir}" ]]; then
    SIZE=$(du -sm "${RUNS_DIR}/${dir}" 2>/dev/null | cut -f1)
    log_msg "删除 ${dir} (${SIZE}MB) - ${reason}"
    rm -rf "${RUNS_DIR:?}/${dir}"
    TOTAL_FREED=$((TOTAL_FREED + SIZE))
  fi
}

log_msg "=== 开始清理旧runs ==="
log_msg "当前磁盘: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4, "available"}')"

# ===== 1. ablation/比较用的 span/sigma 变体 (只保留最优 span040_sigma032) =====
safe_rm "stellar_tube_full6_20260328_histplus_span045"              "span045非最优"
safe_rm "stellar_tube_full6_20260328_histplus_span045_sigma032"     "span045变体"
safe_rm "stellar_tube_full6_20260328_histplus_span045_sigma030"     "span045变体"
safe_rm "stellar_tube_full6_20260328_histplus_span040_sigma030"     "sigma030非最优"
safe_rm "stellar_tube_full6_20260328_histplus_sigma030"             "sigma030非最优"

# ===== 2. 各场景的ablation系列 (q20260328_weak_*) =====
for run in "${RUNS_DIR}"/stellar_tube_q20260328_*; do
  dir=$(basename "$run")
  safe_rm "$dir" "ablation系列q20260328"
done

# ===== 3. worldtube ablation系列 (q20260328) =====
for run in "${RUNS_DIR}"/stellar_worldtube_q20260328_*; do
  dir=$(basename "$run")
  safe_rm "$dir" "worldtube ablation系列"
done

# ===== 4. chickchicken场景（不在benchmark） =====
safe_rm "stellar_tube_4dlangsplat_refresh_20260328_chickchicken"   "chickchicken不在benchmark"
safe_rm "stellar_worldtube_chickchicken_compare5k"                 "chickchicken比较run"
safe_rm "baseline_chickchicken_compare5k"                          "chickchicken baseline"

# ===== 5. 旧4DLangSplat weak14k run =====
safe_rm "stellar_tube_4dlangsplat_weak14k_20260326"                "旧4DLangSplat run，已有refresh替代"

# ===== 6. 旧比较基准 (compare5k) 系列 =====
safe_rm "stellar_worldtube_americano_compare5k"                    "americano旧比较"
safe_rm "stellar_worldtube_espresso_compare5k"                     "espresso旧比较"
safe_rm "baseline_americano_compare5k"                             "americano baseline"
safe_rm "baseline_espresso_compare5k"                              "espresso baseline"
safe_rm "baseline_cut-lemon1_quality5k"                            "cut-lemon1 baseline"
safe_rm "baseline_split-cookie_compare5k"                          "split-cookie baseline"
safe_rm "baseline_split-cookie_compare5k_14000"                    "split-cookie baseline 14k"
safe_rm "stellar_tube_split-cookie_compare5k_weak"                 "split-cookie弱比较"
safe_rm "stellar_tube_split-cookie_compare5k"                      "split-cookie比较"

# ===== 7. flame_steak和coffee_martini旧run（非weaktube，将用新run替代） =====
safe_rm "stellar_tube_flame_steak_full"                            "flame_steak旧配置(span=1.0)，将用weaktube重训"
safe_rm "stellar_tube_flame_steak_smoke300_tubecmp"               "flame_steak smoke300"
safe_rm "baseline_flame_steak_full"                               "flame_steak baseline"
safe_rm "baseline_flame_steak_smoke300_tubecmp"                   "flame_steak baseline smoke300"
safe_rm "stellar_tube_coffee_martini_smoke300_tubecmp"             "coffee_martini仅300iter，将用14k重训"
safe_rm "baseline_coffee_martini_smoke300_tubecmp"                 "coffee_martini baseline"

# ===== 8. cut_lemon1旧smoke/v2/v3 run =====
safe_rm "stellar_tube_cut_lemon1_smoke300"                         "cut_lemon旧run"
safe_rm "stellar_worldtube_cut_lemon1_smoke300_v6a"               "cut_lemon worldtube旧"
safe_rm "stellar_worldtube_cut_lemon1_smoke300_v3"                 "cut_lemon worldtube v3"
safe_rm "stellar_worldtube_cut_lemon1_smoke300_v2"                 "cut_lemon worldtube v2"
safe_rm "stellar_tube_cut_lemon1_smoke1200_v2"                     "cut_lemon smoke1200"
safe_rm "stellar_tube_cut_lemon1_smoke1200"                        "cut_lemon smoke1200"
safe_rm "baseline_cut_lemon1_smoke300"                             "cut_lemon baseline"

# ===== 9. americano和split-cookie的smoke run =====
safe_rm "stellar_tube_americano_smoke300_tubecmp"                  "americano smoke300"
safe_rm "baseline_americano_smoke300_tubecmp"                      "americano baseline smoke"
safe_rm "stellar_tube_split-cookie_smoke300_tubecmp"              "split-cookie smoke300"
safe_rm "baseline_split-cookie_smoke300_tubecmp"                   "split-cookie baseline smoke"

# ===== 10. downstream实验 =====
safe_rm "downstream_split-cookie_complete-cookie_phaseaware_wtcons_contrast_20260326"  "downstream实验"
safe_rm "downstream_split-cookie_complete-cookie_phaseaware_wtcons_20260326"           "downstream实验"
safe_rm "downstream_split-cookie_complete-cookie_phaseaware_visop05_20260326"          "downstream实验"
safe_rm "downstream_split-cookie_complete-cookie_phaseaware_dense010_20260326"         "downstream实验"
safe_rm "downstream_split-cookie_complete-cookie_phaseaware_20260325"                  "downstream实验"
safe_rm "downstream_cut-lemon1_lemon_joint_cluster"                                    "downstream实验"
safe_rm "downstream_cut-lemon1_lemon_wtcons_contrast_20260326"                         "downstream实验"

# ===== 11. 早期开发runs (spacetime/standup/mutant) =====
safe_rm "stellar_spacetime_blend_full"      "早期开发run"
safe_rm "stellar_spacetime_blend_pilot"     "早期开发run"
safe_rm "stellar_spacetime_full"            "早期开发run"
safe_rm "stellar_spacetime_full_pure"       "早期开发run"
safe_rm "stellar_spacetime_opt_full"        "早期开发run"
safe_rm "stellar_spacetime_quad_pilot"      "早期开发run"
safe_rm "stellar_spacetime_quad_smoke"      "早期开发run"
safe_rm "stellar_spacetime_quad_smoke_v2"   "早期开发run"
safe_rm "stellar_spacetime_reg_full"        "早期开发run"
safe_rm "stellar_spacetime_smoke"           "早期开发run"

safe_rm "stellar_worldtube_standup_eval_a"      "standup早期"
safe_rm "stellar_worldtube_standup_eval_b"      "standup早期"
safe_rm "stellar_worldtube_standup_eval_c"      "standup早期"
safe_rm "stellar_worldtube_standup_hybrid_quad_a" "standup早期"
safe_rm "stellar_worldtube_standup_hybrid_quad_b" "standup早期"
safe_rm "stellar_worldtube_standup_pilot_v5"    "standup早期"
safe_rm "stellar_worldtube_standup_pilot_v6a"   "standup早期"
safe_rm "stellar_worldtube_standup_pilot_v6b"   "standup早期"

safe_rm "stellar_worldtube_mutant_pilot_v2"     "mutant早期"
safe_rm "stellar_worldtube_mutant_pilot_v3"     "mutant早期"
safe_rm "stellar_worldtube_mutant_pilot_v4"     "mutant早期"
safe_rm "stellar_worldtube_mutant_pilot_v4b"    "mutant早期"
safe_rm "stellar_worldtube_mutant_pilot_v5"     "mutant早期"
safe_rm "stellar_worldtube_mutant_pilot_v6a"    "mutant早期"
safe_rm "stellar_tube_mutant_pilot"             "mutant早期"
safe_rm "baseline_4dgs_mutant_pilot"            "mutant baseline"
safe_rm "baseline_4dgs_standup_pilot"           "standup baseline"

safe_rm "stellar_worldtube_slice-banana_smoke300" "slice-banana早期"
safe_rm "baseline_slice-banana_smoke300"          "slice-banana baseline"

# ===== 12. 其他早期worldtube/core开发 =====
safe_rm "stellar_core_full"                     "core早期"
safe_rm "stellar_core_smoke"                    "core早期"
safe_rm "stellar_tube_smoke"                    "tube早期"
safe_rm "stellar_tube_weak_da3_smoke"           "早期"
safe_rm "stellar_worldtube_smoke20"             "worldtube早期"
safe_rm "stellar_worldtube_segment_smoke40"     "worldtube早期"
safe_rm "stellar_worldtube_tubeaware_smoke40"   "worldtube早期"
safe_rm "stellar_worldtube_recon_v1_20260328"   "worldtube旧"
safe_rm "stellar_worldtube_recon_v1b_20260328"  "worldtube旧"
safe_rm "stellar_worldtube_recon_v2_20260328"   "worldtube旧"
safe_rm "stellar_worldtube_recon_v2_gate_20260328" "worldtube旧"
safe_rm "stellar_worldtube_generalclean_20260327" "worldtube旧"
safe_rm "stellar_worldtube_splitcookie_v6a14k_20260327" "worldtube旧"
safe_rm "stellar_worldtube_sample3_gatea_20260328" "worldtube ablation"
safe_rm "stellar_worldtube_sample3_shrink90_gatea_20260328" "worldtube ablation"
safe_rm "stellar_worldtube_lite_gatea_20260328" "worldtube ablation"

safe_rm "stellar_worldtube_focusq_20260328_base_confirm" "worldtube ablation"
safe_rm "stellar_worldtube_q20260328_txsplit_only"       "worldtube ablation"
safe_rm "stellar_worldtube_cut-lemon1_quality5k"         "cut_lemon早期"
safe_rm "stellar_worldtube_split-cookie_compare5k"       "split-cookie早期"

safe_rm "chronometric_4dgs"                    "chronometric早期"
safe_rm "chronometric_4dgs_full"               "chronometric早期"
safe_rm "baseline_4dgs"                        "早期baseline"
safe_rm "baseline_4dgs_full"                   "早期baseline"
safe_rm "baseline_4dgs_da3_smoke"              "早期baseline"

log_msg ""
log_msg "=== 清理完成 ==="
log_msg "总计释放约 ${TOTAL_FREED} MB (≈ $((TOTAL_FREED / 1024)) GB)"
log_msg "当前磁盘: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4, "available"}')"
log_msg ""
log_msg "保留的benchmark必要runs:"
ls /root/autodl-tmp/GaussianStellar/runs/ | tee -a "$LOG"

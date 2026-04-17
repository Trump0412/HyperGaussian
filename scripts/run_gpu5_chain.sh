#!/usr/bin/env bash
# GPU5 全链路：weaktube比较训练 → entitybank准备 → 21条query pipeline
set -uo pipefail
export CUDA_VISIBLE_DEVICES=5
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
BASE_14K=$GS_ROOT/runs/baseline_4dgs_20260330
LOG=$GS_ROOT/logs/gpu5_chain.log
mkdir -p $GS_ROOT/logs
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

cd $GS_ROOT

# ─── Phase 1a: 4DGS 3k baseline on split-cookie (comparsion baseline) ─────
log "=== P1a: 4DGS 3k split-cookie ==="
GS_RUN_NAMESPACE=baseline_4dgs_3k \
  bash scripts/train_baseline.sh hypernerf split-cookie --iterations 3000 --port 6024 \
  2>&1 | tee -a $LOG || log "[WARN] 4DGS 3k训练有异常，继续"

# ─── Phase 1b: WeakTube 3k (优化配置 span=0.50, sigma=0.28, samples=4) ───
log "=== P1b: WeakTube 3k 优化配置 ==="
GS_RUN_NAMESPACE=stellar_tube_3k_opt_a \
  TEMPORAL_TUBE_SPAN=0.50 \
  TEMPORAL_TUBE_SIGMA=0.28 \
  TEMPORAL_TUBE_SAMPLES=4 \
  TEMPORAL_TUBE_WEIGHT_POWER=0.7 \
  TEMPORAL_ACCELERATION_ENABLED=0 \
  bash scripts/train_stellar_tube.sh hypernerf split-cookie --iterations 3000 --port 6025 \
  2>&1 | tee -a $LOG || log "[WARN] WeakTube 3k有异常，继续"

# ─── Phase 2: 创建 config.yaml（baseline pipeline兼容性）──────────────────
log "=== P2: 创建 config.yaml ==="
declare -A SCENE_DSSC=(
  ["hypernerf/cut-lemon1"]="hypernerf:interp/cut-lemon1"
  ["hypernerf/espresso"]="hypernerf:misc/espresso"
  ["hypernerf/americano"]="hypernerf:misc/americano"
  ["hypernerf/split-cookie"]="hypernerf:misc/split-cookie"
  ["hypernerf/keyboard"]="hypernerf:misc/keyboard"
  ["hypernerf/torchocolate"]="hypernerf:interp/torchocolate"
  ["dynerf/coffee_martini"]="dynerf:coffee_martini"
)
for scene in "${!SCENE_DSSC[@]}"; do
  dssc="${SCENE_DSSC[$scene]}"
  ds="${dssc%%:*}"; sc="${dssc##*:}"
  run_dir="$BASE_14K/$scene"
  if [[ ! -f "$run_dir/config.yaml" ]]; then
    cat > "$run_dir/config.yaml" << YAML
phase: baseline_4dgs
dataset: ${ds}
scene: ${sc}
source_path: $GS_ROOT/data/${ds}/${sc}
warp_enabled: false
temporal_warp_type: identity
temporal_worldtube_enabled: false
YAML
    log "  config.yaml: $scene"
  else
    log "  config.yaml已存在: $scene"
  fi
done

# ─── Phase 3: export_entitybank（7个已完成baseline场景）─────────────────────
log "=== P3: export_entitybank ==="
for scene in "hypernerf/cut-lemon1" "hypernerf/espresso" "hypernerf/americano" \
             "hypernerf/split-cookie" "hypernerf/keyboard" "hypernerf/torchocolate" \
             "dynerf/coffee_martini"; do
  run_dir="$BASE_14K/$scene"
  if [[ -f "$run_dir/entitybank/entities.json" ]]; then
    log "  [skip] entitybank已存在: $scene"
    continue
  fi
  log "  export_entitybank: $scene"
  gs_python "$GS_ROOT/scripts/export_entitybank.py" --run-dir "$run_dir" \
    2>&1 | tee -a $LOG || log "[WARN] entitybank失败: $scene"
done

# ─── Phase 4: 21条 query pipeline（7个已完成场景 × 3 queries）────────────
log "=== P4: 21条 query pipeline ==="

declare -A SCENE_RUN=(
  ["cut_lemon"]="$BASE_14K/hypernerf/cut-lemon1"
  ["espresso"]="$BASE_14K/hypernerf/espresso"
  ["americano"]="$BASE_14K/hypernerf/americano"
  ["split_cookie"]="$BASE_14K/hypernerf/split-cookie"
  ["keyboard"]="$BASE_14K/hypernerf/keyboard"
  ["torchchocolate"]="$BASE_14K/hypernerf/torchocolate"
  ["coffee_martini"]="$BASE_14K/dynerf/coffee_martini"
)
declare -A SCENE_DATA=(
  ["cut_lemon"]="$GS_ROOT/data/hypernerf/interp/cut-lemon1"
  ["espresso"]="$GS_ROOT/data/hypernerf/misc/espresso"
  ["americano"]="$GS_ROOT/data/hypernerf/misc/americano"
  ["split_cookie"]="$GS_ROOT/data/hypernerf/misc/split-cookie"
  ["keyboard"]="$GS_ROOT/data/hypernerf/misc/keyboard"
  ["torchchocolate"]="$GS_ROOT/data/hypernerf/interp/torchocolate"
  ["coffee_martini"]="$GS_ROOT/data/dynerf/coffee_martini"
)

run_query() {
  local qid="$1" qtext="$2" sk="$3"
  local run_dir="${SCENE_RUN[$sk]:-}" ds_dir="${SCENE_DATA[$sk]:-}"
  if [[ -z "$run_dir" || ! -d "$run_dir/point_cloud" ]]; then
    log "  [skip-no-model] $qid"; return; fi
  if [[ ! -d "$ds_dir" ]]; then
    log "  [skip-no-data] $qid ($ds_dir)"; return; fi
  local final="$run_dir/entitybank/query_guided/$qid/final_query_render_sourcebg/validation.json"
  if [[ "${QUERY_FORCE_RERUN:-0}" != "1" && -f "$final" ]]; then
    log "  [skip-done] $qid"; return; fi
  log "  [run] $qid: $qtext"
  QUERY_FORCE_RERUN=1 bash "$GS_ROOT/scripts/run_query_specific_worldtube_pipeline.sh" \
    "$run_dir" "$ds_dir" "$qtext" "$qid" 2>&1 | tee -a $LOG \
    && log "  [done] $qid" || log "  [fail] $qid"
}

run_query "cut_lemon_q4"  "所有参与切割动作的物体。"             "cut_lemon"
run_query "cut_lemon_q1"  "正在切割柠檬的刀。"                   "cut_lemon"
run_query "cut_lemon_q8"  "在砧板上被切的橙子。"                 "cut_lemon"
run_query "espresso_q4"   "在整个视频中物理位置始终保持静止的所有物体。" "espresso"
run_query "espresso_q1"   "正在被注入液体的玻璃杯。"             "espresso"
run_query "espresso_q5"   "桌子上的蓝色玻璃杯。"                 "espresso"
run_query "keyboard_q6"   "除了桌子以外，所有正在打字的物体。"   "keyboard"
run_query "keyboard_q1"   "正在键盘上打字的左手。"               "keyboard"
run_query "keyboard_q4"   "红色机械键盘。"                       "keyboard"
run_query "torchchocolate_q6" "所有正在被火焰融化并发生形变的巧克力区域。" "torchchocolate"
run_query "torchchocolate_q1" "正在喷射蓝色火焰的喷枪喷嘴。"    "torchchocolate"
run_query "torchchocolate_q5" "完全凝固且坚硬的白色巧克力。"     "torchchocolate"
run_query "split_cookie_q7"   "除了正在施力的手之外的所有物体。" "split_cookie"
run_query "split_cookie_q1"   "被掰断之前的完整曲奇饼干。"       "split_cookie"
run_query "split_cookie_q5"   "木板上的黑色曲奇饼干。"           "split_cookie"
run_query "americano_q5"  "在整个视频中始终保持静止的所有物体。"  "americano"
run_query "americano_q3"  "正在倾倒咖啡的手。"                   "americano"
run_query "americano_q7"  "托盘上的蓝色玻璃杯。"                 "americano"
run_query "coffee_martini_q7" "除了黑色水壶和木碗之外的所有物体。" "coffee_martini"
run_query "coffee_martini_q2" "倾倒任何咖啡之前的金属杯。"        "coffee_martini"
run_query "coffee_martini_q6" "倾倒过程中握持马提尼酒杯的手。"    "coffee_martini"

log "=== GPU5 链路全部完成 ==="

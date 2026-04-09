#!/usr/bin/env bash
# run_ours_benchmark_query_pipeline.sh
# 对已有weaktube重建的场景运行完整query pipeline
# 按 Ours_benchmark.json 中的查询文本和ID运行
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPORT_DIR="${GS_ROOT}/reports/ours_benchmark_eval"
mkdir -p "${REPORT_DIR}"
LOG="${REPORT_DIR}/query_pipeline.log"

BENCHMARK_JSON="/root/autodl-tmp/data/Ours_benchmark.json"
QUERY_ROOT_MAP="${REPORT_DIR}/query_root_map.json"
DATASET_DIR_MAP="${REPORT_DIR}/dataset_dir_map.json"

log_msg() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG}"; }

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 初始化映射文件
if [[ ! -f "${QUERY_ROOT_MAP}" ]]; then
  echo "{}" > "${QUERY_ROOT_MAP}"
fi
if [[ ! -f "${DATASET_DIR_MAP}" ]]; then
  echo "{}" > "${DATASET_DIR_MAP}"
fi

# ===========================================================
# 辅助函数：运行单个query pipeline
# ===========================================================
run_query() {
  local query_id="$1"
  local query_text="$2"
  local run_dir="$3"
  local dataset_dir="$4"
  local query_name="${query_id}"  # 使用query_id作为slug

  local final_validation="${run_dir}/entitybank/query_guided/${query_name}/final_query_render_sourcebg/validation.json"

  if [[ "${QUERY_FORCE_RERUN:-0}" != "1" && -f "${final_validation}" ]]; then
    log_msg "[skip] 已存在: ${query_id}"
    return 0
  fi

  log_msg "[query] 开始: ${query_id} | \"${query_text:0:40}...\""
  bash "${GS_ROOT}/scripts/run_query_specific_worldtube_pipeline.sh" \
    "${run_dir}" \
    "${dataset_dir}" \
    "${query_text}" \
    "${query_name}" 2>&1 | tee -a "${LOG}" || {
      log_msg "[query] 失败: ${query_id}"
      return 1
    }
  log_msg "[query] 完成: ${query_id}"
}

# ===========================================================
# 更新映射JSON的辅助函数
# ===========================================================
update_map() {
  local map_file="$1"
  local key="$2"
  local value="$3"
  # 使用python更新json
  python3 -c "
import json, sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.loads(open(path).read())
except:
    d = {}
d[key] = value
open(path, 'w').write(json.dumps(d, indent=2, ensure_ascii=False))
" "${map_file}" "${key}" "${value}"
}

# ===========================================================
# 从 benchmark JSON 读取各场景查询（优先使用补全后的版本）
# ===========================================================
FULL_QUERIES_JSON="${REPORT_DIR}/benchmark_full_queries.json"

# 若补全版本不存在，先生成
if [[ ! -f "${FULL_QUERIES_JSON}" ]]; then
  log_msg "生成完整查询列表（补全空文本）..."
  python3 "${GS_ROOT}/scripts/prepare_ours_benchmark_queries.py"
fi

# 使用补全后的查询列表
python3 - <<'PYEOF' > "${REPORT_DIR}/benchmark_queries.tsv" "${FULL_QUERIES_JSON}"
import json, sys
queries = json.loads(open(sys.argv[1]).read())
for q in queries:
    qid = q["query_id"]
    question = q.get("question", "").strip()
    print(f"{qid}\t{question}")
PYEOF

log_msg "=== 开始批量query pipeline ==="
log_msg "共 $(wc -l < "${REPORT_DIR}/benchmark_queries.tsv") 条查询"

# ===========================================================
# Scene → run_dir + dataset_dir 映射
# ===========================================================
declare -A SCENE_RUN_DIR=(
  ["espresso"]="${GS_ROOT}/runs/stellar_tube_4dlangsplat_refresh_20260328_espresso/hypernerf/espresso"
  ["americano"]="${GS_ROOT}/runs/stellar_tube_4dlangsplat_refresh_20260328_americano/hypernerf/americano"
  ["cut_lemon"]="${GS_ROOT}/runs/stellar_tube_cutlemon_refresh_20260329/hypernerf/cut-lemon1"
  ["split_cookie"]="${GS_ROOT}/runs/stellar_tube_full6_20260328_histplus_span040_sigma032/hypernerf/split-cookie"
  ["keyboard"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_keyboard/hypernerf/keyboard"
  ["torchchocolate"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_torchocolate/hypernerf/torchocolate"
  ["coffee_martini"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_coffee_martini/dynerf/coffee_martini"
  ["flame_steak"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_flame_steak/dynerf/flame_steak"
  ["cook-spinach"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_cook_spinach/dynerf/cook_spinach"
  ["cut_roasted_beef"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_cut_roasted_beef/dynerf/cut_roasted_beef"
  ["flame_salmon"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_flame_salmon/dynerf/flame_salmon_1"
  ["sear_steak"]="${GS_ROOT}/runs/stellar_tube_ours_benchmark_sear_steak/dynerf/sear_steak"
)

declare -A SCENE_DATASET_DIR=(
  ["espresso"]="${GS_ROOT}/data/hypernerf/misc/espresso"
  ["americano"]="${GS_ROOT}/data/hypernerf/misc/americano"
  ["cut_lemon"]="${GS_ROOT}/data/hypernerf/interp/cut-lemon1"
  ["split_cookie"]="${GS_ROOT}/data/hypernerf/misc/split-cookie"
  ["keyboard"]="${GS_ROOT}/data/hypernerf/misc/keyboard"
  ["torchchocolate"]="${GS_ROOT}/data/hypernerf/interp/torchocolate"
  ["coffee_martini"]="${GS_ROOT}/data/dynerf/coffee_martini"
  ["flame_steak"]="${GS_ROOT}/data/dynerf/flame_steak"
  ["cook-spinach"]="${GS_ROOT}/data/dynerf/cook_spinach"
  ["cut_roasted_beef"]="${GS_ROOT}/data/dynerf/cut_roasted_beef"
  ["flame_salmon"]="${GS_ROOT}/data/dynerf/flame_salmon_1"
  ["sear_steak"]="${GS_ROOT}/data/dynerf/sear_steak"
)

# 从query_id推断scene_key
get_scene_key() {
  local qid="$1"
  case "${qid}" in
    cut_lemon_*) echo "cut_lemon" ;;
    espresso_*) echo "espresso" ;;
    keyboard_*) echo "keyboard" ;;
    torchchocolate_*) echo "torchchocolate" ;;
    "cook-spinach_"*) echo "cook-spinach" ;;
    cut_roasted_beef_*) echo "cut_roasted_beef" ;;
    flame_salmon_*) echo "flame_salmon" ;;
    sear_steak_*) echo "sear_steak" ;;
    split_cookie_*) echo "split_cookie" ;;
    americano_*) echo "americano" ;;
    coffee_martini_*) echo "coffee_martini" ;;
    flame_steak_*) echo "flame_steak" ;;
    *) echo "" ;;
  esac
}

# 场景限制（可通过环境变量控制）
ONLY_SCENES="${OURS_BENCHMARK_SCENES:-espresso americano cut_lemon split_cookie}"

# ===========================================================
# 主循环
# ===========================================================
while IFS=$'\t' read -r query_id query_text; do
  [[ -z "${query_id}" ]] && continue
  [[ -z "${query_text}" ]] && {
    log_msg "[skip] ${query_id}: 空查询文本（负样本查询，只记录映射）"
    # 负样本查询不需要运行pipeline，但需要在map中记录
    # 评测脚本会处理missing_validation情况
    continue
  }

  scene_key="$(get_scene_key "${query_id}")"
  [[ -z "${scene_key}" ]] && {
    log_msg "[warn] 未知场景: ${query_id}"
    continue
  }

  # 检查是否在运行集合内
  skip=1
  for s in ${ONLY_SCENES}; do
    [[ "${scene_key}" == "${s}" ]] && skip=0 && break
  done
  [[ "${skip}" == "1" ]] && {
    log_msg "[skip] 不在当前运行集合: ${query_id} (scene=${scene_key})"
    continue
  }

  run_dir="${SCENE_RUN_DIR[${scene_key}]:-}"
  dataset_dir="${SCENE_DATASET_DIR[${scene_key}]:-}"

  [[ -z "${run_dir}" || -z "${dataset_dir}" ]] && {
    log_msg "[skip] 无配置: ${query_id}"
    continue
  }

  # 检查run_dir是否存在且已训练
  if [[ ! -d "${run_dir}/point_cloud" ]]; then
    log_msg "[skip] 未训练完成: ${query_id} (${run_dir})"
    continue
  fi
  if [[ ! -d "${dataset_dir}" ]]; then
    log_msg "[skip] 数据集不存在: ${query_id} (${dataset_dir})"
    continue
  fi

  # 检查entitybank
  if [[ ! -f "${run_dir}/entitybank/entities.json" ]]; then
    log_msg "[entitybank] ${scene_key} 缺少entitybank，导出..."
    gs_python "${GS_ROOT}/scripts/export_entitybank.py" \
      --run-dir "${run_dir}" 2>&1 | tee -a "${LOG}" || {
        log_msg "[entitybank] 导出失败: ${scene_key}"
        continue
      }
  fi

  # 运行query pipeline
  run_query "${query_id}" "${query_text}" "${run_dir}" "${dataset_dir}" || continue

  # 更新映射
  query_output_dir="${run_dir}/entitybank/query_guided/${query_id}"
  if [[ -d "${query_output_dir}" ]]; then
    update_map "${QUERY_ROOT_MAP}" "${query_id}" "${query_output_dir}"
    update_map "${DATASET_DIR_MAP}" "${query_id}" "${dataset_dir}"
  fi

done < "${REPORT_DIR}/benchmark_queries.tsv"

log_msg "=== Query pipeline 批次完成 ==="
log_msg "query_root_map: ${QUERY_ROOT_MAP}"

# ===========================================================
# 运行最终评测
# ===========================================================
log_msg "=== 运行 evaluate_ours_benchmark.py ==="
gs_python "${GS_ROOT}/scripts/evaluate_ours_benchmark.py" \
  --benchmark "${BENCHMARK_JSON}" \
  --query-root-map "${QUERY_ROOT_MAP}" \
  --dataset-dir-map "${DATASET_DIR_MAP}" \
  --output-json "${REPORT_DIR}/ours_benchmark_eval.json" \
  --output-md "${REPORT_DIR}/ours_benchmark_eval.md" \
  --skip-missing 2>&1 | tee -a "${LOG}"

log_msg "评测完成！结果: ${REPORT_DIR}/ours_benchmark_eval.json"

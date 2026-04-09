#!/usr/bin/env bash
set -euo pipefail

GS_ROOT="/root/autodl-tmp/GaussianStellar"
BENCHMARK_JSON="/root/autodl-tmp/data/Ours_benchmark.json"
REPORT_DIR="${GS_ROOT}/reports/4dgs_baseline_autofill"
RUN_NAMESPACE="baseline_4dgs_20260330"
mkdir -p "${REPORT_DIR}"

source "${GS_ROOT}/scripts/common.sh"
PY_CMD="$(gs_python_cmd)"

LOG="${REPORT_DIR}/queue.log"
TRAIN_RECORDS="${REPORT_DIR}/train_records.jsonl"
SUMMARY_MD="${REPORT_DIR}/summary_latest.md"
SUMMARY_JSON="${REPORT_DIR}/summary_latest.json"
LEGACY_TRAIN_TIMES="${GS_ROOT}/reports/4dgs_baseline/train_times.jsonl"

touch "${TRAIN_RECORDS}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

gpu_uuid_by_index() {
  local gpu_idx="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
    | awk -F',' -v idx="${gpu_idx}" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if($1==idx){print $2; exit 0}}'
}

wait_gpu_idle() {
  local gpu_idx="$1"
  local gpu_uuid
  gpu_uuid="$(gpu_uuid_by_index "${gpu_idx}")"
  if [[ -z "${gpu_uuid}" ]]; then
    log "[GPU${gpu_idx}] 无法获取 GPU UUID，跳过空闲检测"
    return 0
  fi
  while true; do
    local busy
    busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
      | awk -F',' -v u="${gpu_uuid}" '{gsub(/ /,"",$1); if($1==u) print $0}' | wc -l)"
    if [[ "${busy}" -eq 0 ]]; then
      return 0
    fi
    log "[GPU${gpu_idx}] 当前有 ${busy} 个进程占用，等待 30s 后重试..."
    sleep 30
  done
}

# benchmark_scene dataset scene_path config_scene run_scene
SCENES=(
  "americano hypernerf misc/americano americano americano"
  "cut_lemon hypernerf interp/cut-lemon1 cut-lemon1 cut-lemon1"
  "espresso hypernerf misc/espresso espresso espresso"
  "keyboard hypernerf misc/keyboard keyboard keyboard"
  "split_cookie hypernerf misc/split-cookie split-cookie split-cookie"
  "torchchocolate hypernerf interp/torchocolate torchocolate torchocolate"
  "coffee_martini dynerf coffee_martini coffee_martini coffee_martini"
  "cook-spinach dynerf cook_spinach cook_spinach cook_spinach"
  "cut_roasted_beef dynerf cut_roasted_beef cut_roasted_beef cut_roasted_beef"
  "flame_salmon dynerf flame_salmon_1 flame_salmon_1 flame_salmon_1"
  "flame_steak dynerf flame_steak flame_steak flame_steak"
  "sear_steak dynerf sear_steak sear_steak sear_steak"
)

get_scene_field() {
  local scene_key="$1"
  local col="$2"
  local line
  for line in "${SCENES[@]}"; do
    if [[ "$(echo "${line}" | awk '{print $1}')" == "${scene_key}" ]]; then
      echo "${line}" | awk "{print \$${col}}"
      return 0
    fi
  done
  return 1
}

build_coverage_json() {
  local out_json="$1"
  python3 - <<'PY' "${BENCHMARK_JSON}" "${out_json}" "${GS_ROOT}" "${RUN_NAMESPACE}" "${TRAIN_RECORDS}" "${LEGACY_TRAIN_TIMES}"
import json, os, sys

benchmark_path, out_path, gs_root, run_ns, train_records_path, legacy_times_path = sys.argv[1:]

with open(benchmark_path, "r", encoding="utf-8") as f:
    bench = json.load(f)
bench_scenes = sorted(set(item["query_id"].rsplit("_q", 1)[0] for item in bench))

scene_map = {
    "americano": ("hypernerf", "americano"),
    "cut_lemon": ("hypernerf", "cut-lemon1"),
    "espresso": ("hypernerf", "espresso"),
    "keyboard": ("hypernerf", "keyboard"),
    "split_cookie": ("hypernerf", "split-cookie"),
    "torchchocolate": ("hypernerf", "torchocolate"),
    "coffee_martini": ("dynerf", "coffee_martini"),
    "cook-spinach": ("dynerf", "cook_spinach"),
    "cut_roasted_beef": ("dynerf", "cut_roasted_beef"),
    "flame_salmon": ("dynerf", "flame_salmon_1"),
    "flame_steak": ("dynerf", "flame_steak"),
    "sear_steak": ("dynerf", "sear_steak"),
}

records = []
if os.path.isfile(train_records_path):
    with open(train_records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

legacy = []
if os.path.isfile(legacy_times_path):
    with open(legacy_times_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                legacy.append(json.loads(line))
            except Exception:
                pass

def latest_record(dataset, scene):
    for rec in reversed(records):
        if rec.get("dataset") == dataset and rec.get("scene") == scene:
            return rec
    for rec in reversed(legacy):
        if rec.get("dataset") == dataset and rec.get("scene") == scene:
            return {"train_seconds": rec.get("train_seconds"), "gpu_peak_mb": None, "source": "legacy"}
    return {}

rows = []
for bscene in bench_scenes:
    if bscene not in scene_map:
        rows.append({"benchmark_scene": bscene, "status": "unmapped"})
        continue
    dataset, run_scene = scene_map[bscene]
    run_dir = os.path.join(gs_root, "runs", run_ns, dataset, run_scene)
    ckpt = os.path.join(run_dir, "point_cloud", "iteration_14000", "point_cloud.ply")
    render_dir = os.path.join(run_dir, "test", "ours_14000", "renders")
    result_json = os.path.join(run_dir, "results.json")
    rec = latest_record(dataset, run_scene)

    psnr = None
    dssim = None
    lpips = None
    if os.path.isfile(result_json):
        try:
            data = json.load(open(result_json, "r", encoding="utf-8"))
            latest = data.get("ours_14000", {})
            psnr = latest.get("PSNR")
            dssim = latest.get("D-SSIM")
            lpips = latest.get("LPIPS-vgg", latest.get("LPIPS"))
        except Exception:
            pass

    storage_mb = None
    if os.path.isfile(ckpt):
        storage_mb = int((os.path.getsize(ckpt) + (1024 * 1024 - 1)) // (1024 * 1024))

    row = {
        "benchmark_scene": bscene,
        "dataset": dataset,
        "run_scene": run_scene,
        "run_dir": run_dir,
        "has_14k_ckpt": os.path.isfile(ckpt),
        "has_render": os.path.isdir(render_dir) and len(os.listdir(render_dir)) > 0,
        "has_metrics_json": os.path.isfile(result_json),
        "psnr": psnr,
        "dssim": dssim,
        "lpips_vgg": lpips,
        "train_seconds": rec.get("train_seconds"),
        "gpu_peak_mb": rec.get("gpu_peak_mb"),
        "storage_mb": storage_mb,
    }
    row["complete_for_request"] = all([
        row["has_14k_ckpt"],
        row["has_metrics_json"],
        row["psnr"] is not None,
        row["train_seconds"] is not None,
        row["gpu_peak_mb"] is not None,
        row["storage_mb"] is not None,
    ])
    rows.append(row)

payload = {
    "total_benchmark_scenes": len(bench_scenes),
    "complete_scenes": sum(1 for r in rows if r.get("complete_for_request")),
    "rows": rows,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(out_path)
PY
}

extract_metrics_line() {
  local run_dir="$1"
  python3 - <<'PY' "${run_dir}"
import json, os, sys
run_dir = sys.argv[1]
result_json = os.path.join(run_dir, "results.json")
psnr = dssim = lpips = None
if os.path.isfile(result_json):
    try:
        d = json.load(open(result_json, "r", encoding="utf-8"))
        m = d.get("ours_14000", {})
        psnr = m.get("PSNR")
        dssim = m.get("D-SSIM")
        lpips = m.get("LPIPS-vgg", m.get("LPIPS"))
    except Exception:
        pass
def f(x):
    return "N/A" if x is None else f"{x:.4f}"
print(f"{f(psnr)}|{f(dssim)}|{f(lpips)}")
PY
}

latest_train_seconds() {
  local dataset="$1"
  local scene="$2"
  python3 - <<'PY' "${TRAIN_RECORDS}" "${LEGACY_TRAIN_TIMES}" "${dataset}" "${scene}"
import json, os, sys
new_path, old_path, ds, sc = sys.argv[1:]
for p in (new_path, old_path):
    if not os.path.isfile(p):
        continue
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    for r in reversed(rows):
        if r.get("dataset") == ds and r.get("scene") == sc and r.get("train_seconds") is not None:
            print(r.get("train_seconds"))
            sys.exit(0)
print("N/A")
PY
}

latest_peak_mem() {
  local dataset="$1"
  local scene="$2"
  python3 - <<'PY' "${TRAIN_RECORDS}" "${dataset}" "${scene}"
import json, os, sys
p, ds, sc = sys.argv[1:]
if os.path.isfile(p):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    for r in reversed(rows):
        if r.get("dataset") == ds and r.get("scene") == sc:
            peak = r.get("gpu_peak_mb")
            print("N/A" if peak is None else peak)
            sys.exit(0)
print("N/A")
PY
}

record_train_meta() {
  local dataset="$1"
  local scene="$2"
  local meta_json="$3"
  python3 - <<'PY' "${TRAIN_RECORDS}" "${dataset}" "${scene}" "${meta_json}"
import json, os, sys, datetime
out, ds, sc, meta_path = sys.argv[1:]
meta = {}
if os.path.isfile(meta_path):
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
row = {
    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    "dataset": ds,
    "scene": sc,
    "train_seconds": meta.get("elapsed_seconds"),
    "gpu_peak_mb": meta.get("gpu_peak_mb"),
    "status": meta.get("status"),
}
with open(out, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

train_and_fill_scene() {
  local gpu_id="$1"
  local benchmark_scene="$2"
  local dataset="$3"
  local scene_path="$4"
  local config_scene="$5"
  local run_scene="$6"

  local source_path="${GS_ROOT}/data/${dataset}/${scene_path}"
  local run_dir="${GS_ROOT}/runs/${RUN_NAMESPACE}/${dataset}/${run_scene}"
  local config_file="${GS_ROOT}/external/4DGaussians/arguments/${dataset}/${config_scene}.py"
  local train_log="${REPORT_DIR}/logs/train_${dataset}_${run_scene}.log"
  local train_meta="${REPORT_DIR}/logs/train_${dataset}_${run_scene}.meta.json"
  local port="$((7600 + gpu_id * 100 + RANDOM % 50))"

  mkdir -p "${REPORT_DIR}/logs"
  mkdir -p "${run_dir}"

  if [[ ! -f "${config_file}" ]]; then
    config_file="${GS_ROOT}/external/4DGaussians/arguments/${dataset}/default.py"
  fi

  local ckpt="${run_dir}/point_cloud/iteration_14000/point_cloud.ply"
  local render_dir="${run_dir}/test/ours_14000/renders"
  local result_json="${run_dir}/results.json"

  log "[GPU${gpu_id}] ${benchmark_scene} -> ${dataset}/${run_scene}"
  wait_gpu_idle "${gpu_id}"
  if [[ -f "${ckpt}" ]]; then
    log "[GPU${gpu_id}] 14K checkpoint 已存在，跳过训练"
  else
    log "[GPU${gpu_id}] 开始训练（含显存峰值监控）"
    local train_cmd
    train_cmd="cd '${GS_ROOT}' && export PYTHONPATH='${PYTHONPATH}' && CUDA_VISIBLE_DEVICES='${gpu_id}' ${PY_CMD} external/4DGaussians/train.py -s '${source_path}' -m '${run_dir}' --expname 'baseline_4dgs_autofill/${dataset}/${run_scene}' --configs '${config_file}' --port '${port}'"
    run_with_gpu_monitor "${train_log}" "${train_meta}" bash -lc "${train_cmd}"
    record_train_meta "${dataset}" "${run_scene}" "${train_meta}"
    log "[GPU${gpu_id}] 训练完成"
  fi

  if [[ ! -d "${render_dir}" || "$(ls -A "${render_dir}" 2>/dev/null | wc -l)" -eq 0 ]]; then
    log "[GPU${gpu_id}] render ${run_scene}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "
      cd '${GS_ROOT}'
      export PYTHONPATH='${PYTHONPATH}'
      ${PY_CMD} external/4DGaussians/render.py -m '${run_dir}' --iteration -1
    "
  fi

  if [[ ! -f "${result_json}" ]]; then
    log "[GPU${gpu_id}] metrics ${run_scene}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "
      cd '${GS_ROOT}'
      export PYTHONPATH='${PYTHONPATH}'
      ${PY_CMD} external/4DGaussians/metrics.py -m '${run_dir}'
    "
  fi
}

emit_summary() {
  local coverage_json="$1"
  python3 - <<'PY' "${coverage_json}" "${SUMMARY_MD}" "${SUMMARY_JSON}"
import json, sys
cov_path, md_path, json_path = sys.argv[1:]
cov = json.load(open(cov_path, "r", encoding="utf-8"))
rows = cov["rows"]

json.dump(cov, open(json_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

lines = []
lines.append("# 4DGS 14K Benchmark Autofill Summary")
lines.append("")
lines.append(f"- Total scenes: {cov.get('total_benchmark_scenes', 0)}")
lines.append(f"- Fully complete (PSNR+Time+GPU Mem+Storage+14K): {cov.get('complete_scenes', 0)}")
lines.append("")
lines.append("| Scene | Dataset | 14K | PSNR | D-SSIM | LPIPS-vgg | Train(s) | GPU Peak(MB) | Storage(MB) |")
lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for r in rows:
    def fmt(v, n=4):
        if v is None:
            return "N/A"
        if isinstance(v, bool):
            return "Y" if v else "N"
        if isinstance(v, (int, float)):
            if isinstance(v, int):
                return str(v)
            return f"{v:.{n}f}"
        return str(v)
    lines.append(
        f"| {r.get('benchmark_scene')} | {r.get('dataset')} | "
        f"{'Y' if r.get('has_14k_ckpt') else 'N'} | {fmt(r.get('psnr'))} | {fmt(r.get('dssim'))} | "
        f"{fmt(r.get('lpips_vgg'))} | {fmt(r.get('train_seconds'),0)} | {fmt(r.get('gpu_peak_mb'),0)} | {fmt(r.get('storage_mb'),0)} |"
    )
open(md_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(md_path)
PY
}

main() {
  : > "${LOG}"
  local before_json="${REPORT_DIR}/coverage_before.json"
  local after_json="${REPORT_DIR}/coverage_after.json"

  log "开始检查 benchmark 场景覆盖情况"
  build_coverage_json "${before_json}" >/dev/null
  emit_summary "${before_json}" >/dev/null

  log "开始自动补齐缺失项（仅训练缺14K checkpoint的场景）"
  (
    for item in "americano" "cut_lemon" "espresso" "keyboard" "split_cookie" "torchchocolate"; do
      train_and_fill_scene 0 "${item}" "$(get_scene_field "${item}" 2)" "$(get_scene_field "${item}" 3)" "$(get_scene_field "${item}" 4)" "$(get_scene_field "${item}" 5)"
    done
  ) &
  local p0=$!
  (
    for item in "coffee_martini" "cook-spinach" "cut_roasted_beef" "flame_salmon" "flame_steak" "sear_steak"; do
      train_and_fill_scene 1 "${item}" "$(get_scene_field "${item}" 2)" "$(get_scene_field "${item}" 3)" "$(get_scene_field "${item}" 4)" "$(get_scene_field "${item}" 5)"
    done
  ) &
  local p1=$!

  log "GPU0 PID=${p0}, GPU1 PID=${p1}"
  wait "${p0}" "${p1}"

  log "训练补齐阶段完成，生成最终汇总"
  build_coverage_json "${after_json}" >/dev/null
  emit_summary "${after_json}" >/dev/null
  log "完成。汇总文件: ${SUMMARY_MD}"
}

main "$@"

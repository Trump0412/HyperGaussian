#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

REPORT_DIR="${GS_ROOT}/reports/strict_14k_repro"
LOG="${REPORT_DIR}/queue.log"
RESULTS_JSON="${REPORT_DIR}/results_latest.json"
RESULTS_MD="${REPORT_DIR}/results_latest.md"
mkdir -p "${REPORT_DIR}"

SEEDS=(3407 7777 2026)
SCENES=("hypernerf misc/americano americano" "hypernerf misc/espresso espresso")

WTUBE_ENV=(
  "TEMPORAL_TUBE_SAMPLES=3"
  "TEMPORAL_TUBE_SPAN=0.40"
  "TEMPORAL_TUBE_SIGMA=0.32"
  "TEMPORAL_TUBE_COVARIANCE_MIX=0.05"
  "TEMPORAL_TUBE_WEIGHT_POWER=1.0"
  "TEMPORAL_DRIFT_SCALE=1.0"
  "TEMPORAL_GATE_MIX=1.0"
  "TEMPORAL_DRIFT_MIX=1.0"
  "TEMPORAL_ACCELERATION_ENABLED=0"
  "TEMPORAL_VELOCITY_REG_WEIGHT=0.0"
  "TEMPORAL_ACCELERATION_REG_WEIGHT=0.0"
)

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
    log "[GPU${gpu_idx}] 当前有 ${busy} 个进程占用，等待 20s..."
    sleep 20
  done
}

maybe_train_eval_baseline() {
  local gpu="$1"
  local dataset="$2"
  local scene_path="$3"
  local scene_name="$4"
  local seed="$5"
  local ns="strict14k_baseline_seed${seed}"
  local run_dir="${GS_ROOT}/runs/${ns}/${dataset}/${scene_name}"
  local ckpt="${run_dir}/point_cloud/iteration_14000/point_cloud.ply"
  log "[baseline][GPU${gpu}] ${scene_name} seed=${seed}"
  wait_gpu_idle "${gpu}"
  if [[ ! -f "${ckpt}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" GS_RUN_NAMESPACE="${ns}" \
      bash "${GS_ROOT}/scripts/train_baseline.sh" "${dataset}" "${scene_path}" \
      --iterations 14000 --coarse_iterations 3000 \
      --test_iterations 3000 7000 14000 \
      --save_iterations 7000 14000 \
      --checkpoint_iterations 7000 14000 \
      --seed "${seed}"
  else
    log "[baseline][GPU${gpu}] 已有14k checkpoint，跳过训练"
  fi
  if [[ ! -f "${run_dir}/results.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" GS_RUN_NAMESPACE="${ns}" \
      bash "${GS_ROOT}/scripts/eval_baseline.sh" "${dataset}" "${scene_path}"
  fi
}

maybe_train_eval_weaktube() {
  local gpu="$1"
  local dataset="$2"
  local scene_path="$3"
  local scene_name="$4"
  local seed="$5"
  local ns="strict14k_weaktube_seed${seed}"
  local run_dir="${GS_ROOT}/runs/${ns}/${dataset}/${scene_name}"
  local ckpt="${run_dir}/point_cloud/iteration_14000/point_cloud.ply"
  log "[weaktube][GPU${gpu}] ${scene_name} seed=${seed}"
  wait_gpu_idle "${gpu}"
  if [[ ! -f "${ckpt}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" GS_RUN_NAMESPACE="${ns}" \
      "${WTUBE_ENV[@]}" \
      bash "${GS_ROOT}/scripts/train_stellar_tube.sh" "${dataset}" "${scene_path}" \
      --iterations 14000 --coarse_iterations 3000 \
      --test_iterations 3000 7000 14000 \
      --save_iterations 7000 14000 \
      --checkpoint_iterations 7000 14000 \
      --seed "${seed}"
  else
    log "[weaktube][GPU${gpu}] 已有14k checkpoint，跳过训练"
  fi
  if [[ ! -f "${run_dir}/results.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" GS_RUN_NAMESPACE="${ns}" \
      "${WTUBE_ENV[@]}" \
      bash "${GS_ROOT}/scripts/eval_stellar_tube.sh" "${dataset}" "${scene_path}"
  fi
}

build_report() {
  python3 - <<'PY' "${GS_ROOT}" "${RESULTS_JSON}" "${RESULTS_MD}"
import json, os, statistics, sys
gs_root, out_json, out_md = sys.argv[1:]
seeds = [3407, 7777, 2026]
scenes = [("hypernerf", "americano"), ("hypernerf", "espresso")]

def read_metrics(path):
    if not os.path.isfile(path):
        return None
    data = json.load(open(path, "r", encoding="utf-8"))
    m = data.get("ours_14000", {})
    return {
        "PSNR": m.get("PSNR"),
        "D-SSIM": m.get("D-SSIM"),
        "LPIPS-vgg": m.get("LPIPS-vgg", m.get("LPIPS")),
    }

rows = []
for ds, scene in scenes:
    for seed in seeds:
        b = os.path.join(gs_root, "runs", f"strict14k_baseline_seed{seed}", ds, scene, "results.json")
        w = os.path.join(gs_root, "runs", f"strict14k_weaktube_seed{seed}", ds, scene, "results.json")
        bm, wm = read_metrics(b), read_metrics(w)
        rows.append({
            "dataset": ds,
            "scene": scene,
            "seed": seed,
            "baseline": bm,
            "weaktube": wm,
        })

def collect(scene, key, method):
    vals = []
    for r in rows:
        if r["scene"] != scene:
            continue
        m = r[method]
        if m and m.get(key) is not None:
            vals.append(float(m[key]))
    return vals

summary = {}
for _, scene in scenes:
    summary[scene] = {}
    for method in ("baseline", "weaktube"):
        summary[scene][method] = {}
        for key in ("PSNR", "D-SSIM", "LPIPS-vgg"):
            vals = collect(scene, key, method)
            summary[scene][method][key] = {
                "n": len(vals),
                "mean": (sum(vals) / len(vals)) if vals else None,
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else None,
            }

payload = {"rows": rows, "summary": summary}
json.dump(payload, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

lines = ["# Strict 14K Repro Results", ""]
lines.append("| Scene | Seed | Baseline PSNR | Weaktube PSNR | Baseline D-SSIM | Weaktube D-SSIM | Baseline LPIPS | Weaktube LPIPS |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for r in rows:
    b = r["baseline"] or {}
    w = r["weaktube"] or {}
    def f(v): return "N/A" if v is None else f"{v:.4f}"
    lines.append(
        f"| {r['scene']} | {r['seed']} | {f(b.get('PSNR'))} | {f(w.get('PSNR'))} | "
        f"{f(b.get('D-SSIM'))} | {f(w.get('D-SSIM'))} | {f(b.get('LPIPS-vgg'))} | {f(w.get('LPIPS-vgg'))} |"
    )
lines.append("")
lines.append("## Mean ± Std by Scene")
lines.append("")
lines.append("| Scene | Method | PSNR | D-SSIM | LPIPS-vgg | n |")
lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
for _, scene in scenes:
    for method in ("baseline", "weaktube"):
        sm = summary[scene][method]
        n = sm["PSNR"]["n"]
        def g(k):
            m, s = sm[k]["mean"], sm[k]["std"]
            if m is None: return "N/A"
            return f"{m:.4f} ± {s:.4f}"
        lines.append(f"| {scene} | {method} | {g('PSNR')} | {g('D-SSIM')} | {g('LPIPS-vgg')} | {n} |")

open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(out_md)
PY
}

main() {
  : > "${LOG}"
  log "=== strict 14k repro queue start ==="
  log "scenes: americano, espresso | seeds: ${SEEDS[*]}"
  (
    for entry in "${SCENES[@]}"; do
      ds="$(echo "${entry}" | awk '{print $1}')"
      sp="$(echo "${entry}" | awk '{print $2}')"
      sn="$(echo "${entry}" | awk '{print $3}')"
      for s in "${SEEDS[@]}"; do
        maybe_train_eval_baseline 0 "${ds}" "${sp}" "${sn}" "${s}"
        build_report
      done
    done
  ) &
  p0=$!
  (
    for entry in "${SCENES[@]}"; do
      ds="$(echo "${entry}" | awk '{print $1}')"
      sp="$(echo "${entry}" | awk '{print $2}')"
      sn="$(echo "${entry}" | awk '{print $3}')"
      for s in "${SEEDS[@]}"; do
        maybe_train_eval_weaktube 1 "${ds}" "${sp}" "${sn}" "${s}"
        build_report
      done
    done
  ) &
  p1=$!
  log "worker pid: baseline=${p0}, weaktube=${p1}"
  wait "${p0}" "${p1}"
  build_report
  log "done. report: ${RESULTS_MD}"
}

main "$@"

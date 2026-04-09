#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_BASELINE_SH = REPO_ROOT / "scripts" / "train_baseline.sh"
TRAIN_TUBE_SH = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_TUBE_SH = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"
COMMON_SH = REPO_ROOT / "scripts" / "common.sh"
FULLFRAME_PY = REPO_ROOT / "scripts" / "fullframe_metrics.py"

REPORT_DIR = REPO_ROOT / "reports" / "splitcookie_weaktube_opt_round2_20260401"
STATUS_JSON = REPORT_DIR / "queue_status.json"
LEADERBOARD_MD = REPORT_DIR / "leaderboard.md"
MANIFEST_JSON = REPORT_DIR / "manifest.json"

DATASET = "hypernerf"
SCENE = "misc/split-cookie"
SCENE_NAME = SCENE.split("/")[-1]
MAX_WORKERS = int(os.environ.get("WEAKTUBE_MAX_WORKERS", "4"))
PROMOTE_TOP_K = int(os.environ.get("WEAKTUBE_PROMOTE_TOP_K", "10"))

BASELINE_3K_NAMESPACE = os.environ.get(
    "WEAKTUBE_BASELINE_3K_NAMESPACE", "baseline_splitcookie_opt3k_round2_20260401"
)
BASELINE_FULL_REF_RUN = REPO_ROOT / "runs" / "baseline_4dgs_20260330" / DATASET / SCENE_NAME
BASELINE_FULL_REF_JSON = BASELINE_FULL_REF_RUN / "full_metrics_with_lpips_recheck_20260330.json"
WEAK_HIST_REF_JSON = (
    REPO_ROOT
    / "runs"
    / "stellar_tube_full6_20260328_histplus_span040_sigma032"
    / DATASET
    / SCENE_NAME
    / "full_metrics_with_lpips_full14k.json"
)

COMMON_3K_ARGS = [
    "--iterations",
    "3000",
    "--coarse_iterations",
    "3000",
    "--test_iterations",
    "3000",
    "--save_iterations",
    "3000",
    "--checkpoint_iterations",
    "3000",
]

COMMON_14K_ARGS = [
    "--iterations",
    "14000",
    "--coarse_iterations",
    "3000",
    "--test_iterations",
    "3000",
    "7000",
    "14000",
    "--save_iterations",
    "7000",
    "14000",
    "--checkpoint_iterations",
    "7000",
    "14000",
]

FULL_METRICS_OUT_NAME = "full_metrics_with_lpips_opt_round2_20260401.json"

BASE_ENV = {
    "TEMPORAL_TUBE_SAMPLES": "3",
    "TEMPORAL_TUBE_SPAN": "0.40",
    "TEMPORAL_TUBE_SIGMA": "0.32",
    "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
    "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
    "TEMPORAL_DRIFT_SCALE": "1.0",
    "TEMPORAL_GATE_MIX": "1.0",
    "TEMPORAL_DRIFT_MIX": "1.0",
    "TEMPORAL_ACCELERATION_ENABLED": "0",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
}

VARIANTS = [
    {
        "name": "c040_s032_cov005_w100",
        "description": "Control around historical strongest full-score recipe.",
        "env": {},
    },
    {
        "name": "c040_s032_cov004_w100",
        "description": "Control with slightly lower covariance mixing.",
        "env": {
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.04",
        },
    },
    {
        "name": "c040_s032_cov006_w100",
        "description": "Control with slightly higher covariance mixing.",
        "env": {
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.06",
        },
    },
    {
        "name": "c040_s030_cov005_w100",
        "description": "Slightly sharper sigma at historical span.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.30",
        },
    },
    {
        "name": "c040_s034_cov005_w100",
        "description": "Slightly softer sigma at historical span.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.34",
        },
    },
    {
        "name": "c038_s032_cov005_w100",
        "description": "Slightly shorter support around historical center.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.38",
        },
    },
    {
        "name": "c042_s032_cov005_w100",
        "description": "Slightly wider support around historical center.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.42",
        },
    },
    {
        "name": "c045_s032_cov005_w100",
        "description": "Wider support from historical full6 branch.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.45",
        },
    },
    {
        "name": "c040_s032_cov005_w095",
        "description": "Milder center weighting.",
        "env": {
            "TEMPORAL_TUBE_WEIGHT_POWER": "0.95",
        },
    },
    {
        "name": "c040_s032_cov005_w110",
        "description": "Stronger center weighting.",
        "env": {
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.10",
        },
    },
    {
        "name": "c040_s032_cov005_gs120",
        "description": "Sharper temporal gate while preserving support recipe.",
        "env": {
            "TEMPORAL_GATE_SHARPNESS": "1.20",
        },
    },
    {
        "name": "c040_s032_cov005_d095",
        "description": "Slightly reduced temporal drift scale.",
        "env": {
            "TEMPORAL_DRIFT_SCALE": "0.95",
        },
    },
    {
        "name": "c040_s032_cov005_d105",
        "description": "Slightly increased temporal drift scale.",
        "env": {
            "TEMPORAL_DRIFT_SCALE": "1.05",
        },
    },
    {
        "name": "c040_s032_cov005_gm110_dm095",
        "description": "Stronger gate blend with slightly reduced drift blend.",
        "env": {
            "TEMPORAL_GATE_MIX": "1.10",
            "TEMPORAL_DRIFT_MIX": "0.95",
        },
    },
    {
        "name": "c042_s030_cov005_w105_gs110",
        "description": "Wider-sharper mix with mild gate sharpening.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.42",
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.05",
            "TEMPORAL_GATE_SHARPNESS": "1.10",
        },
    },
    {
        "name": "c038_s030_cov006_w110_gm095_dm105",
        "description": "Shorter-sharper mixed support with conservative gate and stronger drift blend.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.38",
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.06",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.10",
            "TEMPORAL_GATE_MIX": "0.95",
            "TEMPORAL_DRIFT_MIX": "1.05",
        },
    },
    {
        "name": "c035_s028_cov008_w120",
        "description": "Round-1 robust 3k winner branch for compatibility.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.35",
            "TEMPORAL_TUBE_SIGMA": "0.28",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.08",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.20",
        },
    },
    {
        "name": "c035_s026_cov007_w125",
        "description": "Round-1 top 3k branch for direct full-score recheck.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.35",
            "TEMPORAL_TUBE_SIGMA": "0.26",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.07",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.25",
        },
    },
    {
        "name": "c040_s032_cov005_acc1_v1e4_a5e5",
        "description": "Enable acceleration with light velocity/acceleration regularization.",
        "env": {
            "TEMPORAL_ACCELERATION_ENABLED": "1",
            "TEMPORAL_VELOCITY_REG_WEIGHT": "0.00010",
            "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.00005",
        },
    },
    {
        "name": "c042_s032_cov005_acc1_v5e5_a0",
        "description": "Wider support + acceleration with tiny velocity regularization.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.42",
            "TEMPORAL_ACCELERATION_ENABLED": "1",
            "TEMPORAL_VELOCITY_REG_WEIGHT": "0.00005",
            "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
        },
    },
    {
        "name": "c040_s030_cov005_w100_lrhi",
        "description": "Sharper sigma with slightly higher temporal learning rate.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_LR_INIT": "0.00020",
            "TEMPORAL_LR_FINAL": "0.000020",
        },
    },
    {
        "name": "c040_s034_cov005_w100_lrlow",
        "description": "Softer sigma with slightly lower temporal learning rate.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.34",
            "TEMPORAL_LR_INIT": "0.00012",
            "TEMPORAL_LR_FINAL": "0.000012",
        },
    },
    {
        "name": "s5_span070_sigma050_cov060_w100",
        "description": "A moderate 5-sample weak-tube branch to test broader integration.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "5",
            "TEMPORAL_TUBE_SPAN": "0.70",
            "TEMPORAL_TUBE_SIGMA": "0.50",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.60",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
        },
    },
    {
        "name": "s5_span060_sigma045_cov050_w100",
        "description": "A tighter 5-sample branch near the historical support family.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "5",
            "TEMPORAL_TUBE_SPAN": "0.60",
            "TEMPORAL_TUBE_SIGMA": "0.45",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.50",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
        },
    },
]

for idx, variant in enumerate(VARIANTS):
    variant["gpu"] = idx % MAX_WORKERS
    variant["port"] = 6901 + idx


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_dir_for(namespace: str) -> Path:
    return REPO_ROOT / "runs" / namespace / DATASET / SCENE_NAME


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_train_metrics(train_log: Path) -> dict:
    payload = {
        "coarse_3000_l1": None,
        "coarse_3000_psnr": None,
        "fine_3000_l1": None,
        "fine_3000_psnr": None,
        "test_7000_l1": None,
        "test_7000_psnr": None,
        "test_14000_l1": None,
        "test_14000_psnr": None,
        "status": "missing",
    }
    if not train_log.exists():
        return payload

    text = train_log.read_text(errors="ignore")
    matches = re.findall(r"\[ITER (\d+)\] Evaluating test: L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+)", text)

    iter3000_seen = 0
    for iter_str, l1_str, psnr_str in matches:
        step = int(iter_str)
        l1 = float(l1_str)
        psnr = float(psnr_str)
        if step == 3000:
            iter3000_seen += 1
            if iter3000_seen == 1:
                payload["coarse_3000_l1"] = l1
                payload["coarse_3000_psnr"] = psnr
            else:
                payload["fine_3000_l1"] = l1
                payload["fine_3000_psnr"] = psnr
        elif step == 7000:
            payload["test_7000_l1"] = l1
            payload["test_7000_psnr"] = psnr
        elif step == 14000:
            payload["test_14000_l1"] = l1
            payload["test_14000_psnr"] = psnr

    if "Training complete." in text:
        payload["status"] = "trained"
    elif matches:
        payload["status"] = "running"
    return payload


def parse_results_metrics(run_dir: Path, iteration: str) -> dict:
    payload = {"render_psnr": None, "render_ssim": None, "render_msssim": None, "render_lpips": None}
    results = load_json(run_dir / "results.json")
    if not results:
        return payload
    key = f"ours_{iteration}"
    block = results.get(key)
    if block is None and results:
        keys = sorted(results.keys())
        if keys:
            block = results.get(keys[-1], {})
    block = block or {}
    payload["render_psnr"] = float_or_none(block.get("PSNR"))
    payload["render_ssim"] = float_or_none(block.get("SSIM"))
    payload["render_msssim"] = float_or_none(block.get("MS-SSIM"))
    payload["render_lpips"] = float_or_none(block.get("LPIPS-vgg"))
    return payload


def parse_full_metrics(run_dir: Path) -> dict:
    payload = {
        "full_psnr": None,
        "full_ssim": None,
        "full_msssim": None,
        "full_lpips": None,
        "full_metrics_json": None,
    }
    candidates = [
        run_dir / FULL_METRICS_OUT_NAME,
        run_dir / "full_metrics_with_lpips_full14k.json",
        run_dir / "full_metrics_with_lpips_recheck_20260330.json",
        run_dir / "full_metrics_with_lpips_rerun_20260326.json",
        run_dir / "full_metrics.json",
    ]
    for path in candidates:
        data = load_json(path)
        if not data:
            continue
        payload["full_psnr"] = float_or_none(data.get("PSNR"))
        payload["full_ssim"] = float_or_none(data.get("SSIM"))
        payload["full_msssim"] = float_or_none(data.get("MS-SSIM"))
        payload["full_lpips"] = float_or_none(data.get("LPIPS-vgg"))
        payload["full_metrics_json"] = str(path)
        break
    return payload


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def run_cmd(cmd: list[str], env: dict[str, str], label: str) -> int:
    print(f"[run] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd="/root", env=env)
    return proc.returncode


def load_reference_psnr(path: Path) -> float | None:
    data = load_json(path) or {}
    return float_or_none(data.get("PSNR"))


def baseline_3k_result() -> dict:
    namespace = BASELINE_3K_NAMESPACE
    run_dir = run_dir_for(namespace)
    result = {
        "kind": "baseline_3k",
        "name": "baseline_4dgs_3k_rebuilt",
        "namespace": namespace,
        "gpu": 0,
        "port": 6700,
        "run_dir": str(run_dir),
    }
    result.update(parse_train_metrics(run_dir / "train.log"))

    if result["fine_3000_psnr"] is not None and result["status"] == "trained":
        result["status"] = "complete_cached"
        return result

    env = dict(os.environ)
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = "6700"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    train_cmd = ["bash", str(TRAIN_BASELINE_SH), DATASET, SCENE, *COMMON_3K_ARGS]
    rc = run_cmd(train_cmd, env, "baseline3k-train")

    result.update(parse_train_metrics(run_dir / "train.log"))
    if rc != 0:
        result["status"] = f"train_failed_{rc}"
    else:
        result["status"] = "complete"
    return result


def variant_3k_result(variant: dict) -> dict:
    namespace = f"stellar_tube_splitcookie_opt3k_round2_20260401_{variant['name']}"
    run_dir = run_dir_for(namespace)
    result = {
        "kind": "tube_3k",
        "name": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "gpu": variant["gpu"],
        "port": variant["port"],
        "run_dir": str(run_dir),
        "env": {**BASE_ENV, **variant["env"]},
    }
    result.update(parse_train_metrics(run_dir / "train.log"))

    if result["fine_3000_psnr"] is not None and result["status"] == "trained":
        result["status"] = "complete_cached"
        return result

    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(variant["port"])
    env["CUDA_VISIBLE_DEVICES"] = str(variant["gpu"])

    train_cmd = ["bash", str(TRAIN_TUBE_SH), DATASET, SCENE, *COMMON_3K_ARGS]
    rc = run_cmd(train_cmd, env, f"tube3k-{variant['name']}")

    result.update(parse_train_metrics(run_dir / "train.log"))
    if rc != 0:
        result["status"] = f"train_failed_{rc}"
    else:
        result["status"] = "complete"
    return result


def variant_full_result(variant: dict) -> dict:
    namespace = f"stellar_tube_splitcookie_opt14k_round2_20260401_{variant['name']}"
    run_dir = run_dir_for(namespace)
    result = {
        "kind": "tube_14k",
        "name": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "gpu": variant["gpu"],
        "port": variant["port"],
        "run_dir": str(run_dir),
        "env": {**BASE_ENV, **variant["env"]},
    }
    result.update(parse_train_metrics(run_dir / "train.log"))
    result.update(parse_results_metrics(run_dir, "14000"))
    result.update(parse_full_metrics(run_dir))

    if result["test_14000_psnr"] is not None and result["full_psnr"] is not None:
        result["status"] = "complete_cached"
        return result

    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(variant["port"])
    env["CUDA_VISIBLE_DEVICES"] = str(variant["gpu"])

    train_cmd = ["bash", str(TRAIN_TUBE_SH), DATASET, SCENE, *COMMON_14K_ARGS]
    train_rc = run_cmd(train_cmd, env, f"tube14k-train-{variant['name']}")
    result.update(parse_train_metrics(run_dir / "train.log"))
    if train_rc != 0:
        result["status"] = f"train_failed_{train_rc}"
        return result

    eval_rc = run_cmd(["bash", str(EVAL_TUBE_SH), DATASET, SCENE], env, f"tube14k-eval-{variant['name']}")
    result.update(parse_results_metrics(run_dir, "14000"))
    if eval_rc != 0:
        result["status"] = f"eval_failed_{eval_rc}"
        return result

    full_cmd = [
        "bash",
        "-lc",
        (
            f"source '{COMMON_SH}' && "
            f"gs_python '{FULLFRAME_PY}' --run-dir '{run_dir}' --with-lpips --out-name '{FULL_METRICS_OUT_NAME}'"
        ),
    ]
    full_rc = run_cmd(full_cmd, env, f"tube14k-fullmetrics-{variant['name']}")
    result.update(parse_full_metrics(run_dir))
    if full_rc != 0:
        result["status"] = f"fullmetrics_failed_{full_rc}"
        return result

    result["status"] = "complete"
    return result


def write_manifest(reference_baseline_full: float | None, reference_weak_hist: float | None) -> None:
    payload = {
        "dataset": DATASET,
        "scene": SCENE,
        "max_workers": MAX_WORKERS,
        "common_3k_args": COMMON_3K_ARGS,
        "common_14k_args": COMMON_14K_ARGS,
        "promote_top_k": PROMOTE_TOP_K,
        "baseline_3k_namespace": BASELINE_3K_NAMESPACE,
        "baseline_full_reference_run": str(BASELINE_FULL_REF_RUN),
        "baseline_full_reference_json": str(BASELINE_FULL_REF_JSON),
        "baseline_full_reference_psnr": reference_baseline_full,
        "weak_historical_reference_json": str(WEAK_HIST_REF_JSON),
        "weak_historical_reference_psnr": reference_weak_hist,
        "base_env": BASE_ENV,
        "variants": VARIANTS,
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2))


def write_status(
    baseline_3k: dict | None,
    results_3k: list[dict],
    promoted_names: list[str],
    results_14k: list[dict],
    running: list[str],
    baseline_full_ref_psnr: float | None,
    weak_hist_ref_psnr: float | None,
) -> None:
    STATUS_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "scene": SCENE,
                "baseline_3k": baseline_3k,
                "baseline_full_reference_psnr": baseline_full_ref_psnr,
                "weak_historical_reference_psnr": weak_hist_ref_psnr,
                "running": running,
                "results_3k": results_3k,
                "promoted_variants": promoted_names,
                "results_14k": results_14k,
            },
            indent=2,
        )
    )

    baseline_fine = None
    if baseline_3k:
        baseline_fine = baseline_3k.get("fine_3000_psnr") or baseline_3k.get("coarse_3000_psnr")

    rank_3k = sorted(
        results_3k,
        key=lambda item: item.get("fine_3000_psnr") if item.get("fine_3000_psnr") is not None else -1.0,
        reverse=True,
    )
    rank_14k = sorted(
        results_14k,
        key=lambda item: item.get("full_psnr") if item.get("full_psnr") is not None else -1.0,
        reverse=True,
    )

    lines = [
        "# split-cookie WeakTube Optimization Round2 20260401",
        "",
        f"- Dataset/scene: `{DATASET} / {SCENE}`",
        f"- Baseline 4DGS fine-3000 PSNR (rebuilt): `{format_float(baseline_fine)}`",
        f"- Baseline 4DGS full-14000 PSNR (reference): `{format_float(baseline_full_ref_psnr)}`",
        f"- WeakTube historical full-14000 PSNR: `{format_float(weak_hist_ref_psnr)}`",
        f"- Promoted to 14k: `{', '.join(promoted_names) if promoted_names else 'none'}`",
        f"- Running: `{', '.join(running) if running else 'none'}`",
        "",
        "## 3k Gate Leaderboard",
        "",
        "| Variant | Fine3000 | Delta vs Baseline3k | Coarse3000 | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in rank_3k:
        fine = item.get("fine_3000_psnr")
        delta = None if fine is None or baseline_fine is None else fine - baseline_fine
        lines.append(
            f"| `{item['name']}` | {format_float(fine)} | {format_float(delta)} | "
            f"{format_float(item.get('coarse_3000_psnr'))} | {item.get('status', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## 14k Full Leaderboard",
            "",
            "| Variant | Full PSNR | Delta vs 4DGS Full | Test14000 | Render PSNR | Full SSIM | Full LPIPS | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in rank_14k:
        full_psnr = item.get("full_psnr")
        delta_full = None if full_psnr is None or baseline_full_ref_psnr is None else full_psnr - baseline_full_ref_psnr
        lines.append(
            f"| `{item['name']}` | {format_float(full_psnr)} | {format_float(delta_full)} | "
            f"{format_float(item.get('test_14000_psnr'))} | {format_float(item.get('render_psnr'))} | "
            f"{format_float(item.get('full_ssim'))} | {format_float(item.get('full_lpips'))} | "
            f"{item.get('status', 'n/a')} |"
        )

    LEADERBOARD_MD.write_text("\n".join(lines) + "\n")


def make_exception_result(kind: str, variant: dict, suffix: str) -> dict:
    if kind == "3k":
        namespace = f"stellar_tube_splitcookie_opt3k_round2_20260401_{variant['name']}"
    else:
        namespace = f"stellar_tube_splitcookie_opt14k_round2_20260401_{variant['name']}"
    run_dir = run_dir_for(namespace)
    return {
        "kind": f"tube_{kind}",
        "name": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "gpu": variant["gpu"],
        "port": variant["port"],
        "run_dir": str(run_dir),
        "env": {**BASE_ENV, **variant["env"]},
        "status": suffix,
    }


def main() -> int:
    ensure_dirs()
    baseline_full_ref_psnr = load_reference_psnr(BASELINE_FULL_REF_JSON)
    weak_hist_ref_psnr = load_reference_psnr(WEAK_HIST_REF_JSON)
    write_manifest(baseline_full_ref_psnr, weak_hist_ref_psnr)

    baseline_3k = baseline_3k_result()
    results_3k: list[dict] = []
    promoted_names: list[str] = []
    results_14k: list[dict] = []

    write_status(
        baseline_3k,
        results_3k,
        promoted_names,
        results_14k,
        running=[],
        baseline_full_ref_psnr=baseline_full_ref_psnr,
        weak_hist_ref_psnr=weak_hist_ref_psnr,
    )
    print(
        "[baseline3k] fine3000="
        f"{format_float(baseline_3k.get('fine_3000_psnr') or baseline_3k.get('coarse_3000_psnr'))}",
        flush=True,
    )

    future_map_3k = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for variant in VARIANTS:
            future = executor.submit(variant_3k_result, variant)
            future_map_3k[future] = variant
            running = [future_map_3k[f]["name"] for f in future_map_3k if not f.done()]
            write_status(
                baseline_3k,
                results_3k,
                promoted_names,
                results_14k,
                running=running,
                baseline_full_ref_psnr=baseline_full_ref_psnr,
                weak_hist_ref_psnr=weak_hist_ref_psnr,
            )

        for future in as_completed(future_map_3k):
            variant = future_map_3k[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = make_exception_result("3k", variant, f"exception_{type(exc).__name__}")
            results_3k.append(result)
            running = [future_map_3k[f]["name"] for f in future_map_3k if not f.done()]
            write_status(
                baseline_3k,
                results_3k,
                promoted_names,
                results_14k,
                running=running,
                baseline_full_ref_psnr=baseline_full_ref_psnr,
                weak_hist_ref_psnr=weak_hist_ref_psnr,
            )
            print(
                f"[done-3k] {result['name']} fine3000={format_float(result.get('fine_3000_psnr'))} "
                f"status={result.get('status')}",
                flush=True,
            )

    baseline_gate = baseline_3k.get("fine_3000_psnr") or baseline_3k.get("coarse_3000_psnr")
    if baseline_gate is None:
        baseline_gate = float("-inf")

    promoted_pool = [
        item
        for item in results_3k
        if item.get("fine_3000_psnr") is not None and item["fine_3000_psnr"] > baseline_gate
    ]
    promoted_pool.sort(key=lambda item: item["fine_3000_psnr"], reverse=True)
    promoted = promoted_pool[:PROMOTE_TOP_K] if PROMOTE_TOP_K > 0 else promoted_pool
    promoted_names = [item["name"] for item in promoted]
    write_status(
        baseline_3k,
        results_3k,
        promoted_names,
        results_14k,
        running=[],
        baseline_full_ref_psnr=baseline_full_ref_psnr,
        weak_hist_ref_psnr=weak_hist_ref_psnr,
    )
    print(
        f"[gate] promoted_topk={len(promoted_names)} "
        f"over_baseline={len(promoted_pool)} names={promoted_names if promoted_names else 'none'}",
        flush=True,
    )

    if promoted:
        variant_map = {item["name"]: item for item in VARIANTS}
        full_variants = []
        for idx, name in enumerate(promoted_names):
            variant = dict(variant_map[name])
            # Rebalance full-run scheduling to alternate GPUs by submission order.
            variant["gpu"] = idx % MAX_WORKERS
            variant["port"] = 7901 + idx
            full_variants.append(variant)
        future_map_14k = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for variant in full_variants:
                future = executor.submit(variant_full_result, variant)
                future_map_14k[future] = variant
                running = [future_map_14k[f]["name"] for f in future_map_14k if not f.done()]
                write_status(
                    baseline_3k,
                    results_3k,
                    promoted_names,
                    results_14k,
                    running=running,
                    baseline_full_ref_psnr=baseline_full_ref_psnr,
                    weak_hist_ref_psnr=weak_hist_ref_psnr,
                )

            for future in as_completed(future_map_14k):
                variant = future_map_14k[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    result = make_exception_result("14k", variant, f"exception_{type(exc).__name__}")
                results_14k.append(result)
                running = [future_map_14k[f]["name"] for f in future_map_14k if not f.done()]
                write_status(
                    baseline_3k,
                    results_3k,
                    promoted_names,
                    results_14k,
                    running=running,
                    baseline_full_ref_psnr=baseline_full_ref_psnr,
                    weak_hist_ref_psnr=weak_hist_ref_psnr,
                )
                print(
                    f"[done-14k] {result['name']} full={format_float(result.get('full_psnr'))} "
                    f"status={result.get('status')}",
                    flush=True,
                )

    write_status(
        baseline_3k,
        results_3k,
        promoted_names,
        results_14k,
        running=[],
        baseline_full_ref_psnr=baseline_full_ref_psnr,
        weak_hist_ref_psnr=weak_hist_ref_psnr,
    )

    best_3k = sorted(
        results_3k,
        key=lambda item: item.get("fine_3000_psnr") if item.get("fine_3000_psnr") is not None else -1.0,
        reverse=True,
    )
    if best_3k:
        print(
            f"[best-3k] {best_3k[0]['name']} fine3000={format_float(best_3k[0].get('fine_3000_psnr'))}",
            flush=True,
        )
    best_14k = sorted(
        results_14k,
        key=lambda item: item.get("full_psnr") if item.get("full_psnr") is not None else -1.0,
        reverse=True,
    )
    if best_14k:
        print(
            f"[best-14k] {best_14k[0]['name']} full_psnr={format_float(best_14k[0].get('full_psnr'))}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

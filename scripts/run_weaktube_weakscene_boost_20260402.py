#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_BASELINE_SH = REPO_ROOT / "scripts" / "train_baseline.sh"
TRAIN_TUBE_SH = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_TUBE_SH = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"

REPORT_DIR = REPO_ROOT / "reports" / "weakscene_boost_20260402"
STATUS_JSON = REPORT_DIR / "queue_status.json"
LEADERBOARD_MD = REPORT_DIR / "leaderboard.md"
MANIFEST_JSON = REPORT_DIR / "manifest.json"

GPU_LIST = [int(x.strip()) for x in os.environ.get("WEAKSCENE_GPUS", "0,1,2,3,4,5").split(",") if x.strip()]
MAX_WORKERS = int(os.environ.get("WEAKSCENE_MAX_WORKERS", str(len(GPU_LIST) if GPU_LIST else 1)))
PROMOTE_PER_SCENE = int(os.environ.get("WEAKSCENE_PROMOTE_PER_SCENE", "2"))
PROMOTE_FALLBACK_TOP1 = int(os.environ.get("WEAKSCENE_PROMOTE_FALLBACK_TOP1", "0"))
RUN_TAG = os.environ.get("WEAKSCENE_RUN_TAG", "20260402")

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

TARGET_SCENES: dict[str, dict[str, Any]] = {
    "americano": {
        "dataset": "hypernerf",
        "scene": "misc/americano",
        "baseline_full_psnr": 30.5351,
    },
    "cut_roasted_beef": {
        "dataset": "dynerf",
        "scene": "cut_roasted_beef",
        "baseline_full_psnr": 15.1763,
    },
    "espresso": {
        "dataset": "hypernerf",
        "scene": "misc/espresso",
        "baseline_full_psnr": 25.4668,
    },
    "torchchocolate": {
        "dataset": "hypernerf",
        "scene": "interp/torchocolate",
        "baseline_full_psnr": 27.1864,
    },
    "split_cookie": {
        "dataset": "hypernerf",
        "scene": "misc/split-cookie",
        "baseline_full_psnr": 31.3522,
    },
}

BASE_ENV = {
    "TEMPORAL_TUBE_SAMPLES": "3",
    "TEMPORAL_TUBE_SPAN": "0.40",
    "TEMPORAL_TUBE_SIGMA": "0.34",
    "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
    "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
    "TEMPORAL_DRIFT_SCALE": "1.0",
    "TEMPORAL_GATE_MIX": "1.0",
    "TEMPORAL_DRIFT_MIX": "1.0",
    "TEMPORAL_ACCELERATION_ENABLED": "0",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
    "TEMPORAL_LR_INIT": "0.00012",
    "TEMPORAL_LR_FINAL": "0.000012",
    "TEMPORAL_LR_DELAY_MULT": "0.01",
}

VARIANTS = [
    {
        "name": "c040_s034_cov005_w100_lrlow",
        "description": "Current benchmark-12 best control.",
        "env": {},
    },
    {
        "name": "c040_s030_cov005_w100_lrhi",
        "description": "Split-cookie 3k winner family, sharper sigma + higher LR.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_LR_INIT": "0.00020",
            "TEMPORAL_LR_FINAL": "0.000020",
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
        "name": "c040_s032_cov005_w100_stdlr",
        "description": "Historical center support with standard tube LR.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_LR_INIT": "0.00016",
            "TEMPORAL_LR_FINAL": "0.000016",
        },
    },
    {
        "name": "c038_s032_cov005_w100",
        "description": "Slightly shorter support.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.38",
            "TEMPORAL_TUBE_SIGMA": "0.32",
        },
    },
    {
        "name": "c042_s032_cov005_w100",
        "description": "Slightly wider support.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.42",
            "TEMPORAL_TUBE_SIGMA": "0.32",
        },
    },
    {
        "name": "c040_s032_cov004_w100",
        "description": "Lower covariance mix.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.04",
        },
    },
    {
        "name": "c040_s032_cov006_w100",
        "description": "Higher covariance mix.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.06",
        },
    },
    {
        "name": "c040_s032_cov005_d095",
        "description": "Reduced drift scale.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_DRIFT_SCALE": "0.95",
        },
    },
    {
        "name": "c040_s032_cov005_d105",
        "description": "Increased drift scale.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_DRIFT_SCALE": "1.05",
        },
    },
    {
        "name": "c040_s032_cov005_gm110_dm095",
        "description": "Stronger gate mix with lighter drift mix.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_GATE_MIX": "1.10",
            "TEMPORAL_DRIFT_MIX": "0.95",
        },
    },
    {
        "name": "c035_s028_cov008_w120",
        "description": "Round-1 robust branch.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.35",
            "TEMPORAL_TUBE_SIGMA": "0.28",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.08",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.20",
        },
    },
    {
        "name": "c035_s026_cov007_w125",
        "description": "Round-1 aggressive sharp branch.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.35",
            "TEMPORAL_TUBE_SIGMA": "0.26",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.07",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.25",
        },
    },
    {
        "name": "s5_span060_sigma045_cov050_w100",
        "description": "Moderate 5-sample branch for transferability check.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "5",
            "TEMPORAL_TUBE_SPAN": "0.60",
            "TEMPORAL_TUBE_SIGMA": "0.45",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.50",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
        },
    },
]


def scene_name(scene_path: str) -> str:
    return scene_path.split("/")[-1]


def run_dir_for(namespace: str, dataset: str, scene_path: str) -> Path:
    return REPO_ROOT / "runs" / namespace / dataset / scene_name(scene_path)


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def parse_train_metrics(train_log: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
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


def parse_results_metrics(run_dir: Path, iteration: str) -> dict[str, float | None]:
    payload = {
        "render_psnr": None,
        "render_ssim": None,
        "render_lpips": None,
    }
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
    payload["render_lpips"] = float_or_none(block.get("LPIPS-vgg"))
    return payload


def parse_collect_metrics(run_dir: Path) -> dict[str, float | None]:
    payload = {
        "time_seconds": None,
        "train_seconds": None,
        "render_seconds": None,
        "fps": None,
        "storage_mb": None,
    }
    metrics = load_json(run_dir / "metrics.json") or {}
    train_s = float_or_none(metrics.get("train_seconds"))
    render_s = float_or_none(metrics.get("render_seconds"))
    fps = float_or_none(metrics.get("render_fps"))
    storage_bytes = float_or_none(metrics.get("storage_bytes"))

    if train_s is not None or render_s is not None:
        payload["time_seconds"] = float(train_s or 0.0) + float(render_s or 0.0)
    payload["train_seconds"] = train_s
    payload["render_seconds"] = render_s
    payload["fps"] = fps

    if storage_bytes is not None:
        payload["storage_mb"] = storage_bytes / (1024.0 * 1024.0)
    else:
        ckpt = run_dir / "point_cloud" / "iteration_14000" / "point_cloud.ply"
        if ckpt.exists():
            payload["storage_mb"] = ckpt.stat().st_size / (1024.0 * 1024.0)
    return payload


def run_cmd(cmd: list[str], env: dict[str, str], label: str) -> int:
    print(f"[run] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return proc.returncode


def baseline_namespace(scene_key: str) -> str:
    return f"baseline_weakscene3k_{RUN_TAG}_{scene_key}"


def ours_3k_namespace(scene_key: str, variant_name: str) -> str:
    return f"stellar_tube_weakscene3k_{RUN_TAG}_{scene_key}_{variant_name}"


def ours_14k_namespace(scene_key: str, variant_name: str) -> str:
    return f"stellar_tube_weakscene14k_{RUN_TAG}_{scene_key}_{variant_name}"


def baseline_3k_result(scene_key: str, gpu: int, port: int) -> dict[str, Any]:
    cfg = TARGET_SCENES[scene_key]
    dataset = cfg["dataset"]
    scene_path = cfg["scene"]
    namespace = baseline_namespace(scene_key)
    run_dir = run_dir_for(namespace, dataset, scene_path)

    result: dict[str, Any] = {
        "kind": "baseline_3k",
        "scene_key": scene_key,
        "dataset": dataset,
        "scene": scene_path,
        "namespace": namespace,
        "gpu": gpu,
        "port": port,
        "run_dir": str(run_dir),
    }
    result.update(parse_train_metrics(run_dir / "train.log"))

    if result.get("fine_3000_psnr") is not None:
        result["status"] = "complete_cached"
        return result

    env = dict(os.environ)
    env["GS_RUN_NAMESPACE"] = namespace
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        "bash",
        str(TRAIN_BASELINE_SH),
        dataset,
        scene_path,
        *COMMON_3K_ARGS,
        "--port",
        str(port),
    ]
    rc = run_cmd(cmd, env, f"baseline3k-{scene_key}")
    result.update(parse_train_metrics(run_dir / "train.log"))
    result["status"] = "complete" if rc == 0 else f"train_failed_{rc}"
    return result


def variant_3k_result(scene_key: str, variant: dict[str, Any], gpu: int, port: int, baseline_3k_psnr: float | None) -> dict[str, Any]:
    cfg = TARGET_SCENES[scene_key]
    dataset = cfg["dataset"]
    scene_path = cfg["scene"]
    namespace = ours_3k_namespace(scene_key, variant["name"])
    run_dir = run_dir_for(namespace, dataset, scene_path)

    result: dict[str, Any] = {
        "kind": "tube_3k",
        "scene_key": scene_key,
        "dataset": dataset,
        "scene": scene_path,
        "variant": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "gpu": gpu,
        "port": port,
        "run_dir": str(run_dir),
        "env": {**BASE_ENV, **variant["env"]},
        "baseline_3k_psnr": baseline_3k_psnr,
    }
    result.update(parse_train_metrics(run_dir / "train.log"))

    fine = result.get("fine_3000_psnr")
    result["delta_vs_baseline3k"] = None if fine is None or baseline_3k_psnr is None else float(fine) - float(baseline_3k_psnr)

    if fine is not None:
        result["status"] = "complete_cached"
        return result

    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = ["bash", str(TRAIN_TUBE_SH), dataset, scene_path, *COMMON_3K_ARGS]
    rc = run_cmd(cmd, env, f"tube3k-{scene_key}-{variant['name']}")
    result.update(parse_train_metrics(run_dir / "train.log"))
    fine = result.get("fine_3000_psnr")
    result["delta_vs_baseline3k"] = None if fine is None or baseline_3k_psnr is None else float(fine) - float(baseline_3k_psnr)
    result["status"] = "complete" if rc == 0 else f"train_failed_{rc}"
    return result


def variant_14k_result(scene_key: str, variant: dict[str, Any], gpu: int, port: int) -> dict[str, Any]:
    cfg = TARGET_SCENES[scene_key]
    dataset = cfg["dataset"]
    scene_path = cfg["scene"]
    baseline_full_psnr = float(cfg.get("baseline_full_psnr") or 0.0)
    namespace = ours_14k_namespace(scene_key, variant["name"])
    run_dir = run_dir_for(namespace, dataset, scene_path)

    result: dict[str, Any] = {
        "kind": "tube_14k",
        "scene_key": scene_key,
        "dataset": dataset,
        "scene": scene_path,
        "variant": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "gpu": gpu,
        "port": port,
        "run_dir": str(run_dir),
        "env": {**BASE_ENV, **variant["env"]},
        "baseline_full_psnr": baseline_full_psnr,
    }
    result.update(parse_train_metrics(run_dir / "train.log"))
    result.update(parse_results_metrics(run_dir, "14000"))
    result.update(parse_collect_metrics(run_dir))

    eval_psnr = result.get("render_psnr")
    result["delta_vs_baseline_full"] = None if eval_psnr is None else float(eval_psnr) - baseline_full_psnr

    if result.get("test_14000_psnr") is not None and eval_psnr is not None:
        result["status"] = "complete_cached"
        return result

    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ckpt = run_dir / "point_cloud" / "iteration_14000" / "point_cloud.ply"
    if not ckpt.exists():
        train_rc = run_cmd(["bash", str(TRAIN_TUBE_SH), dataset, scene_path, *COMMON_14K_ARGS], env, f"tube14k-train-{scene_key}-{variant['name']}")
        result.update(parse_train_metrics(run_dir / "train.log"))
        if train_rc != 0:
            result["status"] = f"train_failed_{train_rc}"
            return result

    eval_rc = run_cmd(["bash", str(EVAL_TUBE_SH), dataset, scene_path], env, f"tube14k-eval-{scene_key}-{variant['name']}")
    result.update(parse_results_metrics(run_dir, "14000"))
    result.update(parse_collect_metrics(run_dir))
    eval_psnr = result.get("render_psnr")
    result["delta_vs_baseline_full"] = None if eval_psnr is None else float(eval_psnr) - baseline_full_psnr
    if eval_rc != 0:
        result["status"] = f"eval_failed_{eval_rc}"
        return result

    result["status"] = "complete"
    return result


def write_manifest() -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "report_dir": str(REPORT_DIR),
        "gpus": GPU_LIST,
        "max_workers": MAX_WORKERS,
        "promote_per_scene": PROMOTE_PER_SCENE,
        "promote_fallback_top1": PROMOTE_FALLBACK_TOP1,
        "target_scenes": TARGET_SCENES,
        "base_env": BASE_ENV,
        "variants": VARIANTS,
        "common_3k_args": COMMON_3K_ARGS,
        "common_14k_args": COMMON_14K_ARGS,
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_status(
    baseline_3k: dict[str, dict[str, Any]],
    results_3k: list[dict[str, Any]],
    promoted: dict[str, list[str]],
    results_14k: list[dict[str, Any]],
    running: list[str],
) -> None:
    STATUS_JSON.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "gpus": GPU_LIST,
                "max_workers": MAX_WORKERS,
                "promote_per_scene": PROMOTE_PER_SCENE,
                "promote_fallback_top1": PROMOTE_FALLBACK_TOP1,
                "target_scenes": TARGET_SCENES,
                "running": running,
                "baseline_3k": baseline_3k,
                "results_3k": results_3k,
                "promoted": promoted,
                "results_14k": results_14k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# WeakScene Boost Sweep (2026-04-02)")
    lines.append("")
    lines.append(f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`")
    lines.append(f"- GPUs: `{GPU_LIST}`")
    lines.append(f"- Max workers: `{MAX_WORKERS}`")
    lines.append(f"- Promote per scene: `{PROMOTE_PER_SCENE}`")
    lines.append(f"- Running: `{', '.join(running) if running else 'none'}`")
    lines.append("")

    lines.append("## Baseline 3k")
    lines.append("")
    lines.append("| Scene | Baseline3k fine PSNR | Baseline3k coarse PSNR | Status |")
    lines.append("| --- | ---: | ---: | --- |")
    for scene_key in TARGET_SCENES:
        b = baseline_3k.get(scene_key, {})
        lines.append(
            f"| `{scene_key}` | {format_float(float_or_none(b.get('fine_3000_psnr')))} | "
            f"{format_float(float_or_none(b.get('coarse_3000_psnr')))} | {b.get('status', 'pending')} |"
        )

    for scene_key in TARGET_SCENES:
        lines.append("")
        lines.append(f"## 3k Leaderboard - {scene_key}")
        lines.append("")
        lines.append("| Rank | Variant | Fine3000 | Delta vs Baseline3k | Status |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        scene_rows = [r for r in results_3k if r.get("scene_key") == scene_key]
        scene_rows.sort(key=lambda x: x.get("fine_3000_psnr") if x.get("fine_3000_psnr") is not None else -1.0, reverse=True)
        for i, row in enumerate(scene_rows, 1):
            lines.append(
                f"| {i} | `{row.get('variant')}` | {format_float(float_or_none(row.get('fine_3000_psnr')))} | "
                f"{format_float(float_or_none(row.get('delta_vs_baseline3k')))} | {row.get('status', 'n/a')} |"
            )
        if not scene_rows:
            lines.append("| n/a | n/a | n/a | n/a | pending |")

    lines.append("")
    lines.append("## Promotions")
    lines.append("")
    lines.append("| Scene | Promoted Variants |")
    lines.append("| --- | --- |")
    for scene_key in TARGET_SCENES:
        names = promoted.get(scene_key, [])
        lines.append(f"| `{scene_key}` | `{', '.join(names) if names else 'none'}` |")

    lines.append("")
    lines.append("## 14k Results")
    lines.append("")
    lines.append("| Scene | Variant | Eval PSNR | Delta vs 4DGS Full | SSIM | LPIPS | Time(s) | FPS | Storage(MB) | Status |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    full_rows = sorted(
        results_14k,
        key=lambda x: (
            x.get("scene_key", ""),
            -(x.get("render_psnr") if x.get("render_psnr") is not None else -1.0),
        ),
    )
    for row in full_rows:
        lines.append(
            f"| `{row.get('scene_key')}` | `{row.get('variant')}` | {format_float(float_or_none(row.get('render_psnr')))} | "
            f"{format_float(float_or_none(row.get('delta_vs_baseline_full')))} | {format_float(float_or_none(row.get('render_ssim')))} | "
            f"{format_float(float_or_none(row.get('render_lpips')))} | {format_float(float_or_none(row.get('time_seconds')))} | "
            f"{format_float(float_or_none(row.get('fps')))} | {format_float(float_or_none(row.get('storage_mb')))} | {row.get('status', 'n/a')} |"
        )
    if not full_rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | pending |")

    LEADERBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not GPU_LIST:
        raise SystemExit("No GPUs specified via WEAKSCENE_GPUS")

    ensure_dirs()
    write_manifest()

    baseline_3k: dict[str, dict[str, Any]] = {}
    results_3k: list[dict[str, Any]] = []
    promoted: dict[str, list[str]] = {k: [] for k in TARGET_SCENES}
    results_14k: list[dict[str, Any]] = []
    lock = threading.Lock()

    write_status(baseline_3k, results_3k, promoted, results_14k, running=[])

    baseline_specs: list[dict[str, Any]] = []
    for idx, scene_key in enumerate(TARGET_SCENES):
        baseline_specs.append(
            {
                "scene_key": scene_key,
                "gpu": GPU_LIST[idx % len(GPU_LIST)],
                "port": 8600 + idx,
            }
        )

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(baseline_specs))) as executor:
        future_map = {
            executor.submit(
                baseline_3k_result,
                spec["scene_key"],
                spec["gpu"],
                spec["port"],
            ): spec
            for spec in baseline_specs
        }
        running = [f"baseline3k-{future_map[f]['scene_key']}@gpu{future_map[f]['gpu']}" for f in future_map]
        write_status(baseline_3k, results_3k, promoted, results_14k, running=running)
        for future in as_completed(future_map):
            spec = future_map[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "scene_key": spec["scene_key"],
                    "gpu": spec["gpu"],
                    "port": spec["port"],
                    "status": f"exception_{type(exc).__name__}",
                }
            with lock:
                baseline_3k[spec["scene_key"]] = res
            running = [
                f"baseline3k-{future_map[f]['scene_key']}@gpu{future_map[f]['gpu']}"
                for f in future_map
                if not f.done()
            ]
            write_status(baseline_3k, results_3k, promoted, results_14k, running=running)
            print(
                f"[done-baseline3k] {spec['scene_key']} fine3000={format_float(float_or_none(res.get('fine_3000_psnr')))} "
                f"status={res.get('status')}",
                flush=True,
            )

    tasks_3k: list[dict[str, Any]] = []
    for scene_key in TARGET_SCENES:
        baseline_scene = baseline_3k.get(scene_key, {})
        baseline_psnr = float_or_none(baseline_scene.get("fine_3000_psnr"))
        if baseline_psnr is None:
            baseline_psnr = float_or_none(baseline_scene.get("coarse_3000_psnr"))
        for variant in VARIANTS:
            idx = len(tasks_3k)
            tasks_3k.append(
                {
                    "scene_key": scene_key,
                    "variant": variant,
                    "gpu": GPU_LIST[idx % len(GPU_LIST)],
                    "port": 9000 + idx,
                    "baseline_3k_psnr": baseline_psnr,
                }
            )

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks_3k))) as executor:
        future_map = {
            executor.submit(
                variant_3k_result,
                task["scene_key"],
                task["variant"],
                task["gpu"],
                task["port"],
                task["baseline_3k_psnr"],
            ): task
            for task in tasks_3k
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                res = future.result()
            except Exception as exc:
                res = {
                    "scene_key": task["scene_key"],
                    "variant": task["variant"]["name"],
                    "gpu": task["gpu"],
                    "port": task["port"],
                    "status": f"exception_{type(exc).__name__}",
                }
            results_3k.append(res)
            running = [
                f"{future_map[f]['scene_key']}/{future_map[f]['variant']['name']}@gpu{future_map[f]['gpu']}"
                for f in future_map
                if not f.done()
            ]
            write_status(baseline_3k, results_3k, promoted, results_14k, running=running)
            print(
                f"[done-3k] {res.get('scene_key')} {res.get('variant')} fine3000={format_float(float_or_none(res.get('fine_3000_psnr')))} "
                f"delta={format_float(float_or_none(res.get('delta_vs_baseline3k')))} status={res.get('status')}",
                flush=True,
            )

    for scene_key in TARGET_SCENES:
        baseline_scene = baseline_3k.get(scene_key, {})
        baseline_psnr = float_or_none(baseline_scene.get("fine_3000_psnr"))
        if baseline_psnr is None:
            baseline_psnr = float_or_none(baseline_scene.get("coarse_3000_psnr"))
        scene_rows = [
            r
            for r in results_3k
            if r.get("scene_key") == scene_key and float_or_none(r.get("fine_3000_psnr")) is not None
        ]
        over = []
        if baseline_psnr is not None:
            over = [r for r in scene_rows if float_or_none(r.get("fine_3000_psnr")) is not None and float(r["fine_3000_psnr"]) > baseline_psnr]
        over.sort(key=lambda x: float_or_none(x.get("fine_3000_psnr")) or -1.0, reverse=True)
        promoted_names = [r["variant"] for r in over[:PROMOTE_PER_SCENE]]

        if not promoted_names and PROMOTE_FALLBACK_TOP1 == 1 and scene_rows:
            scene_rows.sort(key=lambda x: float_or_none(x.get("fine_3000_psnr")) or -1.0, reverse=True)
            promoted_names = [scene_rows[0]["variant"]]

        promoted[scene_key] = promoted_names

    write_status(baseline_3k, results_3k, promoted, results_14k, running=[])

    variant_by_name = {v["name"]: v for v in VARIANTS}
    tasks_14k: list[dict[str, Any]] = []
    for scene_key in TARGET_SCENES:
        for name in promoted.get(scene_key, []):
            variant = variant_by_name.get(name)
            if variant is None:
                continue
            idx = len(tasks_14k)
            tasks_14k.append(
                {
                    "scene_key": scene_key,
                    "variant": variant,
                    "gpu": GPU_LIST[idx % len(GPU_LIST)],
                    "port": 9800 + idx,
                }
            )

    if tasks_14k:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks_14k))) as executor:
            future_map = {
                executor.submit(
                    variant_14k_result,
                    task["scene_key"],
                    task["variant"],
                    task["gpu"],
                    task["port"],
                ): task
                for task in tasks_14k
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "scene_key": task["scene_key"],
                        "variant": task["variant"]["name"],
                        "gpu": task["gpu"],
                        "port": task["port"],
                        "status": f"exception_{type(exc).__name__}",
                    }
                results_14k.append(res)
                running = [
                    f"{future_map[f]['scene_key']}/{future_map[f]['variant']['name']}@gpu{future_map[f]['gpu']}"
                    for f in future_map
                    if not f.done()
                ]
                write_status(baseline_3k, results_3k, promoted, results_14k, running=running)
                print(
                    f"[done-14k] {res.get('scene_key')} {res.get('variant')} eval_psnr={format_float(float_or_none(res.get('render_psnr')))} "
                    f"delta_vs_4dgs={format_float(float_or_none(res.get('delta_vs_baseline_full')))} status={res.get('status')}",
                    flush=True,
                )

    write_status(baseline_3k, results_3k, promoted, results_14k, running=[])

    scene_best = {}
    for scene_key in TARGET_SCENES:
        rows = [r for r in results_14k if r.get("scene_key") == scene_key and float_or_none(r.get("render_psnr")) is not None]
        rows.sort(key=lambda x: float_or_none(x.get("render_psnr")) or -1.0, reverse=True)
        if rows:
            scene_best[scene_key] = rows[0]

    print("[summary] best per scene (14k):", flush=True)
    for scene_key in TARGET_SCENES:
        row = scene_best.get(scene_key)
        if not row:
            print(f"  - {scene_key}: no 14k result", flush=True)
            continue
        print(
            f"  - {scene_key}: {row.get('variant')} eval_psnr={format_float(float_or_none(row.get('render_psnr')))} "
            f"delta_vs_4dgs={format_float(float_or_none(row.get('delta_vs_baseline_full')))}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

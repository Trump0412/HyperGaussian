#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"
COMMON_SH = REPO_ROOT / "scripts" / "common.sh"

REPORT_DIR = REPO_ROOT / "reports" / "weaktube_full6_20260328"
STATUS_JSON = REPORT_DIR / "queue_status.json"
LEADERBOARD_MD = REPORT_DIR / "leaderboard.md"
MANIFEST_JSON = REPORT_DIR / "manifest.json"
QUEUE_LOG = REPORT_DIR / "queue.log"

DATASET = "hypernerf"
SCENE = "misc/split-cookie"
SCENE_NAME = SCENE.split("/")[-1]
MAX_WORKERS = 6

HISTORICAL_REFERENCE = {
    "name": "stellar_tube_split-cookie_compare5k_weak",
    "run_dir": str(
        REPO_ROOT / "runs" / "stellar_tube_split-cookie_compare5k_weak" / DATASET / SCENE_NAME
    ),
    "config_path": str(
        REPO_ROOT
        / "runs"
        / "stellar_tube_split-cookie_compare5k_weak"
        / DATASET
        / SCENE_NAME
        / "config.yaml"
    ),
    "fine_3000_psnr": 28.933096156400794,
    "final_14000_psnr": 30.526937148150274,
    "full_psnr": 31.164094725651527,
    "full_ssim": 0.9009000267555465,
    "full_lpips": 0.1811073250067768,
    "config": {
        "temporal_tube_samples": 3,
        "temporal_tube_span": 0.5,
        "temporal_tube_sigma": 0.35,
        "temporal_tube_covariance_mix": 0.05,
        "temporal_acceleration_enabled": 0,
        "temporal_drift_scale": 1.0,
        "temporal_gate_mix": 1.0,
        "temporal_drift_mix": 1.0,
    },
}

BASE_ENV = {
    "TEMPORAL_TUBE_SAMPLES": "3",
    "TEMPORAL_TUBE_SPAN": "0.50",
    "TEMPORAL_TUBE_SIGMA": "0.35",
    "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
    "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
    "TEMPORAL_DRIFT_SCALE": "1.0",
    "TEMPORAL_GATE_MIX": "1.0",
    "TEMPORAL_DRIFT_MIX": "1.0",
    "TEMPORAL_ACCELERATION_ENABLED": "0",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
}

COMMON_TRAIN_ARGS = [
    "--iterations", "14000",
    "--coarse_iterations", "3000",
    "--test_iterations", "3000", "7000", "14000",
    "--save_iterations", "7000", "14000",
    "--checkpoint_iterations", "7000", "14000",
]

VARIANTS = [
    {
        "name": "histplus_span045",
        "description": "Historical best recipe plus slightly shorter support to absorb the queue win from span shortening.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.45",
        },
        "gpu": 0,
        "port": 6201,
    },
    {
        "name": "histplus_sigma030",
        "description": "Historical best recipe plus a sharper temporal kernel to absorb the queue win from sigma sharpening.",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.30",
        },
        "gpu": 0,
        "port": 6202,
    },
    {
        "name": "histplus_span045_sigma030",
        "description": "Directly combine the two strongest queue ideas on top of the historical best weak-tube recipe.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.45",
            "TEMPORAL_TUBE_SIGMA": "0.30",
        },
        "gpu": 1,
        "port": 6203,
    },
    {
        "name": "histplus_span040_sigma030",
        "description": "A slightly more aggressive concentrated-support version around the combined recipe.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.40",
            "TEMPORAL_TUBE_SIGMA": "0.30",
        },
        "gpu": 1,
        "port": 6204,
    },
    {
        "name": "histplus_span045_sigma032",
        "description": "A milder combined version that keeps the shorter support but softens the sharpened sigma slightly.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.45",
            "TEMPORAL_TUBE_SIGMA": "0.32",
        },
        "gpu": 2,
        "port": 6205,
    },
    {
        "name": "histplus_span040_sigma032",
        "description": "An aggressive span reduction paired with a milder sigma refinement for a second combined branch.",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.40",
            "TEMPORAL_TUBE_SIGMA": "0.32",
        },
        "gpu": 2,
        "port": 6206,
    },
]


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def append_queue_log(message: str) -> None:
    with QUEUE_LOG.open("a") as f:
        f.write(message + "\n")


def run_dir_for(namespace: str) -> Path:
    return REPO_ROOT / "runs" / namespace / DATASET / SCENE_NAME


def parse_train_metrics(train_log: Path) -> dict:
    result = {
        "coarse_test_psnr": None,
        "fine_3000_test_psnr": None,
        "test_7000_psnr": None,
        "test_14000_psnr": None,
        "coarse_test_l1": None,
        "fine_3000_test_l1": None,
        "test_7000_l1": None,
        "test_14000_l1": None,
        "status": "missing",
    }
    if not train_log.exists():
        return result
    text = train_log.read_text(errors="ignore")
    matches = re.findall(r"\[ITER (\d+)\] Evaluating test: L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+)", text)
    iter_3000_seen = 0
    for iter_str, l1_str, psnr_str in matches:
        step = int(iter_str)
        l1 = float(l1_str)
        psnr = float(psnr_str)
        if step == 3000:
            iter_3000_seen += 1
            if iter_3000_seen == 1:
                result["coarse_test_l1"] = l1
                result["coarse_test_psnr"] = psnr
            else:
                result["fine_3000_test_l1"] = l1
                result["fine_3000_test_psnr"] = psnr
        elif step == 7000:
            result["test_7000_l1"] = l1
            result["test_7000_psnr"] = psnr
        elif step == 14000:
            result["test_14000_l1"] = l1
            result["test_14000_psnr"] = psnr
    if "Training complete." in text:
        result["status"] = "trained"
    return result


def parse_results_json(run_dir: Path) -> dict:
    results_json = run_dir / "results.json"
    if not results_json.exists():
        return {"render_PSNR": None, "render_SSIM": None, "render_LPIPS-vgg": None}
    data = json.loads(results_json.read_text()).get("ours_14000", {})
    return {
        "render_PSNR": data.get("PSNR"),
        "render_SSIM": data.get("SSIM"),
        "render_LPIPS-vgg": data.get("LPIPS-vgg"),
    }


def parse_fullframe_metrics(run_dir: Path) -> dict:
    metrics_json = run_dir / "full_metrics_with_lpips_full14k.json"
    if not metrics_json.exists():
        return {"full_PSNR": None, "full_SSIM": None, "full_MS-SSIM": None, "full_LPIPS-vgg": None}
    data = json.loads(metrics_json.read_text())
    return {
        "full_PSNR": data.get("PSNR"),
        "full_SSIM": data.get("SSIM"),
        "full_MS-SSIM": data.get("MS-SSIM"),
        "full_LPIPS-vgg": data.get("LPIPS-vgg"),
    }


def current_result(variant: dict) -> dict:
    namespace = f"stellar_tube_full6_20260328_{variant['name']}"
    run_dir = run_dir_for(namespace)
    result = {
        "name": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "gpu": variant["gpu"],
        "port": variant["port"],
        "run_dir": str(run_dir),
        "env": {**BASE_ENV, **variant["env"]},
    }
    result.update(parse_train_metrics(run_dir / "train.log"))
    result.update(parse_results_json(run_dir))
    result.update(parse_fullframe_metrics(run_dir))
    return result


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_status(results: list[dict], running: list[str]) -> None:
    STATUS_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "scene": SCENE,
                "historical_reference": HISTORICAL_REFERENCE,
                "running": running,
                "results": results,
            },
            indent=2,
        )
    )

    ranked = sorted(
        results,
        key=lambda item: item["full_PSNR"] if item["full_PSNR"] is not None else -1.0,
        reverse=True,
    )
    lines = [
        "# WeakTube Full6 20260328",
        "",
        f"- Dataset/scene: `{DATASET} / {SCENE}`",
        f"- Historical reference run: `{HISTORICAL_REFERENCE['name']}`",
        f"- Historical reference fine 3000 PSNR: `{HISTORICAL_REFERENCE['fine_3000_psnr']:.4f}`",
        f"- Historical reference final 14000 PSNR: `{HISTORICAL_REFERENCE['final_14000_psnr']:.4f}`",
        f"- Historical reference full PSNR: `{HISTORICAL_REFERENCE['full_psnr']:.4f}`",
        f"- Running: `{', '.join(running) if running else 'none'}`",
        "",
        "## Variant Manifest",
        "",
    ]
    for variant in VARIANTS:
        merged_env = {**BASE_ENV, **variant["env"]}
        env_bits = ", ".join(
            [
                f"samples={merged_env['TEMPORAL_TUBE_SAMPLES']}",
                f"span={merged_env['TEMPORAL_TUBE_SPAN']}",
                f"sigma={merged_env['TEMPORAL_TUBE_SIGMA']}",
                f"covmix={merged_env['TEMPORAL_TUBE_COVARIANCE_MIX']}",
                f"accel={merged_env['TEMPORAL_ACCELERATION_ENABLED']}",
                f"gpu={variant['gpu']}",
            ]
        )
        lines.append(f"- `{variant['name']}`: {variant['description']} [{env_bits}]")
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Variant | GPU | Fine3000 | Test7000 | Test14000 | Full PSNR | Delta vs hist full | SSIM | LPIPS | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in ranked:
        delta_hist = (
            None
            if item["full_PSNR"] is None
            else item["full_PSNR"] - HISTORICAL_REFERENCE["full_psnr"]
        )
        lines.append(
            "| `{name}` | {gpu} | {fine3000} | {test7000} | {test14000} | {full_psnr} | {delta_hist} | {ssim} | {lpips} | {status} |".format(
                name=item["name"],
                gpu=item["gpu"],
                fine3000=format_float(item["fine_3000_test_psnr"]),
                test7000=format_float(item["test_7000_psnr"]),
                test14000=format_float(item["test_14000_psnr"]),
                full_psnr=format_float(item["full_PSNR"]),
                delta_hist=format_float(delta_hist),
                ssim=format_float(item["full_SSIM"]),
                lpips=format_float(item["full_LPIPS-vgg"]),
                status=item["status"],
            )
        )
    LEADERBOARD_MD.write_text("\n".join(lines) + "\n")


def run_bash(cmd: str, env: dict, cwd: Path) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=str(cwd), env=env, check=True)


def run_variant(variant: dict) -> dict:
    namespace = f"stellar_tube_full6_20260328_{variant['name']}"
    run_dir = run_dir_for(namespace)
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(variant["port"])
    env["CUDA_VISIBLE_DEVICES"] = str(variant["gpu"])

    append_queue_log(
        f"[queue] start {variant['name']} gpu={variant['gpu']} port={variant['port']} env={json.dumps({**BASE_ENV, **variant['env']}, sort_keys=True)}"
    )

    train_cmd = [str(TRAIN_SCRIPT), DATASET, SCENE, *COMMON_TRAIN_ARGS]
    subprocess.run(["bash", *train_cmd], cwd=str(REPO_ROOT), env=env, check=True)

    append_queue_log(f"[queue] eval {variant['name']}")
    subprocess.run(["bash", str(EVAL_SCRIPT), DATASET, SCENE], cwd=str(REPO_ROOT), env=env, check=True)

    append_queue_log(f"[queue] fullframe {variant['name']}")
    fullframe_cmd = (
        f"source '{COMMON_SH}' && "
        f"gs_python '{REPO_ROOT}/scripts/fullframe_metrics.py' "
        f"--run-dir '{run_dir}' --with-lpips "
        f"--out-name 'full_metrics_with_lpips_full14k.json'"
    )
    run_bash(fullframe_cmd, env=env, cwd=REPO_ROOT)

    append_queue_log(f"[queue] done {variant['name']}")
    return current_result(variant)


def main() -> int:
    ensure_dirs()
    manifest = {
        "dataset": DATASET,
        "scene": SCENE,
        "historical_reference": HISTORICAL_REFERENCE,
        "base_env": BASE_ENV,
        "common_train_args": COMMON_TRAIN_ARGS,
        "variants": VARIANTS,
        "launched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
    QUEUE_LOG.write_text("")

    results = [current_result(variant) for variant in VARIANTS]
    running = [variant["name"] for variant in VARIANTS]
    write_status(results, running)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_variant, variant): variant for variant in VARIANTS}
        for future in as_completed(futures):
            variant = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                append_queue_log(f"[queue] fail {variant['name']} -> {exc}")
                result = current_result(variant)
                result["status"] = f"failed: {exc}"
            results = [current_result(v) for v in VARIANTS]
            for idx, existing in enumerate(results):
                if existing["name"] == result["name"]:
                    results[idx] = result
                    break
            running = [futures[f]["name"] for f in futures if not f.done()]
            write_status(results, running)

    return 0


if __name__ == "__main__":
    sys.exit(main())

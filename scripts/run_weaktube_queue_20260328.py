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
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"
REPORT_DIR = REPO_ROOT / "reports" / "weaktube_queue_20260328"
STATUS_JSON = REPORT_DIR / "queue_status.json"
LEADERBOARD_MD = REPORT_DIR / "leaderboard.md"
MANIFEST_JSON = REPORT_DIR / "manifest.json"

DATASET = "hypernerf"
SCENE = "misc/split-cookie"
SCENE_NAME = SCENE.split("/")[-1]
MAX_WORKERS = 2

REFERENCE_BASELINE_FINE_3000 = 28.5610855326933
REFERENCE_WEAK_FINE_3000 = 28.933096156400794
REFERENCE_WEAK_FULL_PSNR = 31.164094725651527

COMMON_TRAIN_ARGS = [
    "--iterations", "3000",
    "--coarse_iterations", "3000",
    "--test_iterations", "3000",
    "--save_iterations", "3000",
    "--checkpoint_iterations", "3000",
]

BASE_ENV = {
    "TEMPORAL_TUBE_SAMPLES": "5",
    "TEMPORAL_TUBE_SPAN": "1.0",
    "TEMPORAL_TUBE_SIGMA": "0.75",
    "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
    "TEMPORAL_TUBE_COVARIANCE_MIX": "1.0",
    "TEMPORAL_DRIFT_SCALE": "1.0",
    "TEMPORAL_ACCELERATION_ENABLED": "1",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
}

VARIANTS = [
    {
        "name": "weak_base_confirm",
        "description": "Reproduce the current weak-tube baseline under a 3000-step gate.",
        "env": {},
        "extra_args": [],
    },
    {
        "name": "weak_s3",
        "description": "Reduce tube samples from 5 to 3 to cut blur while keeping the same overall support.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_sigma060",
        "description": "Three samples with a narrower Gaussian kernel.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SIGMA": "0.60",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_sigma055",
        "description": "Three samples with a slightly sharper temporal kernel.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SIGMA": "0.55",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_span090",
        "description": "Three samples with slightly shorter temporal support.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SPAN": "0.90",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_cov090",
        "description": "Three samples with slightly reduced covariance inflation.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.90",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_sigma060_cov090",
        "description": "Combine fewer samples, narrower kernel, and milder covariance.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SIGMA": "0.60",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.90",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_sigma055_span090_cov090",
        "description": "A more concentrated support variant while keeping weak-tube rendering intact.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SIGMA": "0.55",
            "TEMPORAL_TUBE_SPAN": "0.90",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.90",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_sigma060_cov090_drift090",
        "description": "Slightly reduce temporal drift amplitude on the best concentrated recipe.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SIGMA": "0.60",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.90",
            "TEMPORAL_DRIFT_SCALE": "0.90",
        },
        "extra_args": [],
    },
    {
        "name": "weak_s3_sigma060_cov090_w125",
        "description": "Borrow a mild center-weight sharpening idea to emphasize central tube samples.",
        "env": {
            "TEMPORAL_TUBE_SAMPLES": "3",
            "TEMPORAL_TUBE_SIGMA": "0.60",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.90",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.25",
        },
        "extra_args": [],
    },
]

for idx, variant in enumerate(VARIANTS):
    env = dict(variant.get("env", {}))
    env.setdefault("GS_PORT", str(6121 + idx))
    variant["env"] = env


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_dir_for(namespace: str) -> Path:
    return REPO_ROOT / "runs" / namespace / DATASET / SCENE_NAME


def parse_train_metrics(train_log: Path) -> dict:
    result = {
        "coarse_test_psnr": None,
        "fine_test_psnr": None,
        "coarse_test_l1": None,
        "fine_test_l1": None,
        "status": "missing",
    }
    if not train_log.exists():
        return result
    text = train_log.read_text(errors="ignore")
    matches = re.findall(r"\[ITER 3000\] Evaluating test: L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+)", text)
    if matches:
        result["coarse_test_l1"] = float(matches[0][0])
        result["coarse_test_psnr"] = float(matches[0][1])
        if len(matches) >= 2:
            result["fine_test_l1"] = float(matches[1][0])
            result["fine_test_psnr"] = float(matches[1][1])
    if "Training complete." in text:
        result["status"] = "complete"
    return result


def parse_eval_metrics(run_dir: Path) -> dict:
    results_json = run_dir / "results.json"
    if not results_json.exists():
        return {"PSNR": None, "SSIM": None, "MS-SSIM": None, "LPIPS-vgg": None}
    data = json.loads(results_json.read_text()).get("ours_3000", {})
    return {
        "PSNR": data.get("PSNR"),
        "SSIM": data.get("SSIM"),
        "MS-SSIM": data.get("MS-SSIM"),
        "LPIPS-vgg": data.get("LPIPS-vgg"),
    }


def current_result(variant: dict) -> dict:
    namespace = f"stellar_tube_q20260328_{variant['name']}"
    run_dir = run_dir_for(namespace)
    result = {
        "name": variant["name"],
        "description": variant["description"],
        "namespace": namespace,
        "run_dir": str(run_dir),
        "env": variant["env"],
        "extra_args": variant["extra_args"],
    }
    result.update(parse_train_metrics(run_dir / "train.log"))
    result.update(parse_eval_metrics(run_dir))
    return result


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_status(results: list[dict], running: list[str]) -> None:
    STATUS_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "scene": SCENE,
                "baseline_fine_3000": REFERENCE_BASELINE_FINE_3000,
                "weak_fine_3000": REFERENCE_WEAK_FINE_3000,
                "weak_full_psnr": REFERENCE_WEAK_FULL_PSNR,
                "running": running,
                "results": results,
            },
            indent=2,
        )
    )

    ranked = sorted(
        results,
        key=lambda item: item["fine_test_psnr"] if item["fine_test_psnr"] is not None else -1.0,
        reverse=True,
    )
    lines = [
        "# WeakTube Queue 20260328",
        "",
        f"- Dataset/scene: `{DATASET} / {SCENE}`",
        f"- Baseline fine 3000 PSNR: `{REFERENCE_BASELINE_FINE_3000:.4f}`",
        f"- Weak tube fine 3000 PSNR: `{REFERENCE_WEAK_FINE_3000:.4f}`",
        f"- Weak tube full PSNR: `{REFERENCE_WEAK_FULL_PSNR:.4f}`",
        f"- Running: `{', '.join(running) if running else 'none'}`",
        "",
        "## Variant Manifest",
        "",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{variant['name']}`: {variant['description']}")
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Variant | Fine3000 | Delta vs baseline | Delta vs weak | Full PSNR | SSIM | LPIPS | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in ranked:
        fine = item["fine_test_psnr"]
        delta_baseline = None if fine is None else fine - REFERENCE_BASELINE_FINE_3000
        delta_weak = None if fine is None else fine - REFERENCE_WEAK_FINE_3000
        lines.append(
            f"| `{item['name']}` | "
            f"{format_float(fine)} | "
            f"{format_float(delta_baseline)} | "
            f"{format_float(delta_weak)} | "
            f"{format_float(item['PSNR'])} | "
            f"{format_float(item['SSIM'])} | "
            f"{format_float(item['LPIPS-vgg'])} | "
            f"{item['status']} |"
        )
    LEADERBOARD_MD.write_text("\n".join(lines) + "\n")


def print_result(prefix: str, result: dict) -> None:
    print(
        f"{prefix}: {result['name']} fine3000={format_float(result['fine_test_psnr'])}, "
        f"full_psnr={format_float(result['PSNR'])}, status={result['status']}",
        flush=True,
    )


def run_variant(variant: dict) -> dict:
    namespace = f"stellar_tube_q20260328_{variant['name']}"
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    run_dir = run_dir_for(namespace)

    train_log = run_dir / "train.log"
    results_json = run_dir / "results.json"
    if train_log.exists() and results_json.exists():
        result = current_result(variant)
        result["status"] = "complete_cached"
        return result

    train_cmd = ["bash", str(TRAIN_SCRIPT), DATASET, SCENE, *COMMON_TRAIN_ARGS, *variant["extra_args"]]
    train_proc = subprocess.run(train_cmd, cwd="/root", env=env)
    result = current_result(variant)
    if train_proc.returncode != 0:
        result["status"] = f"train_failed_{train_proc.returncode}"
        return result

    eval_cmd = ["bash", str(EVAL_SCRIPT), DATASET, SCENE]
    eval_proc = subprocess.run(eval_cmd, cwd="/root", env=env)
    result = current_result(variant)
    if eval_proc.returncode != 0:
        result["status"] = f"eval_failed_{eval_proc.returncode}"
        return result

    result["status"] = "complete"
    return result


def main() -> int:
    ensure_dirs()
    MANIFEST_JSON.write_text(json.dumps({"dataset": DATASET, "scene": SCENE, "variants": VARIANTS}, indent=2))

    results: list[dict] = []
    future_to_variant = {}
    completed = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for variant in VARIANTS:
            print(f"[queue] submit -> {variant['name']}", flush=True)
            future = executor.submit(run_variant, variant)
            future_to_variant[future] = variant
            running = [future_to_variant[f]["name"] for f in future_to_variant if not f.done()]
            write_status(results, running)

        for future in as_completed(future_to_variant):
            variant = future_to_variant[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = {
                    "name": variant["name"],
                    "description": variant["description"],
                    "namespace": f"stellar_tube_q20260328_{variant['name']}",
                    "run_dir": str(run_dir_for(f"stellar_tube_q20260328_{variant['name']}")),
                    "env": variant["env"],
                    "extra_args": variant["extra_args"],
                    "coarse_test_psnr": None,
                    "fine_test_psnr": None,
                    "coarse_test_l1": None,
                    "fine_test_l1": None,
                    "PSNR": None,
                    "SSIM": None,
                    "MS-SSIM": None,
                    "LPIPS-vgg": None,
                    "status": f"exception_{type(exc).__name__}",
                }
            results.append(result)
            completed.add(variant["name"])
            running = [future_to_variant[f]["name"] for f in future_to_variant if not f.done()]
            write_status(results, running)
            print_result("[done]", result)

    print("[queue] all variants finished", flush=True)
    ranked = sorted(results, key=lambda item: item["fine_test_psnr"] if item["fine_test_psnr"] is not None else -1.0, reverse=True)
    if ranked:
        print_result("[best]", ranked[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())

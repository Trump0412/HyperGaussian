#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_stellar_worldtube.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_stellar_worldtube.sh"
REPORT_DIR = REPO_ROOT / "reports" / "worldtube_focus_queue_20260328"
STATUS_JSON = REPORT_DIR / "queue_status.json"
LEADERBOARD_MD = REPORT_DIR / "leaderboard.md"
MANIFEST_JSON = REPORT_DIR / "manifest.json"

DATASET = "hypernerf"
SCENE = "misc/split-cookie"
SCENE_NAME = SCENE.split("/")[-1]
FINAL_ITERATION = "3000"

REFERENCE_BASELINE_FINE_3000 = 28.5610855326933
REFERENCE_WEAK_FINE_3000 = 28.933096156400794
REFERENCE_CURRENT_BEST_FINE_3000 = 27.43046547384823

BASE_ENV = {
    "GS_ITERATIONS": "3000",
    "GS_COARSE_ITERATIONS": "3000",
    "GS_TEST_ITERATIONS": "3000",
    "GS_SAVE_ITERATIONS": "3000",
    "GS_CHECKPOINT_ITERATIONS": "3000",
    "GS_RENDER_ITERATION": FINAL_ITERATION,
    "TEMPORAL_ACCELERATION_ENABLED": "0",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
    "TEMPORAL_WORLDTUBE_LITE": "0",
    "TEMPORAL_WORLDTUBE_TRANSMITTANCE_SPLIT": "0",
    "TEMPORAL_WORLDTUBE_DENSIFY_GRAD_NORMALIZE": "0",
    "TEMPORAL_WORLDTUBE_DENSIFY_GRAD_POWER": "0.5",
    "TEMPORAL_WORLDTUBE_SAMPLES": "3",
    "TEMPORAL_WORLDTUBE_SPAN": "0.50",
    "TEMPORAL_WORLDTUBE_SIGMA": "0.35",
    "TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.0",
    "TEMPORAL_WORLDTUBE_SCALE_MIX": "0.0",
    "TEMPORAL_WORLDTUBE_SUPPORT_SCALE_MIX": "0.0",
    "TEMPORAL_WORLDTUBE_CHILD_SCALE_SHRINK": "0.95",
    "TEMPORAL_WORLDTUBE_ENERGY_PRESERVING": "1",
    "TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX": "0.20",
    "TEMPORAL_WORLDTUBE_REG_WEIGHT": "0.0",
    "TEMPORAL_WORLDTUBE_RATIO_WEIGHT": "0.0",
    "TEMPORAL_WORLDTUBE_DENSIFY_WEIGHT": "0.0",
    "TEMPORAL_WORLDTUBE_SPLIT_SHRINK": "1.0",
    "TEMPORAL_WORLDTUBE_ADAPTIVE_SUPPORT": "0",
    "TEMPORAL_WORLDTUBE_OPACITY_FLOOR": "0.0",
    "TEMPORAL_WORLDTUBE_VISIBILITY_MIX": "0.0",
    "TEMPORAL_WORLDTUBE_INTEGRAL_MIX": "0.0",
    "TEMPORAL_WORLDTUBE_PRUNE_KEEP_QUANTILE": "1.0",
    "SPACETIME_AWARE_OPTIMIZATION": "0",
    "TEMPORAL_DRIFT_SCALE": "1.0",
}

VARIANTS = [
    {
        "name": "base_confirm",
        "description": "Repeat the current best sample3 gate recipe to verify the post-patch baseline.",
        "env": {},
        "extra_args": [],
    },
    {
        "name": "sigma030",
        "description": "Narrower temporal kernel on the current best recipe.",
        "env": {"TEMPORAL_WORLDTUBE_SIGMA": "0.30"},
        "extra_args": [],
    },
    {
        "name": "sigma025",
        "description": "More concentrated kernel to reduce render blur.",
        "env": {"TEMPORAL_WORLDTUBE_SIGMA": "0.25"},
        "extra_args": [],
    },
    {
        "name": "span045",
        "description": "Slightly shorter support span on the current best recipe.",
        "env": {"TEMPORAL_WORLDTUBE_SPAN": "0.45"},
        "extra_args": [],
    },
    {
        "name": "span040",
        "description": "More aggressive support shortening while keeping the rest unchanged.",
        "env": {"TEMPORAL_WORLDTUBE_SPAN": "0.40"},
        "extra_args": [],
    },
    {
        "name": "shrink098",
        "description": "Recover detail by shrinking child scales less.",
        "env": {"TEMPORAL_WORLDTUBE_CHILD_SCALE_SHRINK": "0.98"},
        "extra_args": [],
    },
    {
        "name": "energy010",
        "description": "Reduce density-mass bias in energy-preserving alpha split.",
        "env": {"TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX": "0.10"},
        "extra_args": [],
    },
    {
        "name": "energy000",
        "description": "Use pure normalized sample weights in energy-preserving alpha split.",
        "env": {"TEMPORAL_WORLDTUBE_ENERGY_GATE_MIX": "0.00"},
        "extra_args": [],
    },
    {
        "name": "renderpow125",
        "description": "Sharpen render-time sample weights with power 1.25.",
        "env": {"TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.25"},
        "extra_args": [],
    },
    {
        "name": "renderpow150",
        "description": "Sharpen render-time sample weights with power 1.50.",
        "env": {"TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.50"},
        "extra_args": [],
    },
    {
        "name": "renderpow200",
        "description": "Aggressively sharpen render-time sample weights with power 2.00.",
        "env": {"TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "2.00"},
        "extra_args": [],
    },
    {
        "name": "renderpow150_sigma030",
        "description": "Combine moderate weight sharpening with a narrower kernel.",
        "env": {
            "TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.50",
            "TEMPORAL_WORLDTUBE_SIGMA": "0.30",
        },
        "extra_args": [],
    },
    {
        "name": "renderpow150_span045",
        "description": "Combine moderate weight sharpening with a shorter support span.",
        "env": {
            "TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.50",
            "TEMPORAL_WORLDTUBE_SPAN": "0.45",
        },
        "extra_args": [],
    },
    {
        "name": "drift090_renderpow150",
        "description": "Slightly reduce temporal drift amplitude while sharpening central weights.",
        "env": {
            "TEMPORAL_DRIFT_SCALE": "0.90",
            "TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.50",
        },
        "extra_args": [],
    },
    {
        "name": "sample2_renderpow150",
        "description": "Use two samples plus moderate central-weight sharpening to reduce blur footprint.",
        "env": {
            "TEMPORAL_WORLDTUBE_SAMPLES": "2",
            "TEMPORAL_WORLDTUBE_RENDER_WEIGHT_POWER": "1.50",
        },
        "extra_args": [],
    },
]


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
        if len(matches) >= 1:
            result["coarse_test_l1"] = float(matches[0][0])
            result["coarse_test_psnr"] = float(matches[0][1])
        if len(matches) >= 2:
            result["fine_test_l1"] = float(matches[1][0])
            result["fine_test_psnr"] = float(matches[1][1])
    if "Training complete." in text:
        result["status"] = "complete"
    return result


def parse_full_metrics(full_metrics: Path) -> dict:
    if not full_metrics.exists():
        results_json = full_metrics.parent / "results.json"
        if not results_json.exists():
            return {
                "PSNR": None,
                "SSIM": None,
                "MS-SSIM": None,
                "LPIPS-vgg": None,
            }
        data = json.loads(results_json.read_text()).get("ours_3000", {})
        return {
            "PSNR": data.get("PSNR"),
            "SSIM": data.get("SSIM"),
            "MS-SSIM": data.get("MS-SSIM"),
            "LPIPS-vgg": data.get("LPIPS-vgg"),
        }
    data = json.loads(full_metrics.read_text())
    return {
        "PSNR": data.get("PSNR"),
        "SSIM": data.get("SSIM"),
        "MS-SSIM": data.get("MS-SSIM"),
        "LPIPS-vgg": data.get("LPIPS-vgg"),
    }


def current_result(variant: dict) -> dict:
    namespace = f"stellar_worldtube_focusq_20260328_{variant['name']}"
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
    result.update(parse_full_metrics(run_dir / "full_metrics_with_lpips_wait_eval.json"))
    return result


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_status(results: list[dict]) -> None:
    STATUS_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "scene": SCENE,
                "baseline_fine_3000": REFERENCE_BASELINE_FINE_3000,
                "weak_fine_3000": REFERENCE_WEAK_FINE_3000,
                "current_best_fine_3000": REFERENCE_CURRENT_BEST_FINE_3000,
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
        "# Worldtube Focus Queue 20260328",
        "",
        f"- Dataset/scene: `{DATASET} / {SCENE}`",
        f"- Baseline fine 3000 PSNR: `{REFERENCE_BASELINE_FINE_3000:.4f}`",
        f"- Weak tube fine 3000 PSNR: `{REFERENCE_WEAK_FINE_3000:.4f}`",
        f"- Previous best worldtube fine 3000 PSNR: `{REFERENCE_CURRENT_BEST_FINE_3000:.4f}`",
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
            "| Variant | Fine3000 | Delta vs baseline | Delta vs prev best | Full PSNR | SSIM | LPIPS | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in ranked:
        fine = item["fine_test_psnr"]
        delta_baseline = None if fine is None else fine - REFERENCE_BASELINE_FINE_3000
        delta_prev = None if fine is None else fine - REFERENCE_CURRENT_BEST_FINE_3000
        lines.append(
            f"| `{item['name']}` | "
            f"{format_float(fine)} | "
            f"{format_float(delta_baseline)} | "
            f"{format_float(delta_prev)} | "
            f"{format_float(item['PSNR'])} | "
            f"{format_float(item['SSIM'])} | "
            f"{format_float(item['LPIPS-vgg'])} | "
            f"{item['status']} |"
        )
    LEADERBOARD_MD.write_text("\n".join(lines) + "\n")


def print_result(prefix: str, result: dict) -> None:
    print(
        f"{prefix}: fine3000={format_float(result['fine_test_psnr'])}, "
        f"full_psnr={format_float(result['PSNR'])}, status={result['status']}",
        flush=True,
    )


def run_variant(variant: dict) -> dict:
    namespace = f"stellar_worldtube_focusq_20260328_{variant['name']}"
    env = dict(**BASE_ENV, **variant["env"])
    merged_env = dict(**os.environ, **env)
    merged_env["GS_RUN_NAMESPACE"] = namespace
    merged_env["GS_RENDER_ITERATION"] = FINAL_ITERATION

    run_dir = run_dir_for(namespace)
    train_log = run_dir / "train.log"
    full_metrics = run_dir / "full_metrics_with_lpips_wait_eval.json"
    results_json = run_dir / "results.json"

    if train_log.exists() and (full_metrics.exists() or results_json.exists()):
        result = current_result(variant)
        result["status"] = "complete_cached"
        return result

    print(f"\n=== Running {variant['name']} ===", flush=True)
    print(variant["description"], flush=True)
    train_cmd = ["bash", str(TRAIN_SCRIPT), DATASET, SCENE, *variant["extra_args"]]
    train_proc = subprocess.run(train_cmd, cwd="/root", env=merged_env)
    result = current_result(variant)
    if train_proc.returncode != 0:
        result["status"] = f"train_failed_{train_proc.returncode}"
        return result

    eval_cmd = ["bash", str(EVAL_SCRIPT), DATASET, SCENE]
    eval_proc = subprocess.run(eval_cmd, cwd="/root", env=merged_env)
    result = current_result(variant)
    if eval_proc.returncode != 0:
        result["status"] = f"eval_failed_{eval_proc.returncode}"
        return result

    result["status"] = "complete"
    return result


def main() -> int:
    ensure_dirs()
    MANIFEST_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "scene": SCENE,
                "variants": VARIANTS,
            },
            indent=2,
        )
    )

    results: list[dict] = []
    for idx, variant in enumerate(VARIANTS, start=1):
        print(f"[queue] {idx}/{len(VARIANTS)} -> {variant['name']}", flush=True)
        result = run_variant(variant)
        results.append(result)
        write_status(results)
        print_result("[done]", result)

    print("[queue] all variants finished", flush=True)
    ranked = sorted(
        results,
        key=lambda item: item["fine_test_psnr"] if item["fine_test_psnr"] is not None else -1.0,
        reverse=True,
    )
    if ranked:
        print_result("[best]", ranked[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())

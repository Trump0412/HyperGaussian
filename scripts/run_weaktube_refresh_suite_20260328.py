#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_SH = REPO_ROOT / "scripts" / "common.sh"
TRAIN_TUBE_SH = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_TUBE_SH = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"
RUN_PUBLIC_PROTOCOL_SH = REPO_ROOT / "scripts" / "run_public_query_protocol.sh"
RUN_QUERY_PIPELINE_SH = REPO_ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh"
RUN_REMOVAL_SH = REPO_ROOT / "scripts" / "run_scene_deepfill_removal_experiment.sh"
PREPARE_CUT_LEMON_SH = REPO_ROOT / "scripts" / "prepare_cut_lemon1.sh"
FULLFRAME_PY = REPO_ROOT / "scripts" / "fullframe_metrics.py"
EVAL_PUBLIC_PY = REPO_ROOT / "scripts" / "evaluate_public_query_protocol.py"

REPORT_DIR = REPO_ROOT / "reports" / "weaktube_refresh_suite_20260328"
QUEUE_LOG = REPORT_DIR / "queue.log"
STATUS_JSON = REPORT_DIR / "queue_status.json"
LEADERBOARD_MD = REPORT_DIR / "leaderboard.md"
MANIFEST_JSON = REPORT_DIR / "manifest.json"
PUBLIC_EVAL_DIR = REPORT_DIR / "public_eval"

FULLFRAME_OUT_NAME = "full_metrics_with_lpips_refresh_20260328.json"
EXPERIMENT_ROOT = Path(os.environ.get("GS_EXPERIMENT_ROOT", str(REPO_ROOT / "experiments")))
BASE_SIGNATURE = "s3_span040_sigma032_cov005_noaccel"

BASE_ENV = {
    "TEMPORAL_TUBE_SAMPLES": "3",
    "TEMPORAL_TUBE_SPAN": "0.40",
    "TEMPORAL_TUBE_SIGMA": "0.32",
    "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
    "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
    "TEMPORAL_DRIFT_SCALE": "1.0",
    "TEMPORAL_GATE_MIX": "1.0",
    "TEMPORAL_DRIFT_MIX": "1.0",
    "TEMPORAL_ACCELERATION_ENABLED": "0",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
}

TRAIN_ARGS = [
    "--coarse_iterations", "3000",
    "--iterations", "14000",
    "--test_iterations", "3000", "7000", "14000",
    "--save_iterations", "7000", "14000",
    "--checkpoint_iterations", "7000", "14000",
]

OLD_PUBLIC_EVAL = {
    "americano": REPO_ROOT / "reports" / "4dlangsplat_compare" / "americano_public_query_eval.json",
    "chickchicken": REPO_ROOT / "reports" / "4dlangsplat_compare" / "chickchicken_public_query_eval.json",
    "espresso": REPO_ROOT / "reports" / "4dlangsplat_compare" / "espresso_public_query_eval.json",
    "split-cookie": REPO_ROOT / "reports" / "4dlangsplat_compare" / "split-cookie_public_query_eval_phaseaware.json",
}

PUBLIC_SCENES = {
    "americano": {
        "scene_rel": "misc/americano",
        "group": "misc",
        "config_scene_name": "americano",
        "run_namespace": "stellar_tube_4dlangsplat_refresh_20260328_americano",
        "protocol_json": REPO_ROOT / "reports" / "4dlangsplat_compare" / "protocol_splits" / "americano.json",
        "annotation_dir": REPO_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "americano",
        "dataset_dir": REPO_ROOT / "data" / "hypernerf" / "misc" / "americano",
        "baseline_run_dir": REPO_ROOT / "runs" / "baseline_americano_compare5k" / "hypernerf" / "americano",
        "port": 6311,
    },
    "chickchicken": {
        "scene_rel": "interp/chickchicken",
        "group": "interp",
        "config_scene_name": "chickchicken",
        "run_namespace": "stellar_tube_4dlangsplat_refresh_20260328_chickchicken",
        "protocol_json": REPO_ROOT / "reports" / "4dlangsplat_compare" / "protocol_splits" / "chickchicken.json",
        "annotation_dir": REPO_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "chickchicken",
        "dataset_dir": REPO_ROOT / "data" / "hypernerf" / "interp" / "chickchicken",
        "baseline_run_dir": REPO_ROOT / "runs" / "baseline_chickchicken_compare5k" / "hypernerf" / "chickchicken",
        "port": 6312,
    },
    "espresso": {
        "scene_rel": "misc/espresso",
        "group": "misc",
        "config_scene_name": "espresso",
        "run_namespace": "stellar_tube_4dlangsplat_refresh_20260328_espresso",
        "protocol_json": REPO_ROOT / "reports" / "4dlangsplat_compare" / "protocol_splits" / "espresso.json",
        "annotation_dir": REPO_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "espresso",
        "dataset_dir": REPO_ROOT / "data" / "hypernerf" / "misc" / "espresso",
        "baseline_run_dir": REPO_ROOT / "runs" / "baseline_espresso_compare5k" / "hypernerf" / "espresso",
        "port": 6313,
    },
}

SPLIT_COOKIE = {
    "scene_key": "split-cookie",
    "scene_rel": "misc/split-cookie",
    "run_dir": REPO_ROOT / "runs" / "stellar_tube_full6_20260328_histplus_span040_sigma032" / "hypernerf" / "split-cookie",
    "protocol_json": REPO_ROOT / "reports" / "4dlangsplat_compare" / "split-cookie_query_protocol_phaseaware.json",
    "annotation_dir": REPO_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "split-cookie",
    "dataset_dir": REPO_ROOT / "data" / "hypernerf" / "misc" / "split-cookie",
    "baseline_run_dir": REPO_ROOT / "runs" / "baseline_split-cookie_compare5k_14000" / "hypernerf" / "split-cookie",
    "removal_query_slug": "split-cookie__the_complete_cookie_phaseaware",
    "removal_stamp": "20260328_refresh_splitcookie_base040_sigma032",
    "removal_tag": "splitcookie_phaseaware_refresh_base040_sigma032",
}

CUT_LEMON = {
    "scene_rel": "interp/cut-lemon1",
    "dataset_dir": REPO_ROOT / "data" / "hypernerf" / "interp" / "cut-lemon1",
    "run_namespace": "stellar_tube_cutlemon_refresh_20260328",
    "run_dir": REPO_ROOT / "runs" / "stellar_tube_cutlemon_refresh_20260328" / "hypernerf" / "cut-lemon1",
    "query_text": "the lemon",
    "query_name": "cut_the_lemon_final",
    "removal_stamp": "20260328_refresh_cutlemon_base040_sigma032",
    "removal_tag": "cutlemon_queryguided_refresh_base040_sigma032",
    "port": 6314,
}

WORKER_TASKS = {
    0: [
        {"kind": "public_scene", "scene_key": "americano"},
        {"kind": "cut_lemon"},
    ],
    1: [
        {"kind": "public_scene", "scene_key": "chickchicken"},
        {"kind": "split_public"},
        {"kind": "split_removal"},
    ],
    2: [
        {"kind": "public_scene", "scene_key": "espresso"},
    ],
}

LOCK = threading.Lock()
RUNNING: dict[int, str] = {}


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_EVAL_DIR.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOCK:
        with QUEUE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")


def run_dir_for_namespace(namespace: str, scene_name: str) -> Path:
    return REPO_ROOT / "runs" / namespace / "hypernerf" / scene_name


def scene_name_from_rel(scene_rel: str) -> str:
    return scene_rel.split("/")[-1]


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def train_log_status(train_log: Path) -> dict:
    payload = {
        "coarse_test_psnr": None,
        "fine_3000_test_psnr": None,
        "test_7000_psnr": None,
        "test_14000_psnr": None,
        "status": "missing",
    }
    if not train_log.exists():
        return payload
    text = train_log.read_text(errors="ignore")
    matches = re.findall(r"\[ITER (\d+)\] Evaluating test: L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+)", text)
    iter_3000_seen = 0
    for iter_str, _, psnr_str in matches:
        step = int(iter_str)
        psnr = float(psnr_str)
        if step == 3000:
            iter_3000_seen += 1
            if iter_3000_seen == 1:
                payload["coarse_test_psnr"] = psnr
            else:
                payload["fine_3000_test_psnr"] = psnr
        elif step == 7000:
            payload["test_7000_psnr"] = psnr
        elif step == 14000:
            payload["test_14000_psnr"] = psnr
    if "Training complete." in text:
        payload["status"] = "trained"
    elif matches:
        payload["status"] = "running"
    return payload


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metrics_summary(run_dir: Path) -> dict:
    metrics = load_json(run_dir / "metrics.json") or {}
    full = (
        load_json(run_dir / FULLFRAME_OUT_NAME)
        or load_json(run_dir / "full_metrics_with_lpips_full14k.json")
        or load_json(run_dir / "full_metrics_with_lpips_rerun_20260326.json")
        or load_json(run_dir / "full_metrics_with_lpips_wait_eval.json")
        or {}
    )
    return {
        "metrics_psnr": float_or_none(metrics.get("PSNR")),
        "metrics_ssim": float_or_none(metrics.get("SSIM")),
        "metrics_lpips": float_or_none(metrics.get("LPIPS-vgg")),
        "full_psnr": float_or_none(full.get("PSNR")),
        "full_ssim": float_or_none(full.get("SSIM")),
        "full_msssim": float_or_none(full.get("MS-SSIM")),
        "full_lpips": float_or_none(full.get("LPIPS-vgg")),
    }


def load_public_eval_summary(path: Path) -> dict:
    payload = load_json(path) or {}
    summary = payload.get("summary") or {}
    return {
        "path": str(path),
        "Acc": float_or_none(summary.get("Acc")),
        "vIoU": float_or_none(summary.get("vIoU")),
        "tIoU": float_or_none(summary.get("temporal_tIoU")),
        "query_count": summary.get("query_count"),
    }


def load_removal_summary(path: Path) -> dict:
    payload = load_json(path) or {}
    artifacts = payload.get("artifacts") or {}
    return {
        "path": str(path),
        "selected_gaussian_count": payload.get("selected_gaussian_count"),
        "remaining_gaussian_count": payload.get("remaining_gaussian_count"),
        "render_triptych": artifacts.get("render_triptych"),
        "complete_cookie_run": artifacts.get("complete_cookie_run"),
        "scene_without_cookie_run": artifacts.get("scene_without_cookie_run"),
    }


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def bash_run(command: str, env: dict[str, str], cwd: Path = REPO_ROOT) -> None:
    subprocess.run(["bash", "-lc", command], cwd=str(cwd), env=env, check=True)


def call_script(script: Path, args: list[str], env: dict[str, str], cwd: Path = REPO_ROOT) -> None:
    subprocess.run(["bash", str(script), *args], cwd=str(cwd), env=env, check=True)


def base_runtime_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def public_eval_output(scene_key: str) -> tuple[Path, Path]:
    suffix = "phaseaware_refresh_20260328" if scene_key == "split-cookie" else "refresh_20260328"
    if scene_key == "split-cookie":
        stem = f"{scene_key}_public_query_eval_phaseaware_{suffix}"
    else:
        stem = f"{scene_key}_public_query_eval_{suffix}"
    return PUBLIC_EVAL_DIR / f"{stem}.json", PUBLIC_EVAL_DIR / f"{stem}.md"


def split_cookie_removal_root() -> Path:
    return EXPERIMENT_ROOT / (
        f"{SPLIT_COOKIE['removal_stamp']}_{SPLIT_COOKIE['removal_tag']}"
    )


def cut_lemon_removal_root() -> Path:
    return EXPERIMENT_ROOT / (
        f"{CUT_LEMON['removal_stamp']}_{CUT_LEMON['removal_tag']}"
    )


def write_manifest() -> None:
    manifest = {
        "frozen_base_signature": BASE_SIGNATURE,
        "base_env": BASE_ENV,
        "train_args": TRAIN_ARGS,
        "public_scenes": PUBLIC_SCENES,
        "split_cookie": SPLIT_COOKIE,
        "cut_lemon": CUT_LEMON,
        "worker_tasks": WORKER_TASKS,
        "historical_public_eval": {key: str(path) for key, path in OLD_PUBLIC_EVAL.items()},
    }
    MANIFEST_JSON.write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")


def status_payload() -> dict:
    reconstruction_rows = []
    for scene_key, cfg in PUBLIC_SCENES.items():
        run_dir = run_dir_for_namespace(cfg["run_namespace"], cfg["config_scene_name"])
        row = {
            "scene": scene_key,
            "run_dir": str(run_dir),
            "train": train_log_status(run_dir / "train.log"),
            "metrics": load_metrics_summary(run_dir),
            "baseline_run_dir": str(cfg["baseline_run_dir"]),
            "baseline_metrics": load_metrics_summary(cfg["baseline_run_dir"]),
        }
        reconstruction_rows.append(row)

    split_row = {
        "scene": "split-cookie",
        "run_dir": str(SPLIT_COOKIE["run_dir"]),
        "train": train_log_status(SPLIT_COOKIE["run_dir"] / "train.log"),
        "metrics": load_metrics_summary(SPLIT_COOKIE["run_dir"]),
        "baseline_run_dir": str(SPLIT_COOKIE["baseline_run_dir"]),
        "baseline_metrics": load_metrics_summary(SPLIT_COOKIE["baseline_run_dir"]),
    }
    reconstruction_rows.append(split_row)

    semantics_rows = []
    for scene_key in ["americano", "chickchicken", "espresso", "split-cookie"]:
        new_json, _ = public_eval_output(scene_key)
        semantics_rows.append(
            {
                "scene": scene_key,
                "old": load_public_eval_summary(OLD_PUBLIC_EVAL[scene_key]),
                "new": load_public_eval_summary(new_json),
            }
        )

    removal_rows = [
        {
            "scene": "split-cookie",
            "summary": load_removal_summary(split_cookie_removal_root() / "bundle_summary.json"),
        },
        {
            "scene": "cut-lemon1",
            "summary": load_removal_summary(cut_lemon_removal_root() / "bundle_summary.json"),
        },
    ]

    return {
        "frozen_base_signature": BASE_SIGNATURE,
        "running": {str(gpu): label for gpu, label in RUNNING.items()},
        "reconstruction": reconstruction_rows,
        "semantics": semantics_rows,
        "removals": removal_rows,
    }


def write_status() -> None:
    payload = status_payload()
    with LOCK:
        STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        lines = [
            "# WeakTube Refresh Suite 20260328",
            "",
            f"- Frozen base: `{BASE_SIGNATURE}`",
            "- Frozen params: `samples=3, span=0.40, sigma=0.32, covariance_mix=0.05, acceleration=off`",
            f"- Running: `{', '.join(payload['running'].values()) if payload['running'] else 'none'}`",
            "",
            "## Reconstruction",
            "",
            "| Scene | Fine3000 | Test14000 | Full PSNR | Baseline PSNR | Delta | Run |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in payload["reconstruction"]:
            fine = row["train"]["fine_3000_test_psnr"]
            test14000 = row["train"]["test_14000_psnr"]
            full_psnr = row["metrics"]["full_psnr"] or row["metrics"]["metrics_psnr"]
            baseline_psnr = row["baseline_metrics"]["full_psnr"] or row["baseline_metrics"]["metrics_psnr"]
            delta = None if full_psnr is None or baseline_psnr is None else full_psnr - baseline_psnr
            lines.append(
                f"| `{row['scene']}` | {format_float(fine)} | {format_float(test14000)} | {format_float(full_psnr)} | {format_float(baseline_psnr)} | {format_float(delta)} | `{row['run_dir']}` |"
            )

        lines.extend(
            [
                "",
                "## Public Semantics",
                "",
                "| Scene | Old vIoU | New vIoU | Delta | Old tIoU | New tIoU | Output |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in payload["semantics"]:
            old_viou = row["old"]["vIoU"]
            new_viou = row["new"]["vIoU"]
            delta = None if old_viou is None or new_viou is None else new_viou - old_viou
            lines.append(
                f"| `{row['scene']}` | {format_float(old_viou)} | {format_float(new_viou)} | {format_float(delta)} | {format_float(row['old']['tIoU'])} | {format_float(row['new']['tIoU'])} | `{row['new']['path']}` |"
            )

        lines.extend(
            [
                "",
                "## Removal",
                "",
                "| Scene | Selected | Remaining | Triptych | Summary |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in payload["removals"]:
            summary = row["summary"]
            lines.append(
                f"| `{row['scene']}` | {summary.get('selected_gaussian_count', 'n/a')} | {summary.get('remaining_gaussian_count', 'n/a')} | `{summary.get('render_triptych', 'n/a')}` | `{summary.get('path', 'n/a')}` |"
            )

        LEADERBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_running(gpu: int, label: str | None) -> None:
    with LOCK:
        if label is None:
            RUNNING.pop(gpu, None)
        else:
            RUNNING[gpu] = label
    write_status()


def fullframe_metrics(run_dir: Path, gpu: int, out_name: str = FULLFRAME_OUT_NAME) -> None:
    out_path = run_dir / out_name
    if out_path.exists():
        return
    env = base_runtime_env(gpu)
    cmd = (
        f"source '{COMMON_SH}' && "
        f"gs_python '{FULLFRAME_PY}' --run-dir '{run_dir}' --with-lpips --out-name '{out_name}'"
    )
    bash_run(cmd, env=env)


def baseline_fullframe_if_needed(run_dir: Path, gpu: int) -> None:
    if (run_dir / FULLFRAME_OUT_NAME).exists():
        return
    if (run_dir / "full_metrics_with_lpips_rerun_20260326.json").exists():
        return
    if (run_dir / "full_metrics_with_lpips_wait_eval.json").exists():
        return
    fullframe_metrics(run_dir, gpu=gpu, out_name=FULLFRAME_OUT_NAME)


def has_standard_eval_outputs(run_dir: Path) -> bool:
    test_dir = run_dir / "test"
    if not test_dir.exists():
        return False
    return any(path.is_dir() and path.name.startswith("ours_") for path in test_dir.iterdir())


def run_public_eval(scene_key: str, run_dir: Path, dataset_dir: Path, protocol_json: Path, annotation_dir: Path, gpu: int) -> None:
    output_json, output_md = public_eval_output(scene_key)
    if output_json.exists():
        return
    env = base_runtime_env(gpu)
    env["QUERY_ANNOTATION_DIR"] = str(annotation_dir)
    call_script(
        RUN_PUBLIC_PROTOCOL_SH,
        [str(protocol_json), str(run_dir), str(dataset_dir)],
        env=env,
    )
    cmd = (
        f"source '{COMMON_SH}' && "
        f"gs_python '{EVAL_PUBLIC_PY}' "
        f"--protocol-json '{protocol_json}' "
        f"--annotation-dir '{annotation_dir}' "
        f"--dataset-dir '{dataset_dir}' "
        f"--query-root '{run_dir / 'entitybank' / 'query_guided'}' "
        f"--output-json '{output_json}' "
        f"--output-md '{output_md}'"
    )
    bash_run(cmd, env=env)


def ensure_cut_lemon_prepared(gpu: int) -> None:
    if CUT_LEMON["dataset_dir"].exists():
        return
    env = base_runtime_env(gpu)
    call_script(PREPARE_CUT_LEMON_SH, [], env=env)


def train_public_scene(scene_key: str, gpu: int) -> None:
    cfg = PUBLIC_SCENES[scene_key]
    run_dir = run_dir_for_namespace(cfg["run_namespace"], cfg["config_scene_name"])
    env = base_runtime_env(gpu)
    env["GS_RUN_NAMESPACE"] = cfg["run_namespace"]
    env["GS_PORT"] = str(cfg["port"])

    train_status = train_log_status(run_dir / "train.log")
    if train_status["status"] != "trained":
        append_log(f"train {scene_key} gpu={gpu}")
        call_script(TRAIN_TUBE_SH, ["hypernerf", cfg["scene_rel"], *TRAIN_ARGS], env=env)

    if not (run_dir / "metrics.json").exists() or not has_standard_eval_outputs(run_dir):
        append_log(f"eval {scene_key} gpu={gpu}")
        call_script(EVAL_TUBE_SH, ["hypernerf", cfg["scene_rel"]], env=env)

    append_log(f"fullframe {scene_key} gpu={gpu}")
    fullframe_metrics(run_dir, gpu=gpu)

    append_log(f"public_eval {scene_key} gpu={gpu}")
    run_public_eval(
        scene_key=scene_key,
        run_dir=run_dir,
        dataset_dir=cfg["dataset_dir"],
        protocol_json=cfg["protocol_json"],
        annotation_dir=cfg["annotation_dir"],
        gpu=gpu,
    )

    append_log(f"baseline_fullframe {scene_key} gpu={gpu}")
    baseline_fullframe_if_needed(cfg["baseline_run_dir"], gpu=gpu)


def run_split_cookie_public(gpu: int) -> None:
    append_log(f"split-cookie public_eval gpu={gpu}")
    run_public_eval(
        scene_key="split-cookie",
        run_dir=SPLIT_COOKIE["run_dir"],
        dataset_dir=SPLIT_COOKIE["dataset_dir"],
        protocol_json=SPLIT_COOKIE["protocol_json"],
        annotation_dir=SPLIT_COOKIE["annotation_dir"],
        gpu=gpu,
    )


def run_split_cookie_removal(gpu: int) -> None:
    summary_path = split_cookie_removal_root() / "bundle_summary.json"
    if summary_path.exists():
        return
    env = base_runtime_env(gpu)
    env["DEEPFILL_SOURCE_RUN_DIR"] = str(SPLIT_COOKIE["run_dir"])
    env["DEEPFILL_QUERY_ROOT"] = str(
        SPLIT_COOKIE["run_dir"] / "entitybank" / "query_guided" / SPLIT_COOKIE["removal_query_slug"]
    )
    env["GS_EXPERIMENT_STAMP"] = SPLIT_COOKIE["removal_stamp"]
    env["DEEPFILL_EXPERIMENT_TAG"] = SPLIT_COOKIE["removal_tag"]
    append_log(f"split-cookie removal gpu={gpu}")
    call_script(RUN_REMOVAL_SH, ["split-cookie"], env=env)


def train_cut_lemon_and_removal(gpu: int) -> None:
    ensure_cut_lemon_prepared(gpu)
    env = base_runtime_env(gpu)
    env["GS_RUN_NAMESPACE"] = CUT_LEMON["run_namespace"]
    env["GS_PORT"] = str(CUT_LEMON["port"])

    train_status = train_log_status(CUT_LEMON["run_dir"] / "train.log")
    if train_status["status"] != "trained":
        append_log(f"train cut-lemon1 gpu={gpu}")
        call_script(TRAIN_TUBE_SH, ["hypernerf", CUT_LEMON["scene_rel"], *TRAIN_ARGS], env=env)

    if not (CUT_LEMON["run_dir"] / "metrics.json").exists() or not has_standard_eval_outputs(CUT_LEMON["run_dir"]):
        append_log(f"eval cut-lemon1 gpu={gpu}")
        call_script(EVAL_TUBE_SH, ["hypernerf", CUT_LEMON["scene_rel"]], env=env)

    append_log(f"fullframe cut-lemon1 gpu={gpu}")
    fullframe_metrics(CUT_LEMON["run_dir"], gpu=gpu)

    query_root = CUT_LEMON["run_dir"] / "entitybank" / "query_guided" / CUT_LEMON["query_name"]
    if not (query_root / "final_query_render_sourcebg" / "validation.json").exists():
        append_log(f"query cut-lemon1 gpu={gpu}")
        call_script(
            RUN_QUERY_PIPELINE_SH,
            [
                str(CUT_LEMON["run_dir"]),
                str(CUT_LEMON["dataset_dir"]),
                CUT_LEMON["query_text"],
                CUT_LEMON["query_name"],
            ],
            env=env,
        )

    summary_path = cut_lemon_removal_root() / "bundle_summary.json"
    if not summary_path.exists():
        env["DEEPFILL_SOURCE_RUN_DIR"] = str(CUT_LEMON["run_dir"])
        env["DEEPFILL_QUERY_ROOT"] = str(query_root)
        env["GS_EXPERIMENT_STAMP"] = CUT_LEMON["removal_stamp"]
        env["DEEPFILL_EXPERIMENT_TAG"] = CUT_LEMON["removal_tag"]
        append_log(f"removal cut-lemon1 gpu={gpu}")
        call_script(RUN_REMOVAL_SH, ["cut-lemon1"], env=env)


def run_task(task: dict, gpu: int) -> None:
    kind = task["kind"]
    if kind == "public_scene":
        train_public_scene(task["scene_key"], gpu)
    elif kind == "split_public":
        run_split_cookie_public(gpu)
    elif kind == "split_removal":
        run_split_cookie_removal(gpu)
    elif kind == "cut_lemon":
        train_cut_lemon_and_removal(gpu)
    else:
        raise ValueError(f"Unsupported task kind: {kind}")


def worker(gpu: int, tasks: list[dict]) -> None:
    for task in tasks:
        label = f"gpu{gpu}:{task['kind']}:{task.get('scene_key', task.get('scene_rel', 'cut-lemon1'))}"
        set_running(gpu, label)
        append_log(f"start {label}")
        try:
            run_task(task, gpu)
            append_log(f"done {label}")
        except Exception as exc:  # noqa: BLE001
            append_log(f"fail {label}: {exc}")
            raise
        finally:
            set_running(gpu, None)


def main() -> int:
    ensure_dirs()
    QUEUE_LOG.write_text("", encoding="utf-8")
    write_manifest()
    write_status()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, gpu, tasks) for gpu, tasks in WORKER_TASKS.items()]
        for future in futures:
            future.result()
    write_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())

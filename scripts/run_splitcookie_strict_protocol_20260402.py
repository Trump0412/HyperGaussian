#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_BASELINE_SH = REPO_ROOT / "scripts" / "train_baseline.sh"
EVAL_BASELINE_SH = REPO_ROOT / "scripts" / "eval_baseline.sh"
TRAIN_TUBE_SH = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_TUBE_SH = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"

DATASET = "hypernerf"
SCENE = "misc/split-cookie"
SCENE_NAME = SCENE.split("/")[-1]

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

CANDIDATES = [
    {
        "name": "c040_s034_cov005_w100_lrlow",
        "description": "Current best-run default",
        "env": {},
    },
    {
        "name": "c042_s030_cov005_w105_gs110",
        "description": "Round2 top snapshot candidate",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.42",
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.05",
            "TEMPORAL_GATE_SHARPNESS": "1.10",
        },
    },
    {
        "name": "c040_s032_cov005_w100_stdlr",
        "description": "Sigma 0.32 with standard temporal LR",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_LR_INIT": "0.00016",
            "TEMPORAL_LR_FINAL": "0.000016",
        },
    },
    {
        "name": "c038_s032_cov005_w100",
        "description": "Slightly shorter support",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.38",
            "TEMPORAL_TUBE_SIGMA": "0.32",
        },
    },
    {
        "name": "c040_s030_cov005_w100_lrhi",
        "description": "Sharper sigma + higher temporal LR",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_LR_INIT": "0.00020",
            "TEMPORAL_LR_FINAL": "0.000020",
        },
    },
    {
        "name": "c040_s032_cov005_d105",
        "description": "Drift scale up",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_DRIFT_SCALE": "1.05",
        },
    },
    {
        "name": "c040_s032_cov005_d095",
        "description": "Drift scale down",
        "env": {
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_DRIFT_SCALE": "0.95",
        },
    },
    {
        "name": "c035_s028_cov008_w120",
        "description": "Round1 robust branch",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.35",
            "TEMPORAL_TUBE_SIGMA": "0.28",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.08",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.20",
        },
    },
]


REPORT_DIR = REPO_ROOT / "reports" / "splitcookie_strict_protocol_20260402"
LOG_DIR = REPORT_DIR / "logs"
STATUS_JSON = REPORT_DIR / "status.json"
SUMMARY_MD = REPORT_DIR / "summary.md"
WINNER_ENV_JSON = REPORT_DIR / "winner_env.json"

BASELINE_FULL14K_RUN = REPO_ROOT / "runs" / "baseline_4dgs_20260330" / DATASET / SCENE_NAME
BASELINE_FULL14K_RESULTS = BASELINE_FULL14K_RUN / "results.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def parse_results_block(results: dict[str, Any] | None, target_iter: int) -> dict[str, Any]:
    if not isinstance(results, dict) or not results:
        return {}
    key = f"ours_{target_iter}"
    block = results.get(key)
    if isinstance(block, dict):
        return block

    best_iter = -1
    best_block: dict[str, Any] = {}
    for k, v in results.items():
        if not isinstance(v, dict):
            continue
        m = re.match(r"ours_(\d+)$", str(k))
        if not m:
            continue
        it = int(m.group(1))
        if it > best_iter:
            best_iter = it
            best_block = v
    return best_block


def has_results_iter(run_dir: Path, target_iter: int) -> bool:
    results = read_json(run_dir / "results.json")
    block = parse_results_block(results, target_iter)
    return bool(block and block.get("PSNR") is not None)


def parse_fps_from_render_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"FPS:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not matches:
        return None
    return float_or_none(matches[-1])


def run_dir_for(namespace: str) -> Path:
    return REPO_ROOT / "runs" / namespace / DATASET / SCENE_NAME


def collect_row(
    *,
    kind: str,
    stage_iter: int,
    name: str,
    namespace: str,
    gpu: int,
    run_dir: Path,
    status: str,
    env_payload: dict[str, str] | None = None,
) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json") or {}
    results = read_json(run_dir / "results.json") or {}
    block = parse_results_block(results, stage_iter)

    psnr = float_or_none(metrics.get("PSNR"))
    ssim = float_or_none(metrics.get("SSIM"))
    lpips = float_or_none(metrics.get("LPIPS-vgg"))
    if psnr is None:
        psnr = float_or_none(block.get("PSNR"))
    if ssim is None:
        ssim = float_or_none(block.get("SSIM"))
    if lpips is None:
        lpips = float_or_none(block.get("LPIPS-vgg"))

    train_s = float_or_none(metrics.get("train_seconds"))
    if train_s is None:
        train_meta = read_json(run_dir / "train_meta.json") or {}
        train_s = float_or_none(train_meta.get("elapsed_seconds"))

    render_s = float_or_none(metrics.get("render_seconds"))
    if render_s is None:
        render_meta = read_json(run_dir / "render_meta.json") or {}
        render_s = float_or_none(render_meta.get("elapsed_seconds"))

    fps = float_or_none(metrics.get("render_fps"))
    if fps is None:
        fps = parse_fps_from_render_log(run_dir / "render.log")

    storage_mb = None
    storage_bytes = float_or_none(metrics.get("storage_bytes"))
    if storage_bytes is not None:
        storage_mb = storage_bytes / (1024.0 * 1024.0)
    else:
        ckpt = run_dir / "point_cloud" / f"iteration_{stage_iter}" / "point_cloud.ply"
        if ckpt.exists():
            storage_mb = ckpt.stat().st_size / (1024.0 * 1024.0)

    total_s = None
    if train_s is not None or render_s is not None:
        total_s = float(train_s or 0.0) + float(render_s or 0.0)

    return {
        "kind": kind,
        "stage_iter": stage_iter,
        "name": name,
        "namespace": namespace,
        "gpu": gpu,
        "run_dir": str(run_dir),
        "status": status,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "train_seconds": train_s,
        "render_seconds": render_s,
        "time_seconds": total_s,
        "fps": fps,
        "storage_mb": storage_mb,
        "env": env_payload,
        "updated_at_utc": utc_now(),
    }


def run_cmd(cmd: list[str], env: dict[str, str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{utc_now()}] $ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def run_baseline(stage: str, gpu: int, port: int, force: bool) -> dict[str, Any]:
    stage_iter = 3000 if stage == "3k" else 14000
    namespace = f"baseline_splitcookie_strict{stage}_20260402"
    run_dir = run_dir_for(namespace)
    log_file = LOG_DIR / f"baseline_{stage}_gpu{gpu}.log"
    env = dict(os.environ)
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    train_args = COMMON_3K_ARGS if stage == "3k" else COMMON_14K_ARGS
    status = "ok"
    try:
        ckpt = run_dir / "point_cloud" / f"iteration_{stage_iter}" / "point_cloud.ply"
        if force or not ckpt.exists():
            rc = run_cmd(["bash", str(TRAIN_BASELINE_SH), DATASET, SCENE, *train_args], env, log_file)
            if rc != 0:
                raise RuntimeError(f"baseline train failed rc={rc}")

        if force or not has_results_iter(run_dir, stage_iter):
            rc = run_cmd(
                ["bash", str(EVAL_BASELINE_SH), DATASET, SCENE, "--skip_train", "--skip_video"],
                env,
                log_file,
            )
            if rc != 0:
                raise RuntimeError(f"baseline eval failed rc={rc}")
    except Exception as exc:
        status = f"error: {exc}"

    return collect_row(
        kind=f"baseline_{stage}",
        stage_iter=stage_iter,
        name=f"baseline_{stage}",
        namespace=namespace,
        gpu=gpu,
        run_dir=run_dir,
        status=status,
        env_payload=None,
    )


def run_tube_variant(variant: dict[str, Any], stage: str, gpu: int, port: int, force: bool) -> dict[str, Any]:
    stage_iter = 3000 if stage == "3k" else 14000
    namespace = f"stellar_tube_splitcookie_strict{stage}_20260402_{variant['name']}"
    run_dir = run_dir_for(namespace)
    log_file = LOG_DIR / f"{variant['name']}_{stage}_gpu{gpu}.log"

    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(variant["env"])
    env["GS_RUN_NAMESPACE"] = namespace
    env["GS_PORT"] = str(port)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    train_args = COMMON_3K_ARGS if stage == "3k" else COMMON_14K_ARGS
    status = "ok"
    try:
        ckpt = run_dir / "point_cloud" / f"iteration_{stage_iter}" / "point_cloud.ply"
        if force or not ckpt.exists():
            rc = run_cmd(["bash", str(TRAIN_TUBE_SH), DATASET, SCENE, *train_args], env, log_file)
            if rc != 0:
                raise RuntimeError(f"tube train failed rc={rc}")

        if force or not has_results_iter(run_dir, stage_iter):
            rc = run_cmd(
                ["bash", str(EVAL_TUBE_SH), DATASET, SCENE, "--skip_train", "--skip_video"],
                env,
                log_file,
            )
            if rc != 0:
                raise RuntimeError(f"tube eval failed rc={rc}")
    except Exception as exc:
        status = f"error: {exc}"

    merged_env = dict(BASE_ENV)
    merged_env.update(variant["env"])
    return collect_row(
        kind=f"tube_{stage}",
        stage_iter=stage_iter,
        name=variant["name"],
        namespace=namespace,
        gpu=gpu,
        run_dir=run_dir,
        status=status,
        env_payload=merged_env,
    )


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[float, float, float]:
        psnr = r.get("psnr")
        lpips = r.get("lpips")
        ssim = r.get("ssim")
        psnr_v = float(psnr) if isinstance(psnr, (int, float)) else -1e9
        lpips_v = float(lpips) if isinstance(lpips, (int, float)) else 1e9
        ssim_v = float(ssim) if isinstance(ssim, (int, float)) else -1e9
        return (-psnr_v, lpips_v, -ssim_v)

    return sorted(rows, key=key)


def load_baseline14k_ref() -> dict[str, float | None]:
    out = {"psnr": None, "ssim": None, "lpips": None}
    results = read_json(BASELINE_FULL14K_RESULTS) or {}
    block = parse_results_block(results, 14000)
    out["psnr"] = float_or_none(block.get("PSNR"))
    out["ssim"] = float_or_none(block.get("SSIM"))
    out["lpips"] = float_or_none(block.get("LPIPS-vgg"))
    return out


def write_summary(status: dict[str, Any]) -> None:
    baseline_ref = status.get("baseline_14k_ref", {})
    baseline_3k = status.get("baseline_3k")
    rows3k = status.get("results_3k", [])
    rows14k = status.get("results_14k", [])
    promoted = status.get("promoted", [])
    winner = status.get("winner")

    lines = [
        "# Split-Cookie Strict Protocol Optimization (2026-04-02)",
        "",
        f"- Generated (UTC): `{utc_now()}`",
        f"- Dataset/Scene: `{DATASET} / {SCENE}`",
        f"- GPUs: `{status.get('gpus')}`",
        f"- Baseline 14k ref PSNR/SSIM/LPIPS: `{baseline_ref.get('psnr')}` / `{baseline_ref.get('ssim')}` / `{baseline_ref.get('lpips')}`",
        "",
        "## Baseline 3k",
        "",
    ]
    if baseline_3k:
        lines.append(
            f"- PSNR/SSIM/LPIPS: `{baseline_3k.get('psnr')}` / `{baseline_3k.get('ssim')}` / `{baseline_3k.get('lpips')}`"
        )
        lines.append(
            f"- Time/FPS/Storage: `{baseline_3k.get('time_seconds')}` / `{baseline_3k.get('fps')}` / `{baseline_3k.get('storage_mb')}`"
        )
        lines.append(f"- Status: `{baseline_3k.get('status')}`")
    else:
        lines.append("- n/a")

    lines.extend(
        [
            "",
            "## 3k Ranking",
            "",
            "| Rank | Variant | PSNR | SSIM | LPIPS | ΔPSNR vs baseline3k | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    baseline3k_psnr = None
    if isinstance(baseline_3k, dict):
        baseline3k_psnr = baseline_3k.get("psnr")
    for i, row in enumerate(rank_rows(rows3k), 1):
        delta = None
        if isinstance(row.get("psnr"), (int, float)) and isinstance(baseline3k_psnr, (int, float)):
            delta = float(row["psnr"]) - float(baseline3k_psnr)
        lines.append(
            "| {rank} | `{name}` | {psnr} | {ssim} | {lpips} | {delta} | {status} |".format(
                rank=i,
                name=row.get("name"),
                psnr=(f"{row['psnr']:.4f}" if isinstance(row.get("psnr"), (int, float)) else "n/a"),
                ssim=(f"{row['ssim']:.4f}" if isinstance(row.get("ssim"), (int, float)) else "n/a"),
                lpips=(f"{row['lpips']:.4f}" if isinstance(row.get("lpips"), (int, float)) else "n/a"),
                delta=(f"{delta:+.4f}" if isinstance(delta, float) else "n/a"),
                status=row.get("status"),
            )
        )

    lines.extend(
        [
            "",
            "## Promoted To 14k",
            "",
            f"- {', '.join('`'+x+'`' for x in promoted) if promoted else 'n/a'}",
            "",
            "## 14k Ranking",
            "",
            "| Rank | Variant | PSNR | SSIM | LPIPS | ΔPSNR vs 4DGS14k | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    base14_psnr = baseline_ref.get("psnr")
    for i, row in enumerate(rank_rows(rows14k), 1):
        delta = None
        if isinstance(row.get("psnr"), (int, float)) and isinstance(base14_psnr, (int, float)):
            delta = float(row["psnr"]) - float(base14_psnr)
        lines.append(
            "| {rank} | `{name}` | {psnr} | {ssim} | {lpips} | {delta} | {status} |".format(
                rank=i,
                name=row.get("name"),
                psnr=(f"{row['psnr']:.4f}" if isinstance(row.get("psnr"), (int, float)) else "n/a"),
                ssim=(f"{row['ssim']:.4f}" if isinstance(row.get("ssim"), (int, float)) else "n/a"),
                lpips=(f"{row['lpips']:.4f}" if isinstance(row.get("lpips"), (int, float)) else "n/a"),
                delta=(f"{delta:+.4f}" if isinstance(delta, float) else "n/a"),
                status=row.get("status"),
            )
        )

    lines.extend(["", "## Winner", ""])
    if isinstance(winner, dict):
        lines.append(f"- Variant: `{winner.get('name')}`")
        lines.append(
            f"- PSNR/SSIM/LPIPS: `{winner.get('psnr')}` / `{winner.get('ssim')}` / `{winner.get('lpips')}`"
        )
        if isinstance(base14_psnr, (int, float)) and isinstance(winner.get("psnr"), (int, float)):
            lines.append(f"- ΔPSNR vs 4DGS14k: `{winner['psnr'] - base14_psnr:+.4f}`")
    else:
        lines.append("- n/a")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict protocol split-cookie optimization")
    parser.add_argument("--gpus", default="2,3", help="comma-separated gpu ids")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("No GPU provided")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    variants = [dict(v) for v in CANDIDATES]
    for i, v in enumerate(variants):
        v["idx"] = i

    status: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "dataset": DATASET,
        "scene": SCENE,
        "gpus": gpus,
        "force_rerun": args.force_rerun,
        "baseline_14k_ref": load_baseline14k_ref(),
        "baseline_3k": None,
        "results_3k": [],
        "promoted": [],
        "results_14k": [],
        "winner": None,
    }
    write_json(STATUS_JSON, status)

    print(f"[strict] start at {utc_now()}", flush=True)
    print(f"[strict] baseline 14k ref: {status['baseline_14k_ref']}", flush=True)

    # Baseline 3k first
    baseline_gpu = gpus[0]
    baseline_row = run_baseline("3k", baseline_gpu, 9200, args.force_rerun)
    status["baseline_3k"] = baseline_row
    write_json(STATUS_JSON, status)
    print(f"[strict] baseline 3k done: {baseline_row}", flush=True)

    # Stage 3k candidates on per-GPU sequential workers
    assignments: dict[int, list[dict[str, Any]]] = {g: [] for g in gpus}
    for i, v in enumerate(variants):
        assignments[gpus[i % len(gpus)]].append(v)

    rows3k: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker3k(gpu: int, tasks: list[dict[str, Any]]) -> None:
        for task in tasks:
            port = 9100 + task["idx"]
            row = run_tube_variant(task, "3k", gpu, port, args.force_rerun)
            with lock:
                rows3k.append(row)
                status["results_3k"] = rows3k
                write_json(STATUS_JSON, status)
            print(
                f"[strict] 3k done gpu{gpu} {task['name']} psnr={row.get('psnr')} status={row.get('status')}",
                flush=True,
            )

    threads = []
    for gpu, tasks in assignments.items():
        if not tasks:
            continue
        t = threading.Thread(target=worker3k, args=(gpu, tasks), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    ranked3k = rank_rows(rows3k)
    promoted = [r["name"] for r in ranked3k if isinstance(r.get("psnr"), (int, float))][: max(1, args.top_k)]
    status["promoted"] = promoted
    write_json(STATUS_JSON, status)
    print(f"[strict] promoted to 14k: {promoted}", flush=True)

    # Stage 14k for promoted
    name_to_variant = {v["name"]: v for v in variants}
    promoted_variants = [name_to_variant[n] for n in promoted if n in name_to_variant]

    assign14: dict[int, list[dict[str, Any]]] = {g: [] for g in gpus}
    for i, v in enumerate(promoted_variants):
        assign14[gpus[i % len(gpus)]].append(v)

    rows14k: list[dict[str, Any]] = []

    def worker14k(gpu: int, tasks: list[dict[str, Any]]) -> None:
        for task in tasks:
            port = 9300 + task["idx"]
            row = run_tube_variant(task, "14k", gpu, port, args.force_rerun)
            with lock:
                rows14k.append(row)
                status["results_14k"] = rows14k
                write_json(STATUS_JSON, status)
            print(
                f"[strict] 14k done gpu{gpu} {task['name']} psnr={row.get('psnr')} status={row.get('status')}",
                flush=True,
            )

    threads = []
    for gpu, tasks in assign14.items():
        if not tasks:
            continue
        t = threading.Thread(target=worker14k, args=(gpu, tasks), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    ranked14k = rank_rows(rows14k)
    winner = ranked14k[0] if ranked14k else None
    status["winner"] = winner
    write_json(STATUS_JSON, status)

    if isinstance(winner, dict) and isinstance(winner.get("env"), dict):
        winner_env = dict(winner["env"])
        payload = {
            "generated_at_utc": utc_now(),
            "winner_name": winner.get("name"),
            "winner_namespace": winner.get("namespace"),
            "winner_psnr": winner.get("psnr"),
            "winner_ssim": winner.get("ssim"),
            "winner_lpips": winner.get("lpips"),
            "winner_env": winner_env,
        }
        write_json(WINNER_ENV_JSON, payload)

    write_summary(status)

    print("[strict] done")
    if winner:
        print(f"[strict] winner={winner.get('name')} psnr={winner.get('psnr')} ssim={winner.get('ssim')} lpips={winner.get('lpips')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

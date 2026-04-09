#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = REPO_ROOT / "scripts" / "run_baseline_benchmark12_multigpu.py"
FLEX_SCRIPT = REPO_ROOT / "scripts" / "run_weaktube_benchmark12_flex_multigpu.py"

PROGRAM_DIR = REPO_ROOT / "reports" / "weaktube_budget_gate_20260403"
PROGRAM_DIR.mkdir(parents=True, exist_ok=True)
STATUS_JSON = PROGRAM_DIR / "program_status.json"
PLAN_MD = PROGRAM_DIR / "plan.md"
SUMMARY_MD = PROGRAM_DIR / "summary.md"
RUNNER_LOG = PROGRAM_DIR / "runner.log"

SCENES = [
    "espresso",
    "americano",
    "cut_lemon",
    "split_cookie",
    "keyboard",
    "torchchocolate",
    "coffee_martini",
    "flame_steak",
    "cook_spinach",
    "cut_roasted_beef",
    "sear_steak",
    "flame_salmon",
]
SCENES_CSV = ",".join(SCENES)
GPUS_CSV = os.environ.get("BUDGET_GATE_GPUS", "0,1,2,3,4,5")

SEED_SCREEN = int(os.environ.get("BUDGET_GATE_SEED_SCREEN", "3407"))
SEEDS_CONFIRM = [3407, 7777]
SEED_FULL = int(os.environ.get("BUDGET_GATE_SEED_FULL", "2026"))

GATE_DELTA_THRESHOLD = float(os.environ.get("BUDGET_GATE_DELTA_THRESHOLD", "0.0"))
GATE_MIN_SCENES = int(os.environ.get("BUDGET_GATE_MIN_SCENES", "10"))
MAX_RETRY_VARIANTS = int(os.environ.get("BUDGET_GATE_MAX_RETRY_VARIANTS", "2"))
ALLOW_14K = int(os.environ.get("BUDGET_GATE_ALLOW_14K", "1"))
CLEANUP_SCREEN_RUNS = int(os.environ.get("BUDGET_GATE_CLEANUP_SCREEN", "1"))
DYNERF_DOWNSCALE = max(1.0, float(os.environ.get("BUDGET_GATE_DYNERF_DOWNSCALE", "2.0")))
HYPERNERF_PER_GPU = max(1, int(os.environ.get("BUDGET_GATE_HYPERNERF_PER_GPU", "2")))

BEST_ENV_BASE = {
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

# Candidate design based on previously positive patterns:
# - gs110/w105 family for torchchocolate & cut_roasted_beef uplift
# - stdlr and drift095 family for americano/espresso stability
# - cov006 for split_cookie robustness
CANDIDATES = [
    {
        "name": "cand_gs110_w105",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.42",
            "TEMPORAL_TUBE_SIGMA": "0.30",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.05",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
            "TEMPORAL_GATE_SHARPNESS": "1.10",
        },
    },
    {
        "name": "cand_stdlr_sigma032",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.40",
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_LR_INIT": "0.00016",
            "TEMPORAL_LR_FINAL": "0.000016",
        },
    },
    {
        "name": "cand_drift095_sigma032",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.40",
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_DRIFT_SCALE": "0.95",
        },
    },
    {
        "name": "cand_cov006_sigma032",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.40",
            "TEMPORAL_TUBE_SIGMA": "0.32",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.06",
        },
    },
    {
        "name": "cand_robust035028",
        "env": {
            "TEMPORAL_TUBE_SPAN": "0.35",
            "TEMPORAL_TUBE_SIGMA": "0.28",
            "TEMPORAL_TUBE_COVARIANCE_MIX": "0.08",
            "TEMPORAL_TUBE_WEIGHT_POWER": "1.20",
        },
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{now_utc()}] {msg}"
    print(line, flush=True)
    with open(RUNNER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def summarize_report(report_dir: Path) -> dict[str, Any]:
    status_path = report_dir / "queue_status.json"
    if not status_path.exists():
        return {
            "report_dir": str(report_dir),
            "rows": 0,
            "errors": 0,
            "mean_psnr": None,
            "min_psnr": None,
            "max_psnr": None,
            "per_scene": {},
            "status_exists": False,
        }

    st = load_json(status_path)
    comp = [r for r in st.get("completed", []) if isinstance(r, dict)]
    rows = [r for r in comp if isinstance(r.get("psnr"), (int, float))]
    vals = [float(r["psnr"]) for r in rows]
    per_scene = {r["scene_key"]: float(r["psnr"]) for r in rows if isinstance(r.get("scene_key"), str)}
    return {
        "report_dir": str(report_dir),
        "rows": len(rows),
        "errors": len(st.get("errors", [])) if isinstance(st.get("errors"), list) else 0,
        "mean_psnr": (sum(vals) / len(vals)) if vals else None,
        "min_psnr": min(vals) if vals else None,
        "max_psnr": max(vals) if vals else None,
        "per_scene": per_scene,
        "status_exists": True,
    }


def baseline_delta_metrics(per_scene: dict[str, float], baseline_map: dict[str, float]) -> dict[str, Any]:
    deltas = {}
    for scene, b in baseline_map.items():
        o = per_scene.get(scene)
        if isinstance(o, (int, float)):
            deltas[scene] = float(o) - float(b)
    vals = list(deltas.values())
    return {
        "scene_deltas": deltas,
        "delta_mean": (sum(vals) / len(vals)) if vals else None,
        "delta_min": min(vals) if vals else None,
        "delta_max": max(vals) if vals else None,
        "positive_scenes": sum(1 for v in vals if v > 0),
        "total_scenes_compared": len(vals),
    }


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def cleanup_report_runs(report_dir: Path, keep_namespaces: set[str] | None = None) -> dict[str, Any]:
    keep_namespaces = keep_namespaces or set()
    status_path = report_dir / "queue_status.json"
    if not status_path.exists():
        return {"removed_run_dirs": 0, "removed_gb": 0.0}

    st = load_json(status_path)
    completed = [r for r in st.get("completed", []) if isinstance(r, dict)]
    removed = 0
    removed_bytes = 0
    for row in completed:
        ns = row.get("namespace")
        run_dir = row.get("run_dir")
        if not isinstance(ns, str) or not isinstance(run_dir, str):
            continue
        if ns in keep_namespaces:
            continue
        p = Path(run_dir)
        if not p.exists():
            continue
        try:
            if REPO_ROOT / "runs" not in p.parents:
                continue
            removed_bytes += size_bytes(p)
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
        except Exception:
            continue
    return {"removed_run_dirs": removed, "removed_gb": round(removed_bytes / (1024 ** 3), 3)}


def run_baseline_3k(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    report_dir = PROGRAM_DIR / "baseline3k"
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(BASELINE_SCRIPT),
        "--run-namespace-prefix",
        "baseline_gate3k_20260403",
        "--gpus",
        GPUS_CSV,
        "--scenes",
        SCENES_CSV,
        "--report-dir",
        str(report_dir),
        "--base-port",
        "7200",
        "--train-profile",
        "3k",
        "--skip-if-complete",
        "--dynerf-downscale",
        f"{DYNERF_DOWNSCALE}",
        "--hypernerf-concurrency",
        str(HYPERNERF_PER_GPU),
    ]
    rc = run_cmd(cmd)
    summary = summarize_report(report_dir)
    summary.update({"phase": "baseline3k", "rc": rc})
    state["baseline3k"] = summary
    write_json(STATUS_JSON, state)

    baseline_map = summary.get("per_scene", {})
    if len(baseline_map) < GATE_MIN_SCENES:
        log(
            f"warning: baseline3k scenes={len(baseline_map)} < min required {GATE_MIN_SCENES}; gate reliability reduced"
        )
    return summary, baseline_map


def merged_env(custom: dict[str, str]) -> dict[str, str]:
    out = dict(BEST_ENV_BASE)
    out.update({str(k): str(v) for k, v in custom.items()})
    return out


def run_ours_job(
    phase: str,
    variant_name: str,
    variant_env: dict[str, str],
    profile: str,
    seed: int,
    base_port: int,
    baseline_map: dict[str, float],
    cleanup: bool = False,
) -> dict[str, Any]:
    report_dir = PROGRAM_DIR / phase / f"{variant_name}_seed{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)
    namespace = f"stellar_tube_{phase}_{variant_name}_seed{seed}".replace("-", "_")

    cmd = [
        "python",
        str(FLEX_SCRIPT),
        "--run-namespace-prefix",
        namespace,
        "--gpus",
        GPUS_CSV,
        "--scenes",
        SCENES_CSV,
        "--report-dir",
        str(report_dir),
        "--base-port",
        str(base_port),
        "--train-profile",
        profile,
        "--skip-if-complete",
        "--dynerf-downscale",
        f"{DYNERF_DOWNSCALE}",
        "--hypernerf-concurrency",
        str(HYPERNERF_PER_GPU),
        "--train-extra=--seed",
        f"--train-extra={seed}",
    ]
    for k in sorted(variant_env.keys()):
        cmd.extend(["--env", f"{k}={variant_env[k]}"])

    rc = run_cmd(cmd)
    summary = summarize_report(report_dir)
    summary.update(baseline_delta_metrics(summary.get("per_scene", {}), baseline_map))
    summary.update(
        {
            "phase": phase,
            "variant_name": variant_name,
            "seed": seed,
            "profile": profile,
            "namespace": namespace,
            "rc": rc,
        }
    )

    if cleanup:
        summary["cleanup"] = cleanup_report_runs(report_dir)

    return summary


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [r for r in rows if isinstance(r.get("delta_mean"), (int, float))]
    valid.sort(
        key=lambda x: (
            float(x.get("delta_mean") or -999),
            int(x.get("positive_scenes") or 0),
            float(x.get("mean_psnr") or -999),
        ),
        reverse=True,
    )
    return valid


def aggregate_confirm(confirm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in confirm_rows:
        name = r.get("variant_name")
        if not isinstance(name, str):
            continue
        by.setdefault(name, []).append(r)

    out = []
    for name, rows in by.items():
        dmeans = [float(r.get("delta_mean")) for r in rows if isinstance(r.get("delta_mean"), (int, float))]
        mpsnr = [float(r.get("mean_psnr")) for r in rows if isinstance(r.get("mean_psnr"), (int, float))]
        comp = [int(r.get("total_scenes_compared") or 0) for r in rows]
        if not dmeans:
            continue
        out.append(
            {
                "variant": name,
                "runs": len(rows),
                "mean_delta_mean": sum(dmeans) / len(dmeans),
                "min_seed_delta_mean": min(dmeans),
                "max_seed_delta_mean": max(dmeans),
                "mean_psnr": (sum(mpsnr) / len(mpsnr)) if mpsnr else None,
                "min_compared_scenes": min(comp) if comp else 0,
            }
        )

    out.sort(
        key=lambda x: (
            float(x.get("mean_delta_mean") or -999),
            float(x.get("mean_psnr") or -999),
        ),
        reverse=True,
    )
    return out


def write_plan() -> None:
    lines = [
        "# Budget Gate Plan (2026-04-03)",
        "",
        f"- Generated (UTC): `{now_utc()}`",
        f"- GPUs: `{GPUS_CSV}`",
        f"- Hard gate: `3k mean delta > {GATE_DELTA_THRESHOLD:.4f}` before any 14k",
        f"- Min compared scenes for gate: `{GATE_MIN_SCENES}`",
        f"- DyNeRF downscale: `{DYNERF_DOWNSCALE}`",
        f"- HyperNeRF concurrency per GPU: `{HYPERNERF_PER_GPU}`",
        "",
        "## Pipeline",
        "",
        "1. Run baseline 3k on benchmark-12 (once).",
        "2. Run ours 2k screening on curated 5 variants (seed=3407).",
        "3. Select top-2 by 2k delta and run 3k confirm (seeds 3407/7777).",
        "4. Gate check: only if best variant has 3k mean delta > threshold, run 14k full benchmark-12.",
        "5. If gate fails: stop without launching any 14k job.",
        "",
        "## Candidate Design",
        "",
    ]
    for c in CANDIDATES:
        lines.append(f"- `{c['name']}`")

    lines.extend(
        [
            "",
            "## Cost-Oriented Estimate",
            "",
            "- baseline3k: ~0.8-1.2h",
            "- 2k screening (5 jobs): ~3.0-4.5h",
            "- 3k confirm (4 jobs): ~3.0-4.5h",
            "- total before gate decision: ~6.8-10.2h",
            "- 14k full (only if pass): +2.5-4.0h",
        ]
    )
    PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not BASELINE_SCRIPT.exists() or not FLEX_SCRIPT.exists():
        raise RuntimeError("required scripts missing")

    write_plan()

    state: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "gpus": GPUS_CSV,
        "allow_14k": ALLOW_14K,
        "gate_delta_threshold": GATE_DELTA_THRESHOLD,
        "gate_min_scenes": GATE_MIN_SCENES,
        "dynerf_downscale": DYNERF_DOWNSCALE,
        "hypernerf_per_gpu": HYPERNERF_PER_GPU,
        "baseline3k": None,
        "screen2k": [],
        "confirm3k": [],
        "retry3k": [],
        "full14k": [],
        "gate": {"pass": False, "reason": "not_evaluated"},
    }
    write_json(STATUS_JSON, state)

    # Stage 0: baseline 3k
    baseline_summary, baseline_map = run_baseline_3k(state)
    if not baseline_map:
        state["gate"] = {"pass": False, "reason": "baseline3k_missing"}
        write_json(STATUS_JSON, state)
        raise RuntimeError("baseline3k map missing; stop")

    # Stage 1: 2k screening
    base_port = 8200
    cand_map = {c["name"]: merged_env(c["env"]) for c in CANDIDATES}
    for i, c in enumerate(CANDIDATES):
        row = run_ours_job(
            phase="screen2k",
            variant_name=c["name"],
            variant_env=cand_map[c["name"]],
            profile="2k",
            seed=SEED_SCREEN,
            base_port=base_port + i * 40,
            baseline_map=baseline_map,
            cleanup=bool(CLEANUP_SCREEN_RUNS),
        )
        state["screen2k"].append(row)
        write_json(STATUS_JSON, state)

    screen_rank = rank_rows(state["screen2k"])
    top2 = [r["variant_name"] for r in screen_rank[:2]]
    if not top2:
        state["gate"] = {"pass": False, "reason": "no_valid_screen2k"}
        write_json(STATUS_JSON, state)
        raise RuntimeError("no valid 2k screen result")

    # Stage 2: 3k confirm on top2 with 2 seeds
    base_port = 9000
    for i, name in enumerate(top2):
        for j, seed in enumerate(SEEDS_CONFIRM):
            row = run_ours_job(
                phase="confirm3k",
                variant_name=name,
                variant_env=cand_map[name],
                profile="3k",
                seed=seed,
                base_port=base_port + i * 80 + j * 40,
                baseline_map=baseline_map,
                cleanup=False,
            )
            state["confirm3k"].append(row)
            write_json(STATUS_JSON, state)

    agg = aggregate_confirm(state["confirm3k"])

    # Optional small retry if gate fails and budget allows.
    def evaluate_gate(agg_rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
        if not agg_rows:
            return False, {"reason": "no_confirm3k"}
        best = agg_rows[0]
        pass_flag = (
            isinstance(best.get("mean_delta_mean"), (int, float))
            and float(best["mean_delta_mean"]) > GATE_DELTA_THRESHOLD
            and int(best.get("min_compared_scenes") or 0) >= GATE_MIN_SCENES
        )
        info = {
            "best_variant": best.get("variant"),
            "best_mean_delta_mean": best.get("mean_delta_mean"),
            "best_min_seed_delta_mean": best.get("min_seed_delta_mean"),
            "best_mean_psnr": best.get("mean_psnr"),
            "best_min_compared_scenes": best.get("min_compared_scenes"),
        }
        return bool(pass_flag), info

    gate_pass, gate_info = evaluate_gate(agg)

    if not gate_pass and MAX_RETRY_VARIANTS > 0:
        tried = set(top2)
        retry_candidates = [r["variant_name"] for r in screen_rank if r["variant_name"] not in tried][:MAX_RETRY_VARIANTS]
        base_port = 9800
        for i, name in enumerate(retry_candidates):
            row = run_ours_job(
                phase="retry3k",
                variant_name=name,
                variant_env=cand_map[name],
                profile="3k",
                seed=SEED_SCREEN,
                base_port=base_port + i * 40,
                baseline_map=baseline_map,
                cleanup=False,
            )
            state["retry3k"].append(row)
            write_json(STATUS_JSON, state)

        combined = list(state["confirm3k"]) + list(state["retry3k"])
        agg = aggregate_confirm(combined)
        gate_pass, gate_info = evaluate_gate(agg)

    state["gate"] = {
        "pass": bool(gate_pass),
        "reason": "pass" if gate_pass else "delta_not_enough",
        **gate_info,
        "allow_14k": bool(ALLOW_14K),
    }
    write_json(STATUS_JSON, state)

    # Stage 3: 14k only if gate pass
    if gate_pass and ALLOW_14K == 1:
        winner = str(gate_info["best_variant"])
        row = run_ours_job(
            phase="full14k",
            variant_name=winner,
            variant_env=cand_map[winner],
            profile="14k",
            seed=SEED_FULL,
            base_port=10400,
            baseline_map=baseline_map,
            cleanup=False,
        )
        state["full14k"].append(row)
        write_json(STATUS_JSON, state)
        log(f"14k launched for winner={winner}")
    else:
        log("14k blocked by gate; no full training launched")

    # summary markdown
    lines = [
        "# Budget Gate Summary",
        "",
        f"- Finished (UTC): `{now_utc()}`",
        f"- Gate pass: `{state['gate']['pass']}`",
        f"- Gate reason: `{state['gate']['reason']}`",
        f"- Best variant @ gate: `{state['gate'].get('best_variant')}`",
        f"- Best 3k mean delta: `{state['gate'].get('best_mean_delta_mean')}`",
        "",
        "## Baseline3k",
        "",
        f"- rows: `{state['baseline3k']['rows']}`",
        f"- errors: `{state['baseline3k']['errors']}`",
        f"- mean_psnr: `{state['baseline3k']['mean_psnr']}`",
        "",
        "## Screen2k Ranking",
        "",
        "| Rank | Variant | DeltaMean | PositiveScenes | ComparedScenes | MeanPSNR | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(screen_rank, 1):
        lines.append(
            f"| {i} | `{r['variant_name']}` | {float(r.get('delta_mean') or 0.0):.4f} | {int(r.get('positive_scenes') or 0)} | {int(r.get('total_scenes_compared') or 0)} | {float(r.get('mean_psnr') or 0.0):.4f} | {int(r.get('errors') or 0)} |"
        )

    lines.extend(
        [
            "",
            "## Confirm3k Aggregate",
            "",
            "| Rank | Variant | Runs | MeanDeltaMean | MinSeedDelta | MeanPSNR | MinComparedScenes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, r in enumerate(agg, 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['runs']} | {float(r['mean_delta_mean']):.4f} | {float(r['min_seed_delta_mean']):.4f} | {float(r.get('mean_psnr') or 0.0):.4f} | {int(r.get('min_compared_scenes') or 0)} |"
        )

    if state["full14k"]:
        r = state["full14k"][0]
        lines.extend(
            [
                "",
                "## Full14k Result",
                "",
                f"- variant: `{r['variant_name']}`",
                f"- delta_mean_vs_baseline3k: `{float(r.get('delta_mean') or 0.0):.4f}`",
                f"- positive_scenes: `{int(r.get('positive_scenes') or 0)}/{int(r.get('total_scenes_compared') or 0)}`",
                f"- mean_psnr: `{float(r.get('mean_psnr') or 0.0):.4f}`",
            ]
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

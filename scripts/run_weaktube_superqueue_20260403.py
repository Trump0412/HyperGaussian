#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEX_SCRIPT = REPO_ROOT / "scripts" / "run_weaktube_benchmark12_flex_multigpu.py"
PHASE1_SCRIPT = REPO_ROOT / "scripts" / "run_weaktube_weakscene_boost_20260402.py"
PHASE1_PROC_PATTERN = "run_weaktube_weakscene_boost_20260402.py"
PHASE1_STATUS = REPO_ROOT / "reports" / "weakscene_boost_20260402" / "queue_status.json"
PHASE1_MANIFEST = REPO_ROOT / "reports" / "weakscene_boost_20260402" / "manifest.json"
LEGACY_BENCH_MANIFEST = REPO_ROOT / "reports" / "weaktube_benchmark12_20260402" / "remote" / "manifest.json"
BASELINE_4DGS_JSON = REPO_ROOT / "reports" / "weaktube_benchmark12_20260402" / "remote" / "4dgs_benchmark12_table_20260402.json"

PROGRAM_DIR = REPO_ROOT / "reports" / "weaktube_superqueue_20260403"
PROGRAM_DIR.mkdir(parents=True, exist_ok=True)
STATUS_JSON = PROGRAM_DIR / "program_status.json"
PLAN_MD = PROGRAM_DIR / "plan.md"
SUMMARY_MD = PROGRAM_DIR / "summary.md"

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

GPUS_CSV = os.environ.get("SUPERQUEUE_GPUS", "0,1,2,3,4,5")
SEEDS_STAGE_A = [3407, 7777]
SEED_STAGE_B = 2026
SEEDS_REPRO = [3407, 7777, 2026]

EST_MIN_3K_JOB = float(os.environ.get("SUPERQUEUE_EST_3K_MIN", "20"))
EST_MIN_14K_JOB = float(os.environ.get("SUPERQUEUE_EST_14K_MIN", "155"))
EST_MIN_OVERHEAD = float(os.environ.get("SUPERQUEUE_EST_OVERHEAD_MIN", "120"))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log(msg: str) -> None:
    print(f"[{now_utc()}] {msg}", flush=True)


def run_cmd(cmd: list[str]) -> int:
    log(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def proc_alive(pattern: str) -> bool:
    return subprocess.call(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def fget(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, str(default)))
    except Exception:
        return float(default)


def sanitize_name(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        elif ch in {" ", ".", "/", ":"}:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def make_variant_name(env: dict[str, str], prefix: str = "v") -> str:
    span = fget(env, "TEMPORAL_TUBE_SPAN", 0.40)
    sigma = fget(env, "TEMPORAL_TUBE_SIGMA", 0.32)
    cov = fget(env, "TEMPORAL_TUBE_COVARIANCE_MIX", 0.05)
    drift = fget(env, "TEMPORAL_DRIFT_SCALE", 1.0)
    w = fget(env, "TEMPORAL_TUBE_WEIGHT_POWER", 1.0)
    return f"{prefix}_sp{span:.2f}_sg{sigma:.2f}_cv{cov:.2f}_dr{drift:.2f}_w{w:.2f}".replace(".", "")


def aggregate_variant(rows: list[dict], delta_key: str) -> list[dict[str, Any]]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        name = r.get("variant")
        delta = r.get(delta_key)
        if isinstance(name, str) and isinstance(delta, (int, float)):
            by[name].append(float(delta))

    out: list[dict[str, Any]] = []
    for name, vals in by.items():
        pos = sum(1 for v in vals if v > 0)
        out.append(
            {
                "variant": name,
                "count": len(vals),
                "positives": pos,
                "avg_delta": sum(vals) / len(vals),
                "std_delta": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "max_delta": max(vals),
                "min_delta": min(vals),
            }
        )
    out.sort(key=lambda x: (x["positives"], x["avg_delta"], x["max_delta"]), reverse=True)
    return out


def phase1_expected_counts(status: dict, manifest: dict) -> tuple[int, int]:
    target_scenes = status.get("target_scenes") or manifest.get("target_scenes") or {}
    variants = manifest.get("variants") or []
    expected_3k = len(target_scenes) * len(variants)
    promoted = status.get("promoted") or {}
    expected_14k = sum(len(v) for v in promoted.values() if isinstance(v, list))
    return expected_3k, expected_14k


def is_phase1_complete(status: dict, manifest: dict) -> bool:
    expected_3k, expected_14k = phase1_expected_counts(status, manifest)
    got_3k = len(status.get("results_3k", []))
    got_14k = len(status.get("results_14k", []))
    running = len(status.get("running", []))

    complete_3k = got_3k >= expected_3k
    complete_14k = True if expected_14k == 0 else got_14k >= expected_14k
    return complete_3k and complete_14k and running == 0


def wait_or_resume_phase1() -> None:
    if not PHASE1_STATUS.exists() or not PHASE1_MANIFEST.exists():
        raise RuntimeError("phase1 status/manifest missing")

    while True:
        status = load_json(PHASE1_STATUS)
        manifest = load_json(PHASE1_MANIFEST)
        expected_3k, expected_14k = phase1_expected_counts(status, manifest)
        got_3k = len(status.get("results_3k", []))
        got_14k = len(status.get("results_14k", []))
        running = len(status.get("running", []))
        alive = proc_alive(PHASE1_PROC_PATTERN)

        log(
            f"phase1 progress: alive={alive} 3k={got_3k}/{expected_3k} 14k={got_14k}/{expected_14k} running={running}"
        )

        if is_phase1_complete(status, manifest):
            log("phase1 confirmed complete")
            return

        if alive:
            time.sleep(60)
            continue

        log("phase1 process not alive but not complete, restarting phase1 script to auto-resume cached tasks")
        rc = run_cmd(["python", str(PHASE1_SCRIPT)])
        if rc != 0:
            log(f"phase1 restart exited rc={rc}, retry in 60s")
            time.sleep(60)


def select_winner(status: dict) -> tuple[str, str, list[dict[str, Any]]]:
    r14 = [r for r in status.get("results_14k", []) if isinstance(r, dict)]
    if r14:
        ranked = aggregate_variant(r14, "delta_vs_baseline_full")
        if ranked:
            return ranked[0]["variant"], "results_14k", ranked

    r3 = [r for r in status.get("results_3k", []) if isinstance(r, dict)]
    ranked = aggregate_variant(r3, "delta_vs_baseline3k")
    if ranked:
        return ranked[0]["variant"], "results_3k", ranked

    raise RuntimeError("No winner available in phase1 status")


def build_variant_map(manifest: dict) -> dict[str, dict[str, str]]:
    base = {str(k): str(v) for k, v in (manifest.get("base_env") or {}).items()}
    out: dict[str, dict[str, str]] = {}
    for v in manifest.get("variants", []):
        name = str(v.get("name", ""))
        if not name:
            continue
        env = {str(k): str(vv) for k, vv in (v.get("env") or {}).items()}
        merged = dict(base)
        merged.update(env)
        out[name] = merged
    return out


def build_search_candidates(
    winner_name: str,
    winner_env: dict[str, str],
    ranked: list[dict[str, Any]],
    variant_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    base = dict(winner_env)

    def mk(tag: str, updates: dict[str, str]) -> dict[str, Any]:
        e = dict(base)
        e.update(updates)
        return {"name": sanitize_name(f"{tag}_{make_variant_name(e, prefix='hp')[:36]}"), "env": e, "source": tag}

    span = fget(base, "TEMPORAL_TUBE_SPAN", 0.40)
    sigma = fget(base, "TEMPORAL_TUBE_SIGMA", 0.32)
    cov = fget(base, "TEMPORAL_TUBE_COVARIANCE_MIX", 0.05)
    drift = fget(base, "TEMPORAL_DRIFT_SCALE", 1.0)
    w = fget(base, "TEMPORAL_TUBE_WEIGHT_POWER", 1.0)
    lr0 = fget(base, "TEMPORAL_LR_INIT", 0.00016)
    lr1 = fget(base, "TEMPORAL_LR_FINAL", 0.000016)

    cands = [
        {"name": sanitize_name(f"winner_{winner_name}"), "env": dict(base), "source": "winner"},
        mk("span_p2", {"TEMPORAL_TUBE_SPAN": f"{span + 0.02:.2f}"}),
        mk("span_m2", {"TEMPORAL_TUBE_SPAN": f"{max(0.20, span - 0.02):.2f}"}),
        mk("sigma_p2", {"TEMPORAL_TUBE_SIGMA": f"{sigma + 0.02:.2f}"}),
        mk("sigma_m2", {"TEMPORAL_TUBE_SIGMA": f"{max(0.20, sigma - 0.02):.2f}"}),
        mk("cov_m1", {"TEMPORAL_TUBE_COVARIANCE_MIX": f"{max(0.01, cov - 0.01):.2f}"}),
        mk("cov_p1", {"TEMPORAL_TUBE_COVARIANCE_MIX": f"{min(0.20, cov + 0.01):.2f}"}),
        mk("drift_m5", {"TEMPORAL_DRIFT_SCALE": f"{max(0.80, drift - 0.05):.2f}"}),
        mk("drift_p5", {"TEMPORAL_DRIFT_SCALE": f"{min(1.30, drift + 0.05):.2f}"}),
        mk("weight_p5", {"TEMPORAL_TUBE_WEIGHT_POWER": f"{w + 0.05:.2f}"}),
        mk(
            "lr_low",
            {
                "TEMPORAL_LR_INIT": f"{lr0 * 0.80:.8f}",
                "TEMPORAL_LR_FINAL": f"{lr1 * 0.80:.8f}",
            },
        ),
        mk(
            "lr_high",
            {
                "TEMPORAL_LR_INIT": f"{lr0 * 1.20:.8f}",
                "TEMPORAL_LR_FINAL": f"{lr1 * 1.20:.8f}",
            },
        ),
        mk(
            "accel_on",
            {
                "TEMPORAL_ACCELERATION_ENABLED": "1",
                "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0005",
            },
        ),
    ]

    # Bring in top-2 extra anchors from phase1 ranking.
    extra = []
    for row in ranked[1:5]:
        vname = row.get("variant")
        if isinstance(vname, str) and vname in variant_map:
            extra.append(
                {
                    "name": sanitize_name(f"anchor_{vname}"),
                    "env": dict(variant_map[vname]),
                    "source": f"anchor:{vname}",
                }
            )
        if len(extra) >= 2:
            break
    cands.extend(extra)

    uniq: list[dict[str, Any]] = []
    seen = set()
    for c in cands:
        key = json.dumps(c["env"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def build_ablations(base_env: dict[str, str]) -> list[dict[str, Any]]:
    def mk(tag: str, updates: dict[str, str]) -> dict[str, Any]:
        e = dict(base_env)
        e.update(updates)
        return {"name": sanitize_name(f"abl_{tag}_{make_variant_name(e, prefix='ab')[:34]}"), "env": e, "source": tag}

    span = fget(base_env, "TEMPORAL_TUBE_SPAN", 0.40)
    sigma = fget(base_env, "TEMPORAL_TUBE_SIGMA", 0.32)

    cands = [
        mk("gate090", {"TEMPORAL_GATE_MIX": "0.90"}),
        mk("driftmix090", {"TEMPORAL_DRIFT_MIX": "0.90"}),
        mk(
            "accel_on",
            {
                "TEMPORAL_ACCELERATION_ENABLED": "1",
                "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0005",
            },
        ),
        mk("cov_low", {"TEMPORAL_TUBE_COVARIANCE_MIX": "0.03"}),
        mk("cov_high", {"TEMPORAL_TUBE_COVARIANCE_MIX": "0.08"}),
        mk(
            "sample5",
            {
                "TEMPORAL_TUBE_SAMPLES": "5",
                "TEMPORAL_TUBE_SPAN": f"{min(0.70, span + 0.08):.2f}",
                "TEMPORAL_TUBE_SIGMA": f"{min(0.60, sigma + 0.08):.2f}",
            },
        ),
    ]

    uniq: list[dict[str, Any]] = []
    seen = set()
    for c in cands:
        key = json.dumps(c["env"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def read_baseline_map() -> dict[str, float]:
    if not BASELINE_4DGS_JSON.exists():
        return {}
    rows = load_json(BASELINE_4DGS_JSON)
    out: dict[str, float] = {}
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            k = r.get("scene_key")
            v = r.get("psnr")
            if isinstance(k, str) and isinstance(v, (int, float)):
                out[k] = float(v)
    return out


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
        "completed": comp,
    }


def baseline_delta_metrics(per_scene: dict[str, float], baseline_psnr: dict[str, float]) -> dict[str, Any]:
    deltas = {}
    for s, b in baseline_psnr.items():
        o = per_scene.get(s)
        if isinstance(o, (int, float)):
            deltas[s] = float(o) - float(b)
    vals = list(deltas.values())
    return {
        "scene_deltas": deltas,
        "delta_mean": (sum(vals) / len(vals)) if vals else None,
        "delta_min": min(vals) if vals else None,
        "delta_max": max(vals) if vals else None,
        "positive_scenes": sum(1 for v in vals if v > 0),
        "total_scenes_compared": len(vals),
        "all_positive": bool(vals) and all(v > 0 for v in vals),
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


def cleanup_report_runs(summary: dict[str, Any], keep_namespaces: set[str] | None = None) -> dict[str, Any]:
    keep_namespaces = keep_namespaces or set()
    removed = 0
    removed_bytes = 0
    kept = 0
    completed = summary.get("completed") or []

    for row in completed:
        if not isinstance(row, dict):
            continue
        ns = row.get("namespace")
        run_dir = row.get("run_dir")
        if not isinstance(ns, str) or not isinstance(run_dir, str):
            continue
        if ns in keep_namespaces:
            kept += 1
            continue
        p = Path(run_dir)
        if not p.exists():
            continue
        try:
            if REPO_ROOT / "runs" not in p.parents:
                # safety guard
                continue
            s = size_bytes(p)
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
            removed_bytes += s
        except Exception:
            continue

    return {
        "removed_run_dirs": removed,
        "removed_gb": round(removed_bytes / (1024 ** 3), 3),
        "kept_run_dirs": kept,
    }


def run_flex_job(
    phase: str,
    variant_name: str,
    variant_env: dict[str, str],
    profile: str,
    seed: int,
    base_port: int,
    baseline_psnr: dict[str, float],
    cleanup: bool,
    keep_namespaces: set[str] | None = None,
    namespace_override: str | None = None,
) -> dict[str, Any]:
    safe_name = sanitize_name(variant_name)
    namespace = namespace_override or sanitize_name(f"stellar_tube_{phase}_{safe_name}_seed{seed}")
    report_dir = PROGRAM_DIR / phase / f"{safe_name}_seed{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)

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
        "--train-extra=--seed",
        f"--train-extra={seed}",
    ]

    for k in sorted(variant_env.keys()):
        cmd.extend(["--env", f"{k}={variant_env[k]}"])

    rc = run_cmd(cmd)
    summary = summarize_report(report_dir)
    deltas = baseline_delta_metrics(summary.get("per_scene", {}), baseline_psnr)

    cleanup_stats: dict[str, Any] = {}
    if cleanup:
        cleanup_stats = cleanup_report_runs(summary, keep_namespaces=keep_namespaces)

    out = {
        "phase": phase,
        "variant_name": variant_name,
        "namespace": namespace,
        "profile": profile,
        "seed": seed,
        "base_port": base_port,
        "rc": rc,
        **summary,
        **deltas,
        "cleanup": cleanup,
        "cleanup_stats": cleanup_stats,
    }
    return out


def score_rows_by_variant(jobs: list[dict[str, Any]], metric_key: str = "mean_psnr") -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for j in jobs:
        name = j.get("variant_name")
        val = j.get(metric_key)
        if isinstance(name, str) and isinstance(val, (int, float)):
            by[name].append(j)

    scored = []
    for name, rows in by.items():
        vals = [float(r[metric_key]) for r in rows]
        mins = [float(r.get("min_psnr")) for r in rows if isinstance(r.get("min_psnr"), (int, float))]
        mu = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        worst = min(mins) if mins else -1e9
        score = mu - 0.25 * sd
        scored.append(
            {
                "variant": name,
                "count": len(vals),
                "mean": mu,
                "std": sd,
                "worst": worst,
                "score": score,
            }
        )

    scored.sort(key=lambda x: (x["score"], x["worst"]), reverse=True)
    return scored


def score_full_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for j in jobs:
        if not isinstance(j.get("delta_mean"), (int, float)):
            continue
        rows.append(
            {
                "variant": j["variant_name"],
                "seed": j["seed"],
                "delta_mean": float(j.get("delta_mean") or 0.0),
                "delta_min": float(j.get("delta_min") or -999.0),
                "positive_scenes": int(j.get("positive_scenes") or 0),
                "total_scenes_compared": int(j.get("total_scenes_compared") or 0),
                "mean_psnr": float(j.get("mean_psnr") or 0.0),
                "all_positive": bool(j.get("all_positive")),
            }
        )
    rows.sort(
        key=lambda x: (
            x["all_positive"],
            x["positive_scenes"],
            x["delta_mean"],
            x["delta_min"],
            x["mean_psnr"],
        ),
        reverse=True,
    )
    return rows


def estimate_minutes(search_count: int, ablation_count: int, full_count: int, repro_count: int, release_count: int) -> dict[str, float]:
    jobs_3k = search_count * len(SEEDS_STAGE_A) + ablation_count
    jobs_14k = full_count + repro_count + release_count
    m3 = jobs_3k * EST_MIN_3K_JOB
    m14 = jobs_14k * EST_MIN_14K_JOB
    total = m3 + m14 + EST_MIN_OVERHEAD
    return {
        "jobs_3k": jobs_3k,
        "jobs_14k": jobs_14k,
        "minutes_3k": m3,
        "minutes_14k": m14,
        "overhead_minutes": EST_MIN_OVERHEAD,
        "total_minutes": total,
        "total_hours": total / 60.0,
    }


def write_plan(state: dict[str, Any]) -> None:
    est = state["estimate"]
    lines = [
        "# WeakTube SuperQueue Plan (2026-04-03)",
        "",
        f"- Generated (UTC): `{now_utc()}`",
        f"- GPUs: `{GPUS_CSV}`",
        f"- Winner from phase1: `{state['winner_variant']}` ({state['winner_source']})",
        "",
        "## Pipeline",
        "",
        "1. Resume/finish weakscene phase1 automatically if interrupted.",
        "2. Backfill previously interrupted benchmark runs.",
        "3. StageA: robust hyperparameter search on benchmark-12 (3k, 2 seeds).",
        "4. StageB: ablation search on benchmark-12 (3k).",
        "5. StageC: candidate full reconstructions on benchmark-12 (14k).",
        "6. StageD: final winner reproducibility (14k, 3 seeds).",
        "7. StageE: release full reconstruction (14k keep artifacts).",
        "",
        "## Estimated Duration",
        "",
        f"- 3k jobs: `{int(est['jobs_3k'])}` x `{EST_MIN_3K_JOB:.0f} min` = `{est['minutes_3k']:.0f} min`",
        f"- 14k jobs: `{int(est['jobs_14k'])}` x `{EST_MIN_14K_JOB:.0f} min` = `{est['minutes_14k']:.0f} min`",
        f"- Overhead: `{est['overhead_minutes']:.0f} min`",
        f"- Total: `{est['total_minutes']:.0f} min` (~`{est['total_hours']:.1f} h`)",
        "",
    ]
    PLAN_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not FLEX_SCRIPT.exists():
        raise RuntimeError(f"missing flex script: {FLEX_SCRIPT}")

    wait_or_resume_phase1()
    phase1_status = load_json(PHASE1_STATUS)
    phase1_manifest = load_json(PHASE1_MANIFEST)

    winner_variant, winner_source, phase1_ranked = select_winner(phase1_status)
    variant_map = build_variant_map(phase1_manifest)
    if winner_variant not in variant_map:
        raise RuntimeError(f"winner env not found: {winner_variant}")
    winner_env = dict(variant_map[winner_variant])

    search_candidates = build_search_candidates(winner_variant, winner_env, phase1_ranked, variant_map)
    # Use first 12 search candidates to cap runtime while still >=15h total.
    search_candidates = search_candidates[:12]

    baseline_psnr = read_baseline_map()

    # Build plan/initial state
    state: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "winner_variant": winner_variant,
        "winner_source": winner_source,
        "winner_env": winner_env,
        "phase1_ranked": phase1_ranked,
        "search_candidates": search_candidates,
        "ablation_candidates": [],
        "estimate": estimate_minutes(
            search_count=len(search_candidates),
            ablation_count=6,
            full_count=4,
            repro_count=len(SEEDS_REPRO),
            release_count=1,
        ),
        "baseline_psnr": baseline_psnr,
        "backfill": [],
        "stageA": [],
        "stageB": [],
        "stageC": [],
        "stageC_extra": [],
        "stageD": [],
        "stageE_release": [],
    }
    write_plan(state)
    write_json(STATUS_JSON, state)

    # Stage0: backfill legacy interrupted benchmark to avoid stale missing rows.
    if LEGACY_BENCH_MANIFEST.exists():
        legacy_manifest = load_json(LEGACY_BENCH_MANIFEST)
        legacy_env = {str(k): str(v) for k, v in (legacy_manifest.get("best_env") or {}).items()}
        if legacy_env:
            backfill = run_flex_job(
                phase="stage0_backfill_legacy14k",
                variant_name="legacy_best04034_lrlow",
                variant_env=legacy_env,
                profile="14k",
                seed=3407,
                base_port=7600,
                baseline_psnr=baseline_psnr,
                cleanup=False,
                namespace_override="stellar_tube_bench12_best04034_lrlow_20260402",
            )
            state["backfill"].append(backfill)
            write_json(STATUS_JSON, state)

    # StageA: robust search on 12 scenes at 3k with 2 seeds.
    base_port = 8200
    for vi, cand in enumerate(search_candidates):
        for si, seed in enumerate(SEEDS_STAGE_A):
            job = run_flex_job(
                phase="stageA_search3k",
                variant_name=cand["name"],
                variant_env=cand["env"],
                profile="3k",
                seed=seed,
                base_port=base_port + vi * 40 + si * 20,
                baseline_psnr=baseline_psnr,
                cleanup=True,
            )
            state["stageA"].append(job)
            write_json(STATUS_JSON, state)

    stageA_scores = score_rows_by_variant(state["stageA"], metric_key="mean_psnr")
    top_stageA = [r["variant"] for r in stageA_scores[:3]]
    search_map = {c["name"]: c["env"] for c in search_candidates}

    # StageB: ablations around current stageA best.
    if not top_stageA:
        raise RuntimeError("stageA produced no valid candidates")
    base_ablation_env = dict(search_map[top_stageA[0]])
    ablations = build_ablations(base_ablation_env)
    state["ablation_candidates"] = ablations
    write_json(STATUS_JSON, state)

    for i, cand in enumerate(ablations):
        job = run_flex_job(
            phase="stageB_ablation3k",
            variant_name=cand["name"],
            variant_env=cand["env"],
            profile="3k",
            seed=SEED_STAGE_B,
            base_port=9000 + i * 40,
            baseline_psnr=baseline_psnr,
            cleanup=True,
        )
        state["stageB"].append(job)
        write_json(STATUS_JSON, state)

    stageB_scores = score_rows_by_variant(state["stageB"], metric_key="mean_psnr")
    top_stageB = [r["variant"] for r in stageB_scores[:2]]
    ablation_map = {c["name"]: c["env"] for c in ablations}

    # StageC: full 14k on combined top candidates (up to 4).
    candidates_full: list[str] = []
    for name in top_stageA + top_stageB:
        if name not in candidates_full:
            candidates_full.append(name)
    candidates_full = candidates_full[:4]

    env_union = dict(search_map)
    env_union.update(ablation_map)

    for i, name in enumerate(candidates_full):
        env = env_union[name]
        job = run_flex_job(
            phase="stageC_full14k",
            variant_name=name,
            variant_env=env,
            profile="14k",
            seed=SEED_STAGE_B,
            base_port=9800 + i * 40,
            baseline_psnr=baseline_psnr,
            cleanup=True,
        )
        state["stageC"].append(job)
        write_json(STATUS_JSON, state)

    full_rank = score_full_jobs(state["stageC"])
    if not full_rank:
        raise RuntimeError("stageC produced no valid full 14k result")

    best_name = full_rank[0]["variant"]
    best_env = env_union[best_name]

    # If still not fully above baseline, run two extra targeted full jobs.
    if not full_rank[0]["all_positive"]:
        span = fget(best_env, "TEMPORAL_TUBE_SPAN", 0.40)
        sigma = fget(best_env, "TEMPORAL_TUBE_SIGMA", 0.32)
        cov = fget(best_env, "TEMPORAL_TUBE_COVARIANCE_MIX", 0.05)
        extra = [
            {
                "name": sanitize_name(f"extra_sigma_m2_{best_name}"),
                "env": {**best_env, "TEMPORAL_TUBE_SIGMA": f"{max(0.20, sigma - 0.02):.2f}"},
            },
            {
                "name": sanitize_name(f"extra_cov_m1_span_p2_{best_name}"),
                "env": {
                    **best_env,
                    "TEMPORAL_TUBE_COVARIANCE_MIX": f"{max(0.01, cov - 0.01):.2f}",
                    "TEMPORAL_TUBE_SPAN": f"{min(0.70, span + 0.02):.2f}",
                },
            },
        ]
        for i, cand in enumerate(extra):
            job = run_flex_job(
                phase="stageC_extra14k",
                variant_name=cand["name"],
                variant_env=cand["env"],
                profile="14k",
                seed=SEED_STAGE_B,
                base_port=10400 + i * 40,
                baseline_psnr=baseline_psnr,
                cleanup=True,
            )
            state["stageC_extra"].append(job)
            write_json(STATUS_JSON, state)
        merged_rank = score_full_jobs(state["stageC"] + state["stageC_extra"])
        if merged_rank:
            best_name = merged_rank[0]["variant"]
            if best_name in env_union:
                best_env = env_union[best_name]
            else:
                for c in extra:
                    if c["name"] == best_name:
                        best_env = c["env"]
                        break
            full_rank = merged_rank

    # StageD: reproducibility full 14k with 3 seeds (cleanup each run).
    for i, seed in enumerate(SEEDS_REPRO):
        job = run_flex_job(
            phase="stageD_repro14k",
            variant_name=best_name,
            variant_env=best_env,
            profile="14k",
            seed=seed,
            base_port=10800 + i * 40,
            baseline_psnr=baseline_psnr,
            cleanup=True,
        )
        state["stageD"].append(job)
        write_json(STATUS_JSON, state)

    # StageE: one release full run kept on disk for downstream/open-source base.
    release_job = run_flex_job(
        phase="stageE_release14k",
        variant_name=f"release_{best_name}",
        variant_env=best_env,
        profile="14k",
        seed=9090,
        base_port=11200,
        baseline_psnr=baseline_psnr,
        cleanup=False,
        namespace_override="stellar_tube_release14k_20260403",
    )
    state["stageE_release"].append(release_job)

    # Final summary markdown.
    repro_rank = score_full_jobs(state["stageD"])
    final_release = score_full_jobs(state["stageE_release"])

    lines = [
        "# WeakTube SuperQueue Summary",
        "",
        f"- Finished (UTC): `{now_utc()}`",
        f"- Initial winner from phase1: `{winner_variant}` ({winner_source})",
        f"- Selected best variant: `{best_name}`",
        "",
        "## StageA (3k Hyperparam) Top",
        "",
        "| Rank | Variant | Runs | Mean PSNR | Std | Worst | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, r in enumerate(stageA_scores[:10], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['count']} | {r['mean']:.4f} | {r['std']:.4f} | {r['worst']:.4f} | {r['score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## StageB (3k Ablation) Top",
            "",
            "| Rank | Variant | Runs | Mean PSNR | Std | Worst | Score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, r in enumerate(stageB_scores[:10], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['count']} | {r['mean']:.4f} | {r['std']:.4f} | {r['worst']:.4f} | {r['score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## StageC+Extra (14k Full) Ranking",
            "",
            "| Rank | Variant | Mean Delta vs 4DGS | Min Delta | Positive Scenes | Compared Scenes | Mean PSNR | All Positive |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for i, r in enumerate(full_rank, 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['delta_mean']:.4f} | {r['delta_min']:.4f} | {r['positive_scenes']} | {r['total_scenes_compared']} | {r['mean_psnr']:.4f} | {r['all_positive']} |"
        )

    lines.extend(
        [
            "",
            "## StageD Repro (14k)",
            "",
            "| Rank | Seed | Mean Delta vs 4DGS | Min Delta | Positive Scenes | Compared Scenes | Mean PSNR | All Positive |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for i, r in enumerate(repro_rank, 1):
        lines.append(
            f"| {i} | {r['seed']} | {r['delta_mean']:.4f} | {r['delta_min']:.4f} | {r['positive_scenes']} | {r['total_scenes_compared']} | {r['mean_psnr']:.4f} | {r['all_positive']} |"
        )

    if final_release:
        r = final_release[0]
        lines.extend(
            [
                "",
                "## Release Run (Kept on Disk)",
                "",
                f"- Namespace: `stellar_tube_release14k_20260403`",
                f"- Mean Delta vs 4DGS: `{r['delta_mean']:.4f}`",
                f"- Min Delta: `{r['delta_min']:.4f}`",
                f"- Positive scenes: `{r['positive_scenes']}/{r['total_scenes_compared']}`",
                f"- All positive: `{r['all_positive']}`",
            ]
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    state["finished_at_utc"] = now_utc()
    state["final_best_variant"] = best_name
    state["full_rank"] = full_rank
    state["repro_rank"] = repro_rank
    write_json(STATUS_JSON, state)

    log(f"superqueue completed: best_variant={best_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

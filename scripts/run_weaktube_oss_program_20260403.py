#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FLEX_SCRIPT = REPO_ROOT / "scripts" / "run_weaktube_benchmark12_flex_multigpu.py"
PHASE1_SCRIPT_PATTERN = "run_weaktube_weakscene_boost_20260402.py"
PHASE1_STATUS = REPO_ROOT / "reports" / "weakscene_boost_20260402" / "queue_status.json"
PHASE1_MANIFEST = REPO_ROOT / "reports" / "weakscene_boost_20260402" / "manifest.json"

PROGRAM_DIR = REPO_ROOT / "reports" / "weaktube_oss_program_20260403"
PROGRAM_DIR.mkdir(parents=True, exist_ok=True)
PROGRAM_STATUS = PROGRAM_DIR / "program_status.json"
PROGRAM_PLAN = PROGRAM_DIR / "plan.md"
PROGRAM_SUMMARY = PROGRAM_DIR / "summary.md"

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
GPUS_CSV = os.environ.get("OSS_PROGRAM_GPUS", "0,1,2,3,4,5")

SEEDS_PHASE_A = [3407, 7777]
SEEDS_PHASE_C = [3407, 7777, 2026]

EST_MIN_3K_JOB = float(os.environ.get("OSS_EST_MIN_3K_JOB", "40"))
EST_MIN_14K_JOB = float(os.environ.get("OSS_EST_MIN_14K_JOB", "170"))
EST_MIN_OVERHEAD = float(os.environ.get("OSS_EST_MIN_OVERHEAD", "90"))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str]) -> int:
    print(f"[{now_utc()}] $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def proc_alive(pattern: str) -> bool:
    return subprocess.call(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def wait_phase1_done() -> None:
    print(f"[{now_utc()}] waiting current weakscene queue to finish ...", flush=True)
    while proc_alive(PHASE1_SCRIPT_PATTERN):
        if PHASE1_STATUS.exists():
            try:
                st = load_json(PHASE1_STATUS)
                print(
                    f"[{now_utc()}] phase1 progress: 3k={len(st.get('results_3k', []))} 14k={len(st.get('results_14k', []))} running={len(st.get('running', []))}",
                    flush=True,
                )
            except Exception:
                print(f"[{now_utc()}] phase1 status temporarily unreadable", flush=True)
        time.sleep(60)
    print(f"[{now_utc()}] phase1 finished", flush=True)


def aggregate_variant(rows: list[dict], delta_key: str) -> list[dict]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        name = r.get("variant")
        delta = r.get(delta_key)
        if isinstance(name, str) and isinstance(delta, (int, float)):
            by[name].append(float(delta))

    out = []
    for name, vals in by.items():
        pos = sum(1 for v in vals if v > 0)
        out.append(
            {
                "variant": name,
                "count": len(vals),
                "positives": pos,
                "avg_delta": sum(vals) / len(vals),
                "max_delta": max(vals),
                "min_delta": min(vals),
            }
        )
    out.sort(key=lambda x: (x["positives"], x["avg_delta"], x["max_delta"]), reverse=True)
    return out


def select_current_winner(status: dict) -> tuple[str, str, list[dict]]:
    r14 = [r for r in status.get("results_14k", []) if isinstance(r, dict)]
    if r14:
        ranked = aggregate_variant(r14, "delta_vs_baseline_full")
        if ranked:
            return ranked[0]["variant"], "results_14k", ranked

    r3 = [r for r in status.get("results_3k", []) if isinstance(r, dict)]
    ranked = aggregate_variant(r3, "delta_vs_baseline3k")
    if ranked:
        return ranked[0]["variant"], "results_3k", ranked

    raise RuntimeError("No winner available from phase1 status")


def build_winner_env(manifest: dict, winner_variant: str) -> dict[str, str]:
    base = {str(k): str(v) for k, v in (manifest.get("base_env") or {}).items()}
    for v in manifest.get("variants", []):
        if v.get("name") == winner_variant:
            env = {str(k): str(vv) for k, vv in (v.get("env") or {}).items()}
            out = dict(base)
            out.update(env)
            return out
    raise RuntimeError(f"winner variant not found in phase1 manifest: {winner_variant}")


def fget(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, str(default)))
    except Exception:
        return float(default)


def make_variant_name(env: dict[str, str]) -> str:
    span = fget(env, "TEMPORAL_TUBE_SPAN", 0.40)
    sigma = fget(env, "TEMPORAL_TUBE_SIGMA", 0.34)
    cov = fget(env, "TEMPORAL_TUBE_COVARIANCE_MIX", 0.05)
    drift = fget(env, "TEMPORAL_DRIFT_SCALE", 1.0)
    w = fget(env, "TEMPORAL_TUBE_WEIGHT_POWER", 1.0)
    return f"sp{span:.2f}_sg{sigma:.2f}_cv{cov:.2f}_dr{drift:.2f}_w{w:.2f}".replace(".", "")


def build_neighborhood(winner_env: dict[str, str]) -> list[dict[str, Any]]:
    base = dict(winner_env)

    def mk(tag: str, updates: dict[str, str]) -> dict[str, Any]:
        e = dict(base)
        e.update(updates)
        return {"name": f"{tag}_{make_variant_name(e)}", "env": e}

    span = fget(base, "TEMPORAL_TUBE_SPAN", 0.40)
    sigma = fget(base, "TEMPORAL_TUBE_SIGMA", 0.34)
    cov = fget(base, "TEMPORAL_TUBE_COVARIANCE_MIX", 0.05)
    drift = fget(base, "TEMPORAL_DRIFT_SCALE", 1.0)
    w = fget(base, "TEMPORAL_TUBE_WEIGHT_POWER", 1.0)

    cands = [
        mk("winner", {}),
        mk("span_p2", {"TEMPORAL_TUBE_SPAN": f"{span + 0.02:.2f}"}),
        mk("span_m2", {"TEMPORAL_TUBE_SPAN": f"{max(0.20, span - 0.02):.2f}"}),
        mk("sigma_p2", {"TEMPORAL_TUBE_SIGMA": f"{sigma + 0.02:.2f}"}),
        mk("sigma_m2", {"TEMPORAL_TUBE_SIGMA": f"{max(0.20, sigma - 0.02):.2f}"}),
        mk("cov_m1", {"TEMPORAL_TUBE_COVARIANCE_MIX": f"{max(0.01, cov - 0.01):.2f}"}),
        mk("drift_m5", {"TEMPORAL_DRIFT_SCALE": f"{max(0.80, drift - 0.05):.2f}"}),
        mk("weight_p5", {"TEMPORAL_TUBE_WEIGHT_POWER": f"{w + 0.05:.2f}"}),
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


def summarize_run(report_dir: Path) -> dict[str, Any]:
    status_path = report_dir / "queue_status.json"
    if not status_path.exists():
        return {"report_dir": str(report_dir), "mean_psnr": None, "min_psnr": None, "rows": 0, "per_scene": {}}

    st = load_json(status_path)
    rows = [r for r in st.get("completed", []) if isinstance(r, dict) and isinstance(r.get("psnr"), (int, float))]
    if not rows:
        return {"report_dir": str(report_dir), "mean_psnr": None, "min_psnr": None, "rows": 0, "per_scene": {}}

    vals = [float(r["psnr"]) for r in rows]
    per_scene = {r["scene_key"]: float(r["psnr"]) for r in rows if isinstance(r.get("scene_key"), str)}
    return {
        "report_dir": str(report_dir),
        "rows": len(rows),
        "mean_psnr": sum(vals) / len(vals),
        "min_psnr": min(vals),
        "max_psnr": max(vals),
        "per_scene": per_scene,
    }


def run_flex_job(
    phase: str,
    variant_name: str,
    variant_env: dict[str, str],
    profile: str,
    seed: int,
    base_port: int,
) -> dict[str, Any]:
    ns = f"stellar_tube_{phase}_{variant_name}_seed{seed}".replace("-", "_")
    report_dir = PROGRAM_DIR / phase / f"{variant_name}_seed{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(FLEX_SCRIPT),
        "--run-namespace-prefix",
        ns,
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
    summary = summarize_run(report_dir)
    summary.update(
        {
            "phase": phase,
            "variant_name": variant_name,
            "profile": profile,
            "seed": seed,
            "namespace": ns,
            "rc": rc,
            "base_port": base_port,
        }
    )
    return summary


def estimate_minutes(variant_count: int) -> dict[str, float]:
    jobs_a = variant_count * len(SEEDS_PHASE_A)
    jobs_b = 2
    jobs_c = len(SEEDS_PHASE_C)
    m_a = jobs_a * EST_MIN_3K_JOB
    m_b = jobs_b * EST_MIN_14K_JOB
    m_c = jobs_c * EST_MIN_14K_JOB
    total = m_a + m_b + m_c + EST_MIN_OVERHEAD
    return {
        "phaseA_jobs": jobs_a,
        "phaseB_jobs": jobs_b,
        "phaseC_jobs": jobs_c,
        "phaseA_minutes": m_a,
        "phaseB_minutes": m_b,
        "phaseC_minutes": m_c,
        "overhead_minutes": EST_MIN_OVERHEAD,
        "total_minutes": total,
        "total_hours": total / 60.0,
    }


def write_plan_md(
    winner_variant: str,
    winner_source: str,
    variants: list[dict[str, Any]],
    est: dict[str, float],
) -> None:
    lines = [
        "# WeakTube OSS Program Plan (2026-04-03)",
        "",
        f"- Generated (UTC): `{now_utc()}`",
        f"- Winner source: `{winner_source}`",
        f"- Current winner variant: `{winner_variant}`",
        f"- GPUs: `{GPUS_CSV}`",
        "",
        "## Phase Design",
        "",
        "1. PhaseA: 3k robustness neighborhood search on Benchmark-12 (6 GPUs, multi-run sequential).",
        "2. PhaseB: pick top-2 variants from PhaseA and run 14k full benchmark for generalization.",
        "3. PhaseC: final winner 14k reproducibility runs with 3 seeds.",
        "",
        "## Candidate Variants",
        "",
        "| # | Variant Name |",
        "| --- | --- |",
    ]
    for i, v in enumerate(variants, 1):
        lines.append(f"| {i} | `{v['name']}` |")

    lines.extend(
        [
            "",
            "## Estimated Duration",
            "",
            f"- PhaseA jobs: `{int(est['phaseA_jobs'])}` x `{EST_MIN_3K_JOB:.0f} min` ~= `{est['phaseA_minutes']:.0f} min`",
            f"- PhaseB jobs: `{int(est['phaseB_jobs'])}` x `{EST_MIN_14K_JOB:.0f} min` ~= `{est['phaseB_minutes']:.0f} min`",
            f"- PhaseC jobs: `{int(est['phaseC_jobs'])}` x `{EST_MIN_14K_JOB:.0f} min` ~= `{est['phaseC_minutes']:.0f} min`",
            f"- Overhead: `{est['overhead_minutes']:.0f} min`",
            f"- Total: `{est['total_minutes']:.0f} min` (~`{est['total_hours']:.1f} h`)",
            "",
        ]
    )

    PROGRAM_PLAN.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not PHASE1_STATUS.exists() or not PHASE1_MANIFEST.exists():
        raise SystemExit("phase1 files missing")

    wait_phase1_done()
    phase1_status = load_json(PHASE1_STATUS)
    phase1_manifest = load_json(PHASE1_MANIFEST)

    winner_variant, winner_source, ranked = select_current_winner(phase1_status)
    winner_env = build_winner_env(phase1_manifest, winner_variant)
    variants = build_neighborhood(winner_env)

    est = estimate_minutes(len(variants))
    write_plan_md(winner_variant, winner_source, variants, est)

    program_state: dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "winner_variant": winner_variant,
        "winner_source": winner_source,
        "winner_ranked": ranked,
        "winner_env": winner_env,
        "variants": variants,
        "estimate": est,
        "phaseA": [],
        "phaseB": [],
        "phaseC": [],
    }
    write_json(PROGRAM_STATUS, program_state)

    # PhaseA: neighborhood robustness at 3k, 2 seeds
    base_port = 9200
    for vi, v in enumerate(variants):
        for si, seed in enumerate(SEEDS_PHASE_A):
            job = run_flex_job(
                phase="ossA3k_20260403",
                variant_name=v["name"],
                variant_env=v["env"],
                profile="3k",
                seed=seed,
                base_port=base_port + (vi * 40) + (si * 20),
            )
            program_state["phaseA"].append(job)
            write_json(PROGRAM_STATUS, program_state)

    # select top2 from phaseA by (mean_of_means - 0.25*std, then min)
    agg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for j in program_state["phaseA"]:
        if isinstance(j.get("mean_psnr"), (int, float)):
            agg[j["variant_name"]].append(j)

    score_rows = []
    for name, jobs in agg.items():
        means = [float(j["mean_psnr"]) for j in jobs]
        mins = [float(j["min_psnr"]) for j in jobs if isinstance(j.get("min_psnr"), (int, float))]
        mu = sum(means) / len(means)
        sd = statistics.pstdev(means) if len(means) > 1 else 0.0
        worst = min(mins) if mins else -1e9
        score = mu - 0.25 * sd
        score_rows.append({"variant": name, "mean": mu, "std": sd, "worst": worst, "score": score})

    score_rows.sort(key=lambda x: (x["score"], x["worst"]), reverse=True)
    top2 = [x["variant"] for x in score_rows[:2]]
    variant_map = {v["name"]: v["env"] for v in variants}

    # PhaseB: top2 at 14k seed2026
    for i, name in enumerate(top2):
        job = run_flex_job(
            phase="ossB14k_20260403",
            variant_name=name,
            variant_env=variant_map[name],
            profile="14k",
            seed=2026,
            base_port=9800 + i * 40,
        )
        program_state["phaseB"].append(job)
        write_json(PROGRAM_STATUS, program_state)

    phaseb_ok = [j for j in program_state["phaseB"] if isinstance(j.get("mean_psnr"), (int, float))]
    if not phaseb_ok:
        raise RuntimeError("PhaseB produced no valid run")
    phaseb_ok.sort(key=lambda x: x["mean_psnr"], reverse=True)
    final_variant = phaseb_ok[0]["variant_name"]
    final_env = variant_map[final_variant]

    # PhaseC: final variant reproducibility 14k with 3 seeds
    for i, seed in enumerate(SEEDS_PHASE_C):
        job = run_flex_job(
            phase="ossC14k_repro_20260403",
            variant_name=final_variant,
            variant_env=final_env,
            profile="14k",
            seed=seed,
            base_port=10400 + i * 40,
        )
        program_state["phaseC"].append(job)
        write_json(PROGRAM_STATUS, program_state)

    # summary
    lines = [
        "# WeakTube OSS Program Summary",
        "",
        f"- Finished (UTC): `{now_utc()}`",
        f"- Initial winner: `{winner_variant}` ({winner_source})",
        f"- Final selected variant: `{final_variant}`",
        "",
        "## PhaseA Candidate Scores",
        "",
        "| Rank | Variant | Mean PSNR | Std | Worst | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(score_rows, 1):
        lines.append(
            f"| {i} | `{row['variant']}` | {row['mean']:.4f} | {row['std']:.4f} | {row['worst']:.4f} | {row['score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## PhaseB 14k",
            "",
            "| Variant | Seed | Mean PSNR | Min PSNR | Rows |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for j in program_state["phaseB"]:
        lines.append(
            f"| `{j.get('variant_name')}` | {j.get('seed')} | {float(j.get('mean_psnr') or 0.0):.4f} | {float(j.get('min_psnr') or 0.0):.4f} | {j.get('rows', 0)} |"
        )

    lines.extend(
        [
            "",
            "## PhaseC 14k Repro",
            "",
            "| Seed | Mean PSNR | Min PSNR | Rows |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for j in program_state["phaseC"]:
        lines.append(
            f"| {j.get('seed')} | {float(j.get('mean_psnr') or 0.0):.4f} | {float(j.get('min_psnr') or 0.0):.4f} | {j.get('rows', 0)} |"
        )

    PROGRAM_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    program_state["final_variant"] = final_variant
    program_state["finished_at_utc"] = now_utc()
    write_json(PROGRAM_STATUS, program_state)
    print(f"[{now_utc()}] OSS program completed. final_variant={final_variant}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

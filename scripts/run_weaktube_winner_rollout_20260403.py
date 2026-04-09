#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_SCRIPT_PATTERN = "run_weaktube_weakscene_boost_20260402.py"
PHASE1_REPORT_DIR = REPO_ROOT / "reports" / "weakscene_boost_20260402"
PHASE1_STATUS_JSON = PHASE1_REPORT_DIR / "queue_status.json"
PHASE1_MANIFEST_JSON = PHASE1_REPORT_DIR / "manifest.json"

ROLLOUT_REPORT_DIR = REPO_ROOT / "reports" / "weaktube_bench12_winner_rollout_20260403"
ROLLOUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
SELECTION_JSON = ROLLOUT_REPORT_DIR / "winner_selection.json"
SELECTION_MD = ROLLOUT_REPORT_DIR / "winner_selection.md"

BENCH_SCRIPT = REPO_ROOT / "scripts" / "run_weaktube_benchmark12_multigpu.py"

DEFAULT_SCENES = [
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


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_cmd(cmd: list[str]) -> int:
    print(f"[{ts()}] $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def process_alive() -> bool:
    rc = subprocess.call(["pgrep", "-f", PHASE1_SCRIPT_PATTERN], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0


def wait_phase1_done() -> None:
    print(f"[{ts()}] waiting for phase1 process: {PHASE1_SCRIPT_PATTERN}", flush=True)
    while process_alive():
        if PHASE1_STATUS_JSON.exists():
            try:
                status = load_json(PHASE1_STATUS_JSON)
                r3 = len(status.get("results_3k", []))
                r14 = len(status.get("results_14k", []))
                running = len(status.get("running", []))
                print(f"[{ts()}] phase1 running: results3k={r3} results14k={r14} running={running}", flush=True)
            except Exception:
                print(f"[{ts()}] phase1 status unreadable, retrying", flush=True)
        time.sleep(60)
    print(f"[{ts()}] phase1 process finished", flush=True)


def aggregate_variant_scores(rows: list[dict], delta_key: str) -> list[dict]:
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = row.get("variant")
        delta = row.get(delta_key)
        if not isinstance(name, str):
            continue
        if not isinstance(delta, (int, float)):
            continue
        by_variant[name].append(float(delta))

    ranked: list[dict] = []
    for name, deltas in by_variant.items():
        positives = sum(1 for v in deltas if v > 0)
        avg_delta = sum(deltas) / len(deltas)
        ranked.append(
            {
                "variant": name,
                "count": len(deltas),
                "positives": positives,
                "avg_delta": avg_delta,
                "max_delta": max(deltas),
                "min_delta": min(deltas),
            }
        )

    ranked.sort(key=lambda x: (x["positives"], x["avg_delta"], x["max_delta"]), reverse=True)
    return ranked


def pick_winner(status: dict) -> tuple[str, str, list[dict]]:
    r14 = [r for r in status.get("results_14k", []) if isinstance(r, dict)]
    if r14:
        ranked = aggregate_variant_scores(r14, "delta_vs_baseline_full")
        if ranked:
            return ranked[0]["variant"], "results_14k", ranked

    r3 = [r for r in status.get("results_3k", []) if isinstance(r, dict)]
    ranked = aggregate_variant_scores(r3, "delta_vs_baseline3k")
    if ranked:
        return ranked[0]["variant"], "results_3k", ranked

    raise RuntimeError("No usable 3k/14k results to select winner")


def find_variant_env(manifest: dict, winner_variant: str) -> dict[str, str]:
    base = {str(k): str(v) for k, v in (manifest.get("base_env") or {}).items()}
    for item in manifest.get("variants", []):
        if item.get("name") == winner_variant:
            env = {str(k): str(v) for k, v in (item.get("env") or {}).items()}
            out = dict(base)
            out.update(env)
            return out
    raise RuntimeError(f"Winner variant not found in manifest: {winner_variant}")


def write_selection_artifacts(
    winner_variant: str,
    source: str,
    ranked: list[dict],
    winner_env: dict[str, str],
    status: dict,
) -> None:
    payload = {
        "generated_at_utc": ts(),
        "winner_variant": winner_variant,
        "winner_source": source,
        "winner_env": winner_env,
        "ranked": ranked,
        "phase1_counts": {
            "results_3k": len(status.get("results_3k", [])),
            "results_14k": len(status.get("results_14k", [])),
        },
    }
    SELECTION_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Winner Selection 2026-04-03",
        "",
        f"- Generated (UTC): `{ts()}`",
        f"- Winner source: `{source}`",
        f"- Winner variant: `{winner_variant}`",
        f"- Phase1 completed rows: `3k={len(status.get('results_3k', []))}`, `14k={len(status.get('results_14k', []))}`",
        "",
        "## Winner Env",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for k in sorted(winner_env.keys()):
        lines.append(f"| `{k}` | `{winner_env[k]}` |")

    lines.extend(
        [
            "",
            "## Variant Ranking",
            "",
            "| Rank | Variant | Count | Positives | Avg Delta | Max Delta | Min Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(ranked, 1):
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['count']} | {row['positives']} | {row['avg_delta']:.4f} | {row['max_delta']:.4f} | {row['min_delta']:.4f} |"
        )

    SELECTION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rollout(winner_env: dict[str, str]) -> int:
    gpus = os.environ.get("ROLLOUT_GPUS", "0,1,2,3,4,5")
    scenes = os.environ.get("ROLLOUT_SCENES", ",".join(DEFAULT_SCENES))
    run_prefix = os.environ.get("ROLLOUT_NAMESPACE", "stellar_tube_bench12_winner_rollout_20260403")
    base_port = int(os.environ.get("ROLLOUT_BASE_PORT", "8600"))

    cmd = [
        "python",
        str(BENCH_SCRIPT),
        "--run-namespace-prefix",
        run_prefix,
        "--gpus",
        gpus,
        "--scenes",
        scenes,
        "--report-dir",
        str(ROLLOUT_REPORT_DIR),
        "--base-port",
        str(base_port),
        "--skip-if-complete",
    ]

    for k in sorted(winner_env.keys()):
        cmd.extend(["--env", f"{k}={winner_env[k]}"])

    print(f"[{ts()}] rollout start with winner variant env", flush=True)
    return run_cmd(cmd)


def main() -> int:
    if not PHASE1_STATUS_JSON.exists() or not PHASE1_MANIFEST_JSON.exists():
        raise SystemExit("phase1 status/manifest missing, cannot continue")

    wait_phase1_done()

    status = load_json(PHASE1_STATUS_JSON)
    manifest = load_json(PHASE1_MANIFEST_JSON)

    winner_variant, source, ranked = pick_winner(status)
    winner_env = find_variant_env(manifest, winner_variant)

    print(f"[{ts()}] selected winner variant: {winner_variant} from {source}", flush=True)
    if ranked:
        top = ranked[0]
        print(
            f"[{ts()}] top summary: positives={top['positives']} avg_delta={top['avg_delta']:.4f} count={top['count']}",
            flush=True,
        )

    write_selection_artifacts(winner_variant, source, ranked, winner_env, status)
    return run_rollout(winner_env)


if __name__ == "__main__":
    raise SystemExit(main())

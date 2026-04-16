#!/usr/bin/env python3
import os
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/HyperGaussian")
REPORT_LOG = PROJECT_ROOT / "reports" / "refresh_followups_20260329.log"


def log(msg: str) -> None:
    REPORT_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d_%H:%M:%S', time.gmtime())}] {msg}"
    with open(REPORT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def wait_for_path(path: Path, sleep_s: int = 20) -> None:
    while not path.exists():
        time.sleep(sleep_s)


def pick_free_gpu() -> str:
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            idx, util, mem_used, mem_total = [item.strip() for item in line.split(",")]
            rows.append((int(idx), int(util), int(mem_used), int(mem_total)))
        rows.sort(key=lambda row: (row[1], row[2]))
        best = rows[0]
        if best[1] <= 20 and best[2] <= int(best[3] * 0.45):
            return str(best[0])
        time.sleep(30)


def run_task(name: str, cmd: list[str], env_updates: dict[str, str]) -> None:
    gpu = pick_free_gpu()
    env = os.environ.copy()
    env.update(env_updates)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log(f"starting {name} on GPU {gpu}: {' '.join(cmd)}")
    with open(REPORT_LOG, "a", encoding="utf-8") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    log(f"finished {name} with code {proc.returncode}")


def main() -> None:
    split_query_root = PROJECT_ROOT / "runs" / "stellar_tube_full6_20260328_histplus_span040_sigma032" / "hypernerf" / "split-cookie" / "entitybank" / "query_guided" / "split-cookie__the_complete_cookie_phaseaware"
    split_validation = split_query_root / "final_query_render_sourcebg" / "validation.json"
    cut_run_dir = PROJECT_ROOT / "runs" / "stellar_tube_cutlemon_refresh_20260329" / "hypernerf" / "cut-lemon1"
    cut_point_cloud = cut_run_dir / "point_cloud" / "iteration_14000" / "point_cloud.ply"

    log(f"waiting for split-cookie validation: {split_validation}")
    wait_for_path(split_validation, sleep_s=10)
    run_task(
        "split-cookie refresh removal",
        ["bash", str(PROJECT_ROOT / "scripts" / "run_scene_deepfill_removal_experiment.sh"), "split-cookie"],
        {
            "DEEPFILL_SOURCE_RUN_DIR": str(PROJECT_ROOT / "runs" / "stellar_tube_full6_20260328_histplus_span040_sigma032" / "hypernerf" / "split-cookie"),
            "DEEPFILL_QUERY_ROOT": str(split_query_root),
            "GS_EXPERIMENT_STAMP": "20260329_refresh_splitcookie_base040_sigma032",
            "DEEPFILL_EXPERIMENT_TAG": "splitcookie_phaseaware_refresh_base040_sigma032",
        },
    )

    log(f"waiting for cut-lemon point cloud: {cut_point_cloud}")
    wait_for_path(cut_point_cloud, sleep_s=20)
    run_task(
        "cut-lemon refresh query pipeline",
        [
            "bash",
            str(PROJECT_ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh"),
            str(cut_run_dir),
            str(PROJECT_ROOT / "data" / "hypernerf" / "interp" / "cut-lemon1"),
            "the lemon",
            "cut_the_lemon_final",
        ],
        {},
    )
    run_task(
        "cut-lemon refresh removal",
        ["bash", str(PROJECT_ROOT / "scripts" / "run_scene_deepfill_removal_experiment.sh"), "cut-lemon1"],
        {
            "DEEPFILL_SOURCE_RUN_DIR": str(cut_run_dir),
            "DEEPFILL_QUERY_ROOT": str(cut_run_dir / "entitybank" / "query_guided" / "cut_the_lemon_final"),
            "GS_EXPERIMENT_STAMP": "20260329_refresh_cutlemon_base040_sigma032",
            "DEEPFILL_EXPERIMENT_TAG": "cutlemon_queryguided_refresh_base040_sigma032",
        },
    )


if __name__ == "__main__":
    main()

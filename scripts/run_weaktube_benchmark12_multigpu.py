#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"

SCENES: dict[str, dict[str, str]] = {
    "espresso": {"dataset": "hypernerf", "scene": "misc/espresso"},
    "americano": {"dataset": "hypernerf", "scene": "misc/americano"},
    "cut_lemon": {"dataset": "hypernerf", "scene": "interp/cut-lemon1"},
    "split_cookie": {"dataset": "hypernerf", "scene": "misc/split-cookie"},
    "keyboard": {"dataset": "hypernerf", "scene": "misc/keyboard"},
    "torchchocolate": {"dataset": "hypernerf", "scene": "interp/torchocolate"},
    "coffee_martini": {"dataset": "dynerf", "scene": "coffee_martini"},
    "flame_steak": {"dataset": "dynerf", "scene": "flame_steak"},
    "cook_spinach": {"dataset": "dynerf", "scene": "cook_spinach"},
    "cut_roasted_beef": {"dataset": "dynerf", "scene": "cut_roasted_beef"},
    "sear_steak": {"dataset": "dynerf", "scene": "sear_steak"},
    "flame_salmon": {"dataset": "dynerf", "scene": "flame_salmon_1"},
}

BEST_ENV = {
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

TRAIN_ARGS = [
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


class QueueRunner:
    def __init__(
        self,
        run_namespace_prefix: str,
        gpus: list[int],
        scene_keys: list[str],
        report_dir: Path,
        base_port: int,
        force_eval: bool,
        skip_if_complete: bool,
    ) -> None:
        self.run_namespace_prefix = run_namespace_prefix
        self.gpus = gpus
        self.scene_keys = scene_keys
        self.report_dir = report_dir
        self.base_port = base_port
        self.force_eval = force_eval
        self.skip_if_complete = skip_if_complete
        self.log_dir = report_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = report_dir / "queue_status.json"
        self.leaderboard_path = report_dir / "leaderboard.md"
        self.manifest_path = report_dir / "manifest.json"
        self.lock = threading.Lock()
        self.start_ts = datetime.now(timezone.utc)

        self.assignments: dict[int, list[str]] = self._assign_round_robin(gpus, scene_keys)
        self.status: dict[str, Any] = {
            "generated_at_utc": self.start_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "repo_root": str(REPO_ROOT),
            "run_namespace_prefix": run_namespace_prefix,
            "gpus": gpus,
            "scene_keys": scene_keys,
            "best_env": BEST_ENV,
            "assignments": {str(k): v for k, v in self.assignments.items()},
            "running": {},
            "completed": [],
            "errors": [],
        }
        self._write_manifest()
        self._flush()

    @staticmethod
    def _assign_round_robin(gpus: list[int], scene_keys: list[str]) -> dict[int, list[str]]:
        out = {gpu: [] for gpu in gpus}
        for idx, key in enumerate(scene_keys):
            gpu = gpus[idx % len(gpus)]
            out[gpu].append(key)
        return out

    def _write_manifest(self) -> None:
        payload = {
            "generated_at_utc": self.start_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "run_namespace_prefix": self.run_namespace_prefix,
            "gpus": self.gpus,
            "scene_keys": self.scene_keys,
            "best_env": BEST_ENV,
            "train_args": TRAIN_ARGS,
            "assignments": self.assignments,
        }
        self._write_text_atomic(self.manifest_path, json.dumps(payload, indent=2))

    def _flush(self) -> None:
        self._write_text_atomic(self.status_path, json.dumps(self.status, indent=2))
        self._write_leaderboard()

    @staticmethod
    def _write_text_atomic(path: Path, payload: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    def _write_leaderboard(self) -> None:
        rows = [r for r in self.status["completed"] if r.get("psnr") is not None]
        rows.sort(key=lambda x: x["psnr"], reverse=True)
        lines = [
            "# WeakTube Benchmark-12 Leaderboard",
            "",
            f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
            f"- Namespace prefix: `{self.run_namespace_prefix}`",
            "",
            "| Rank | Scene | GPU | PSNR | SSIM | LPIPS-vgg | Time(s) | FPS | Storage(MB) | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for i, row in enumerate(rows, 1):
            lines.append(
                "| {rank} | `{scene}` | {gpu} | {psnr:.4f} | {ssim:.4f} | {lpips:.4f} | {time_s:.1f} | {fps:.2f} | {storage:.1f} | {status} |".format(
                    rank=i,
                    scene=row.get("scene_key", "n/a"),
                    gpu=row.get("gpu", -1),
                    psnr=float(row.get("psnr") or 0.0),
                    ssim=float(row.get("ssim") or 0.0),
                    lpips=float(row.get("lpips") or 0.0),
                    time_s=float(row.get("time_seconds") or 0.0),
                    fps=float(row.get("fps") or 0.0),
                    storage=float(row.get("storage_mb") or 0.0),
                    status=row.get("status", "n/a"),
                )
            )
        self._write_text_atomic(self.leaderboard_path, "\n".join(lines) + "\n")

    @staticmethod
    def _scene_name(scene_path: str) -> str:
        return scene_path.split("/")[-1]

    def _run_namespace(self, scene_key: str) -> str:
        return f"{self.run_namespace_prefix}_{scene_key}".replace("-", "_")

    def _run_dir(self, dataset: str, scene_path: str, namespace: str) -> Path:
        return REPO_ROOT / "runs" / namespace / dataset / self._scene_name(scene_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _collect_row(self, scene_key: str, dataset: str, scene_path: str, gpu: int, namespace: str, run_dir: Path, status: str) -> dict[str, Any]:
        metrics = self._read_json(run_dir / "metrics.json") or {}
        results = self._read_json(run_dir / "results.json") or {}

        psnr = metrics.get("PSNR")
        ssim = metrics.get("SSIM")
        lpips = metrics.get("LPIPS-vgg")
        fps = metrics.get("render_fps")
        train_s = metrics.get("train_seconds")
        render_s = metrics.get("render_seconds")
        storage_bytes = metrics.get("storage_bytes")

        if (psnr is None or ssim is None or lpips is None) and isinstance(results, dict):
            block = results.get("ours_14000") if "ours_14000" in results else None
            if block is None and results:
                k = sorted(results.keys())[-1]
                block = results.get(k)
            if isinstance(block, dict):
                psnr = psnr if psnr is not None else block.get("PSNR")
                ssim = ssim if ssim is not None else block.get("SSIM")
                lpips = lpips if lpips is not None else block.get("LPIPS-vgg")

        storage_mb = None
        if isinstance(storage_bytes, (int, float)):
            storage_mb = float(storage_bytes) / (1024.0 * 1024.0)
        else:
            ckpt = run_dir / "point_cloud" / "iteration_14000" / "point_cloud.ply"
            if ckpt.exists():
                storage_mb = ckpt.stat().st_size / (1024.0 * 1024.0)

        t_total = None
        if isinstance(train_s, (int, float)) or isinstance(render_s, (int, float)):
            t_total = float(train_s or 0.0) + float(render_s or 0.0)

        row = {
            "scene_key": scene_key,
            "dataset": dataset,
            "scene": scene_path,
            "scene_name": self._scene_name(scene_path),
            "gpu": gpu,
            "namespace": namespace,
            "run_dir": str(run_dir),
            "status": status,
            "psnr": float(psnr) if isinstance(psnr, (int, float)) else None,
            "ssim": float(ssim) if isinstance(ssim, (int, float)) else None,
            "lpips": float(lpips) if isinstance(lpips, (int, float)) else None,
            "fps": float(fps) if isinstance(fps, (int, float)) else None,
            "time_seconds": t_total,
            "train_seconds": float(train_s) if isinstance(train_s, (int, float)) else None,
            "render_seconds": float(render_s) if isinstance(render_s, (int, float)) else None,
            "storage_mb": storage_mb,
            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        return row

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _run_cmd(self, cmd: list[str], env: dict[str, str], log_file: Path) -> int:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{self._ts()}] $ {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
            return proc.wait()

    def _is_complete(self, run_dir: Path) -> bool:
        ckpt = run_dir / "point_cloud" / "iteration_14000" / "point_cloud.ply"
        metrics = run_dir / "metrics.json"
        results = run_dir / "results.json"
        return ckpt.exists() and metrics.exists() and results.exists()

    def _worker(self, gpu: int, scene_keys: list[str]) -> None:
        for idx, scene_key in enumerate(scene_keys):
            cfg = SCENES[scene_key]
            dataset = cfg["dataset"]
            scene_path = cfg["scene"]
            namespace = self._run_namespace(scene_key)
            run_dir = self._run_dir(dataset, scene_path, namespace)
            scene_tag = f"{dataset}/{scene_path}"
            port = self.base_port + (gpu * 100) + idx
            log_file = self.log_dir / f"gpu{gpu}_{scene_key}.log"

            with self.lock:
                self.status["running"][str(gpu)] = {
                    "scene_key": scene_key,
                    "dataset": dataset,
                    "scene": scene_path,
                    "namespace": namespace,
                    "run_dir": str(run_dir),
                    "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                self._flush()

            base_env = dict(os.environ)
            base_env.update(BEST_ENV)
            base_env["GS_RUN_NAMESPACE"] = namespace
            base_env["GS_PORT"] = str(port)
            base_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            status = "ok"
            try:
                if self.skip_if_complete and self._is_complete(run_dir):
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{self._ts()}] skip complete {scene_tag}\n")
                else:
                    train_cmd = [
                        "bash",
                        str(TRAIN_SCRIPT),
                        dataset,
                        scene_path,
                        *TRAIN_ARGS,
                    ]
                    if not (run_dir / "point_cloud" / "iteration_14000" / "point_cloud.ply").exists():
                        rc = self._run_cmd(train_cmd, base_env, log_file)
                        if rc != 0:
                            raise RuntimeError(f"train failed rc={rc}")
                    else:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"[{self._ts()}] checkpoint exists, skip train {scene_tag}\n")

                    need_eval = self.force_eval or not (run_dir / "results.json").exists()
                    if need_eval:
                        eval_cmd = ["bash", str(EVAL_SCRIPT), dataset, scene_path]
                        rc = self._run_cmd(eval_cmd, base_env, log_file)
                        if rc != 0:
                            raise RuntimeError(f"eval failed rc={rc}")
                    else:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"[{self._ts()}] results exists, skip eval {scene_tag}\n")
            except Exception as exc:
                status = "error"
                with self.lock:
                    self.status["errors"].append(
                        {
                            "scene_key": scene_key,
                            "dataset": dataset,
                            "scene": scene_path,
                            "gpu": gpu,
                            "namespace": namespace,
                            "run_dir": str(run_dir),
                            "error": str(exc),
                            "at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        }
                    )

            row = self._collect_row(scene_key, dataset, scene_path, gpu, namespace, run_dir, status)
            with self.lock:
                self.status["completed"].append(row)
                self.status["running"].pop(str(gpu), None)
                self._flush()

    def run(self) -> int:
        threads = []
        for gpu, keys in self.assignments.items():
            if not keys:
                continue
            t = threading.Thread(target=self._worker, args=(gpu, keys), daemon=False)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        with self.lock:
            self.status["finished_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            self._flush()

        return 0 if not self.status["errors"] else 1



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WeakTube benchmark-12 queue on multi-GPU host.")
    parser.add_argument("--run-namespace-prefix", default="stellar_tube_bench12_best04034_lrlow_20260402")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU indices, e.g. 0,1,2")
    parser.add_argument("--scenes", required=True, help="Comma-separated scene keys")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--base-port", type=int, default=7800)
    parser.add_argument("--force-eval", action="store_true", help="Always run eval even when results.json exists")
    parser.add_argument("--skip-if-complete", action="store_true", help="Skip scene if 14k ckpt+metrics+results exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    unknown = [s for s in scenes if s not in SCENES]
    if unknown:
        raise SystemExit(f"Unknown scene keys: {unknown}")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    runner = QueueRunner(
        run_namespace_prefix=args.run_namespace_prefix,
        gpus=gpus,
        scene_keys=scenes,
        report_dir=report_dir,
        base_port=args.base_port,
        force_eval=args.force_eval,
        skip_if_complete=args.skip_if_complete,
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())

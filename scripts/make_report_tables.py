import argparse
import json
import os
from pathlib import Path


def iter_metrics(root):
    for path in Path(root).rglob("metrics.json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rel = path.relative_to(root)
        yield rel.parent, payload


def make_table(rows):
    header = "| Run | PSNR | SSIM | LPIPS-vgg | FPS | Train s | Warp non-uniformity | Temporal scale mean | Temporal speed mean | Temporal accel mean |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = [header, sep]
    for run_name, payload in rows:
        warp_summary = payload.get("warp_summary") or {}
        temporal_summary = payload.get("temporal_param_summary") or {}
        body.append(
            "| {run} | {psnr} | {ssim} | {lpips} | {fps} | {train} | {warp} | {scale} | {speed} | {accel} |".format(
                run=run_name,
                psnr=payload.get("PSNR", "n/a"),
                ssim=payload.get("SSIM", "n/a"),
                lpips=payload.get("LPIPS-vgg", "n/a"),
                fps=payload.get("render_fps", "n/a"),
                train=payload.get("train_seconds", "n/a"),
                warp=warp_summary.get("non_uniformity", "n/a"),
                scale=temporal_summary.get("scale_mean", "n/a"),
                speed=temporal_summary.get("speed_mean", "n/a"),
                accel=temporal_summary.get("acceleration_mean", "n/a"),
            )
        )
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "..", "runs"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [(str(run_dir), payload) for run_dir, payload in sorted(iter_metrics(args.root))]
    table = make_table(rows)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(table)
    else:
        print(table)


if __name__ == "__main__":
    main()

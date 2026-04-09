#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", nargs="+", required=True, help="queue_status.json paths")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = []
    src_summary = []
    for p in args.status:
        path = Path(p)
        data = load_json(path)
        completed = data.get("completed", [])
        errors = data.get("errors", [])
        src_summary.append(
            {
                "path": str(path),
                "host": data.get("host"),
                "completed": len(completed),
                "errors": len(errors),
                "running": len(data.get("running", {})),
            }
        )
        for r in completed:
            row = dict(r)
            row["source_status"] = str(path)
            rows.append(row)

    rows.sort(key=lambda x: (x.get("psnr") is None, -(x.get("psnr") or -1e9), x.get("scene_key", "")))

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sources": src_summary,
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    fieldnames = [
        "scene_key",
        "dataset",
        "scene",
        "status",
        "psnr",
        "ssim",
        "lpips",
        "time_seconds",
        "fps",
        "storage_mb",
        "gpu",
        "namespace",
        "run_dir",
        "source_status",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    lines = [
        "# WeakTube Benchmark-12 Dashboard",
        "",
        f"- Generated (UTC): `{out_payload['generated_at_utc']}`",
        "",
        "## Sources",
        "",
        "| Source | Host | Completed | Running | Errors |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for s in src_summary:
        lines.append(
            f"| `{s['path']}` | `{s.get('host')}` | {s['completed']} | {s['running']} | {s['errors']} |"
        )

    lines += [
        "",
        "## Results",
        "",
        "| Rank | Scene | PSNR | SSIM | LPIPS | Time(s) | FPS | Storage(MB) | Status | Host Status File |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    rank = 1
    for r in rows:
        psnr = r.get("psnr")
        ssim = r.get("ssim")
        lpips = r.get("lpips")
        tsec = r.get("time_seconds")
        fps = r.get("fps")
        storage = r.get("storage_mb")
        lines.append(
            "| {rank} | `{scene}` | {psnr} | {ssim} | {lpips} | {time_s} | {fps} | {storage} | {status} | `{src}` |".format(
                rank=rank,
                scene=r.get("scene_key", "n/a"),
                psnr=(f"{psnr:.4f}" if isinstance(psnr, (int, float)) else "n/a"),
                ssim=(f"{ssim:.4f}" if isinstance(ssim, (int, float)) else "n/a"),
                lpips=(f"{lpips:.4f}" if isinstance(lpips, (int, float)) else "n/a"),
                time_s=(f"{tsec:.1f}" if isinstance(tsec, (int, float)) else "n/a"),
                fps=(f"{fps:.2f}" if isinstance(fps, (int, float)) else "n/a"),
                storage=(f"{storage:.1f}" if isinstance(storage, (int, float)) else "n/a"),
                status=r.get("status", "n/a"),
                src=r.get("source_status", "n/a"),
            )
        )
        rank += 1

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

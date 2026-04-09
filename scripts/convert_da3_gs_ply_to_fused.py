#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def sh_to_rgb(sh_dc: np.ndarray) -> np.ndarray:
    return np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)


def find_gs_ply(da3_output_dir: Path) -> Path:
    candidates = sorted((da3_output_dir / "gs_ply").glob("*.ply"))
    if not candidates:
        raise FileNotFoundError(f"No gs_ply/*.ply file found under {da3_output_dir}")
    return candidates[0]


def select_points(vertices, max_points: int, seed: int) -> np.ndarray:
    total = len(vertices)
    if max_points <= 0 or total <= max_points:
        return np.arange(total, dtype=np.int64)

    names = set(vertices.data.dtype.names or ())
    if "opacity" in names:
        opacity = np.asarray(vertices["opacity"], dtype=np.float32)
        return np.argsort(-opacity, kind="stable")[:max_points]

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_points, replace=False))


def convert_ply(source_ply: Path, target_ply: Path, max_points: int, seed: int) -> dict:
    ply = PlyData.read(str(source_ply))
    vertices = ply["vertex"]
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [name for name in required if name not in vertices.data.dtype.names]
    if missing:
        raise KeyError(f"Missing DA3 Gaussian fields: {missing}")

    selection = select_points(vertices, max_points=max_points, seed=seed)
    xyz = np.stack([vertices["x"][selection], vertices["y"][selection], vertices["z"][selection]], axis=1).astype(np.float32)
    sh_dc = np.stack([
        vertices["f_dc_0"][selection],
        vertices["f_dc_1"][selection],
        vertices["f_dc_2"][selection],
    ], axis=1).astype(np.float32)
    rgb = (sh_to_rgb(sh_dc) * 255.0).astype(np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "f4"),
        ("green", "f4"),
        ("blue", "f4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    target_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(target_ply))

    return {
        "source_ply": str(source_ply),
        "target_ply": str(target_ply),
        "source_points": int(len(vertices)),
        "num_points": int(xyz.shape[0]),
        "selection_mode": "opacity_topk" if "opacity" in (vertices.data.dtype.names or ()) else "random",
        "xyz_min": xyz.min(axis=0).astype(float).tolist(),
        "xyz_max": xyz.max(axis=0).astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--da3-output", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--source-ply")
    parser.add_argument("--target-name", default="fused.ply")
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    da3_output_dir = Path(args.da3_output)
    scene_dir = Path(args.scene_dir)
    source_ply = Path(args.source_ply) if args.source_ply else find_gs_ply(da3_output_dir)
    target_ply = scene_dir / args.target_name

    summary = convert_ply(source_ply, target_ply, max_points=args.max_points, seed=args.seed)
    manifest_path = scene_dir / "bootstrap_manifest.json"
    manifest = {
        "schema_version": 1,
        "bootstrap_type": "da3_gs_ply",
        **summary,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(target_ply)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData, PlyElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from export_pointcloud_video import export_pointcloud_video


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not path.exists():
        return payload
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            payload[key.strip()] = value.lower() == "true"
            continue
        try:
            if "." in value:
                payload[key.strip()] = float(value)
            else:
                payload[key.strip()] = int(value)
            continue
        except ValueError:
            payload[key.strip()] = value
    return payload


def _find_latest_iteration_dir(run_dir: Path) -> tuple[int, Path]:
    point_cloud_root = run_dir / "point_cloud"
    candidates: list[tuple[int, Path]] = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        candidates.append((iteration, child))
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directory found under {point_cloud_root}")
    return max(candidates, key=lambda item: item[0])


def _rewrite_cfg_args(cfg_text: str, target_run_dir: Path) -> str:
    target_str = str(target_run_dir)
    updated = re.sub(r"model_path='[^']*'", f"model_path='{target_str}'", cfg_text)
    updated = re.sub(r'model_path="[^"]*"', f'model_path="{target_str}"', updated)
    return updated


def _load_query_selection(query_root: Path) -> dict[str, Any]:
    selection_path = query_root / "query_worldtube_run" / "entitybank" / "selected_query_qwen.json"
    query_entities_path = query_root / "query_entitybank" / "entities.json"
    selection_payload = _read_json(selection_path)
    query_entities_payload = _read_json(query_entities_path)
    query_entity_map = {
        int(entity["id"]): entity
        for entity in query_entities_payload.get("entities", [])
    }

    selected_items = selection_payload.get("selected", [])
    if not selected_items:
        raise ValueError(f"No selected entities found in {selection_path}")

    selected_entity_summaries = []
    selected_gaussian_ids: list[np.ndarray] = []
    for item in selected_items:
        entity_id = int(item["id"])
        entity = query_entity_map.get(entity_id)
        if entity is None:
            raise KeyError(f"Selected query entity {entity_id} was not found in {query_entities_path}")
        gaussian_ids = np.asarray(entity.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
        if gaussian_ids.size == 0:
            raise ValueError(f"Selected query entity {entity_id} has no gaussian_ids")
        selected_gaussian_ids.append(gaussian_ids)
        selected_entity_summaries.append(
            {
                "query_entity_id": entity_id,
                "role": str(item.get("role", "entity")),
                "confidence": float(item.get("confidence", 0.0)),
                "segments": item.get("segments", []),
                "proposal_alias": entity.get("proposal_alias"),
                "proposal_phase": entity.get("proposal_phase"),
                "proposal_variant": entity.get("proposal_variant"),
                "num_gaussians": int(gaussian_ids.size),
            }
        )

    merged_ids = np.unique(np.concatenate(selected_gaussian_ids, axis=0)).astype(np.int64)
    return {
        "selection_path": str(selection_path),
        "query": selection_payload.get("query", ""),
        "notes": selection_payload.get("notes", ""),
        "selected_entities": selected_entity_summaries,
        "selected_gaussian_ids": merged_ids,
    }


def _load_proposal_alias_selection(
    proposal_entities_path: Path,
    proposal_alias: str | None,
    proposal_entity_id: int | None,
    query_text: str,
    segment_start: int | None,
    segment_end: int | None,
    notes: str,
) -> dict[str, Any]:
    proposal_payload = _read_json(proposal_entities_path)
    matched_entity = None
    entities = proposal_payload.get("entities", [])
    if proposal_entity_id is not None:
        for entity in entities:
            if int(entity.get("id", -1)) == int(proposal_entity_id):
                matched_entity = entity
                break
        if matched_entity is None:
            raise KeyError(f"proposal_entity_id={proposal_entity_id} not found in {proposal_entities_path}")
    else:
        alias_norm = str(proposal_alias).strip()
        for entity in entities:
            if str(entity.get("proposal_alias", "")).strip() == alias_norm:
                matched_entity = entity
                break
            if str(entity.get("static_text", "")).strip() == alias_norm:
                matched_entity = entity
                break
        if matched_entity is None:
            raise KeyError(f"proposal_alias='{proposal_alias}' not found in {proposal_entities_path}")

    gaussian_ids = np.asarray(matched_entity.get("gaussian_ids", []), dtype=np.int64).reshape(-1)
    if gaussian_ids.size == 0:
        raise ValueError("Matched proposal entity has no gaussian_ids")

    segments: list[list[int]] = []
    if segment_start is not None and segment_end is not None:
        segments = [[int(segment_start), int(segment_end)]]

    return {
        "selection_path": str(proposal_entities_path),
        "query": str(query_text),
        "notes": str(notes),
        "selected_entities": [
            {
                "query_entity_id": int(matched_entity["id"]),
                "role": "entity",
                "confidence": float(matched_entity.get("quality", 1.0)),
                "segments": segments,
                "proposal_alias": matched_entity.get("proposal_alias"),
                "proposal_phase": matched_entity.get("proposal_phase"),
                "proposal_variant": matched_entity.get("proposal_variant"),
                "num_gaussians": int(gaussian_ids.size),
            }
        ],
        "selected_gaussian_ids": np.unique(gaussian_ids).astype(np.int64),
    }


def _filter_ply(source_ply: Path, target_ply: Path, keep_mask: np.ndarray) -> int:
    ply = PlyData.read(str(source_ply))
    vertex_data = ply["vertex"].data
    if keep_mask.shape[0] != len(vertex_data):
        raise ValueError(f"PLY mask shape mismatch: {keep_mask.shape[0]} vs {len(vertex_data)}")
    filtered = vertex_data[keep_mask]
    if len(filtered) == 0:
        raise ValueError(f"Filtering {source_ply} removed every Gaussian")
    target_ply.parent.mkdir(parents=True, exist_ok=True)
    output = PlyData([PlyElement.describe(filtered, "vertex")], text=ply.text, byte_order=ply.byte_order)
    output.write(str(target_ply))
    return int(len(filtered))


def _filter_tensor_payload(source_path: Path, target_path: Path, keep_mask: np.ndarray) -> None:
    payload = torch.load(source_path, map_location="cpu")
    mask_tensor = torch.from_numpy(keep_mask.astype(np.bool_))
    if isinstance(payload, dict):
        filtered: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == keep_mask.shape[0]:
                filtered[key] = value[mask_tensor]
            else:
                filtered[key] = value
        torch.save(filtered, target_path)
        return
    if isinstance(payload, torch.Tensor) and payload.ndim >= 1 and payload.shape[0] == keep_mask.shape[0]:
        torch.save(payload[mask_tensor], target_path)
        return
    torch.save(payload, target_path)


def _filter_trajectory_samples(source_path: Path, target_path: Path, keep_mask: np.ndarray) -> None:
    payload = np.load(source_path)
    filtered: dict[str, np.ndarray] = {}
    total = int(keep_mask.shape[0])
    for key in payload.files:
        value = payload[key]
        if value.ndim >= 1 and value.shape[0] == total:
            filtered[key] = value[keep_mask]
        else:
            filtered[key] = value
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target_path, **filtered)


def _write_subset_entities(target_path: Path, source_gaussian_ids: np.ndarray, title: str) -> None:
    preview_count = min(int(source_gaussian_ids.shape[0]), 128)
    payload = {
        "schema_version": 1,
        "entities": [
            {
                "id": 0,
                "label": title,
                "static_text": title,
                "num_gaussians": int(source_gaussian_ids.shape[0]),
                "gaussian_id_range": [0, max(int(source_gaussian_ids.shape[0]) - 1, 0)],
                "source_gaussian_ids_preview": source_gaussian_ids[:preview_count].astype(np.int64).tolist(),
            }
        ],
    }
    _write_json(target_path, payload)


def _prepare_subset_run(
    source_run_dir: Path,
    output_run_dir: Path,
    keep_mask: np.ndarray,
    source_gaussian_ids: np.ndarray,
    title: str,
    subset_summary: dict[str, Any],
) -> dict[str, Any]:
    if output_run_dir.exists():
        shutil.rmtree(output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    cfg_text = (source_run_dir / "cfg_args").read_text(encoding="utf-8")
    (output_run_dir / "cfg_args").write_text(_rewrite_cfg_args(cfg_text, output_run_dir), encoding="utf-8")
    shutil.copy2(source_run_dir / "config.yaml", output_run_dir / "config.yaml")

    source_iteration, source_iteration_dir = _find_latest_iteration_dir(source_run_dir)
    target_iteration_dir = output_run_dir / "point_cloud" / f"iteration_{source_iteration}"
    target_iteration_dir.mkdir(parents=True, exist_ok=True)

    kept_count = _filter_ply(
        source_iteration_dir / "point_cloud.ply",
        target_iteration_dir / "point_cloud.ply",
        keep_mask,
    )
    for filename in ("temporal_params.pth", "deformation_table.pth", "deformation_accum.pth"):
        _filter_tensor_payload(
            source_iteration_dir / filename,
            target_iteration_dir / filename,
            keep_mask,
        )
    os.symlink(source_iteration_dir / "deformation.pth", target_iteration_dir / "deformation.pth")
    os.symlink(source_run_dir / "temporal_warp", output_run_dir / "temporal_warp")

    entitybank_dir = output_run_dir / "entitybank"
    _filter_trajectory_samples(
        source_run_dir / "entitybank" / "trajectory_samples.npz",
        entitybank_dir / "trajectory_samples.npz",
        keep_mask,
    )
    _write_subset_entities(entitybank_dir / "entities.json", source_gaussian_ids, title)

    subset_payload = {
        "schema_version": 1,
        "title": title,
        "source_run_dir": str(source_run_dir),
        "output_run_dir": str(output_run_dir),
        "source_iteration": int(source_iteration),
        "num_source_gaussians": int(keep_mask.shape[0]),
        "num_subset_gaussians": int(kept_count),
        **subset_summary,
    }
    _write_json(output_run_dir / "subset_summary.json", subset_payload)
    return subset_payload


def _env_with_pythonpath() -> dict[str, str]:
    env = os.environ.copy()
    repo_pythonpath = f"{PROJECT_ROOT}:{EXTERNAL_ROOT}"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{repo_pythonpath}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = repo_pythonpath
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _worldtube_override_args(run_dir: Path) -> list[str]:
    config = _read_simple_yaml(run_dir / "config.yaml")
    bool_keys = [
        "warp_enabled",
        "temporal_extent_enabled",
        "temporal_acceleration_enabled",
        "temporal_worldtube_enabled",
        "temporal_worldtube_adaptive_support",
        "spacetime_aware_optimization",
    ]
    value_keys = [
        "temporal_warp_type",
        "temporal_gate_sharpness",
        "temporal_drift_scale",
        "temporal_gate_mix",
        "temporal_drift_mix",
        "temporal_velocity_reg_weight",
        "temporal_acceleration_reg_weight",
        "temporal_worldtube_samples",
        "temporal_worldtube_span",
        "temporal_worldtube_sigma",
        "temporal_worldtube_opacity_mix",
        "temporal_worldtube_scale_mix",
        "temporal_worldtube_reg_weight",
        "temporal_worldtube_ratio_weight",
        "temporal_worldtube_support_min",
        "temporal_worldtube_support_max",
        "temporal_worldtube_ratio_target",
        "temporal_worldtube_ratio_tolerance",
        "temporal_worldtube_densify_weight",
        "temporal_worldtube_split_shrink",
        "temporal_worldtube_support_gain",
        "temporal_worldtube_support_min_factor",
        "temporal_worldtube_support_max_factor",
        "temporal_worldtube_opacity_floor",
        "temporal_worldtube_visibility_mix",
        "temporal_worldtube_integral_mix",
        "temporal_activity_weight",
        "temporal_prune_protect_quantile",
    ]

    args: list[str] = []
    for key in bool_keys:
        if bool(config.get(key, False)):
            args.append(f"--{key}")
    for key in value_keys:
        if key in config:
            args.extend([f"--{key}", str(config[key])])
    return args


def _render_test_frames(run_dir: Path) -> Path:
    render_cmd = [
        sys.executable,
        str(EXTERNAL_ROOT / "render.py"),
        "-m",
        str(run_dir),
        "--iteration",
        "-1",
        "--skip_train",
        "--skip_video",
    ]
    render_cmd.extend(_worldtube_override_args(run_dir))
    subprocess.run(
        render_cmd,
        cwd=str(PROJECT_ROOT),
        env=_env_with_pythonpath(),
        check=True,
    )
    test_root = run_dir / "test"
    candidates = sorted(test_root.glob("ours_*/renders"))
    if not candidates:
        raise FileNotFoundError(f"No test renders were written under {test_root}")
    return candidates[-1]


def _resolve_panel_frames(selection_segments: list[list[int]], frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if not selection_segments:
        return [0, frame_count // 2, frame_count - 1]
    start = max(int(selection_segments[0][0]), 0)
    end = min(int(selection_segments[0][1]), frame_count - 1)
    mid = start + (end - start) // 2
    post = min(end + max(frame_count // 4, 8), frame_count - 1)
    frames = []
    for index in (start, mid, end, post):
        if index not in frames:
            frames.append(index)
    return frames


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _build_triptych_panel(
    full_render_dir: Path,
    extract_render_dir: Path,
    remove_render_dir: Path,
    output_path: Path,
    frame_indices: list[int],
) -> Path:
    full_frames = sorted(full_render_dir.glob("*.png"))
    extract_frames = sorted(extract_render_dir.glob("*.png"))
    remove_frames = sorted(remove_render_dir.glob("*.png"))
    frame_count = min(len(full_frames), len(extract_frames), len(remove_frames))
    if frame_count == 0:
        raise ValueError("Triptych panel requires non-empty render directories")

    safe_indices = [min(max(int(index), 0), frame_count - 1) for index in frame_indices]
    selected_rows = []
    for index in safe_indices:
        selected_rows.append(
            [
                ("Full Scene", Image.open(full_frames[index]).convert("RGB")),
                ("Complete Cookie", Image.open(extract_frames[index]).convert("RGB")),
                ("Scene Without Cookie", Image.open(remove_frames[index]).convert("RGB")),
            ]
        )

    image_width, image_height = selected_rows[0][0][1].size
    title_font = _load_font(26)
    label_font = _load_font(22)
    cell_padding = 20
    header_height = 54
    row_label_height = 40
    canvas_width = 3 * image_width + 4 * cell_padding
    canvas_height = header_height + len(selected_rows) * (image_height + row_label_height + cell_padding) + cell_padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    x = cell_padding
    for label in ("Full Scene", "Complete Cookie", "Scene Without Cookie"):
        draw.text((x, 14), label, fill=(25, 25, 25), font=title_font)
        x += image_width + cell_padding

    y = header_height
    for row_index, row in enumerate(selected_rows):
        draw.text((cell_padding, y), f"Test Frame {safe_indices[row_index]:03d}", fill=(55, 55, 55), font=label_font)
        image_y = y + row_label_height
        x = cell_padding
        for _label, image in row:
            canvas.paste(image, (x, image_y))
            x += image_width + cell_padding
            image.close()
        y += row_label_height + image_height + cell_padding

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_query_removal_bundle(
    source_run_dir: Path,
    query_root: Path | None,
    output_root: Path,
    max_points: int,
    frame_stride: int,
    fps: int,
    render_test: bool,
    proposal_entities_path: Path | None = None,
    proposal_alias: str | None = None,
    proposal_entity_id: int | None = None,
    query_text: str | None = None,
    segment_start: int | None = None,
    segment_end: int | None = None,
    selection_notes: str | None = None,
) -> Path:
    if proposal_entities_path is not None:
        if proposal_alias is None and proposal_entity_id is None:
            raise ValueError("proposal_alias or proposal_entity_id is required when proposal_entities_path is provided")
        query_summary = _load_proposal_alias_selection(
            proposal_entities_path=proposal_entities_path,
            proposal_alias=proposal_alias,
            proposal_entity_id=proposal_entity_id,
            query_text=query_text or proposal_alias,
            segment_start=segment_start,
            segment_end=segment_end,
            notes=selection_notes or f"Proposal alias selection for {proposal_alias}",
        )
    else:
        if query_root is None:
            raise ValueError("query_root is required when proposal_entities_path is not provided")
        query_summary = _load_query_selection(query_root)
    selected_gaussian_ids = np.asarray(query_summary["selected_gaussian_ids"], dtype=np.int64)

    iteration, iteration_dir = _find_latest_iteration_dir(source_run_dir)
    ply = PlyData.read(str(iteration_dir / "point_cloud.ply"))
    num_gaussians = len(ply["vertex"].data)
    extract_mask = np.zeros((num_gaussians,), dtype=bool)
    extract_mask[selected_gaussian_ids] = True
    remove_mask = ~extract_mask

    visual_dir = output_root / "visuals"
    complete_cookie_run = output_root / "complete_cookie" / "run"
    removed_scene_run = output_root / "scene_without_complete_cookie" / "run"

    output_root.mkdir(parents=True, exist_ok=True)
    extract_summary = _prepare_subset_run(
        source_run_dir=source_run_dir,
        output_run_dir=complete_cookie_run,
        keep_mask=extract_mask,
        source_gaussian_ids=selected_gaussian_ids,
        title="complete cookie",
        subset_summary={
            "query_root": None if query_root is None else str(query_root),
            "proposal_entities_path": None if proposal_entities_path is None else str(proposal_entities_path),
            "proposal_alias": proposal_alias,
            "proposal_entity_id": proposal_entity_id,
            "query": query_summary["query"],
            "selection_notes": query_summary["notes"],
            "selected_entities": query_summary["selected_entities"],
            "mode": "extract_complete_cookie",
        },
    )
    removed_summary = _prepare_subset_run(
        source_run_dir=source_run_dir,
        output_run_dir=removed_scene_run,
        keep_mask=remove_mask,
        source_gaussian_ids=np.where(remove_mask)[0],
        title="scene without complete cookie",
        subset_summary={
            "query_root": None if query_root is None else str(query_root),
            "proposal_entities_path": None if proposal_entities_path is None else str(proposal_entities_path),
            "proposal_alias": proposal_alias,
            "proposal_entity_id": proposal_entity_id,
            "query": query_summary["query"],
            "selection_notes": query_summary["notes"],
            "selected_entities": query_summary["selected_entities"],
            "removed_source_gaussian_ids": selected_gaussian_ids.astype(np.int64).tolist(),
            "mode": "remove_complete_cookie",
        },
    )

    full_scene_video = export_pointcloud_video(
        run_dir=source_run_dir,
        output_path=visual_dir / "full_scene_pointcloud.mp4",
        max_points=max_points,
        frame_stride=frame_stride,
        seed=0,
        fps=fps,
        azimuth_step=4.0,
        elevation=20.0,
        point_size=0.8,
    )
    extract_video = export_pointcloud_video(
        run_dir=complete_cookie_run,
        output_path=visual_dir / "complete_cookie_pointcloud.mp4",
        max_points=min(max_points, max(int(extract_summary["num_subset_gaussians"]), 1024)),
        frame_stride=frame_stride,
        seed=0,
        fps=fps,
        azimuth_step=4.0,
        elevation=20.0,
        point_size=1.2,
    )
    removed_video = export_pointcloud_video(
        run_dir=removed_scene_run,
        output_path=visual_dir / "scene_without_complete_cookie_pointcloud.mp4",
        max_points=max_points,
        frame_stride=frame_stride,
        seed=0,
        fps=fps,
        azimuth_step=4.0,
        elevation=20.0,
        point_size=0.8,
    )

    full_render_dir = None
    extract_render_dir = None
    removed_render_dir = None
    triptych_path = None
    if render_test:
        original_candidates = sorted((source_run_dir / "test").glob("ours_*/renders"))
        if not original_candidates:
            raise FileNotFoundError(f"No cached full-scene test renders found under {source_run_dir / 'test'}")
        full_render_dir = original_candidates[-1]
        extract_render_dir = _render_test_frames(complete_cookie_run)
        removed_render_dir = _render_test_frames(removed_scene_run)
        frame_count = len(sorted(full_render_dir.glob("*.png")))
        selection_segments = query_summary["selected_entities"][0].get("segments", [])
        panel_frames = _resolve_panel_frames(selection_segments, frame_count)
        triptych_path = _build_triptych_panel(
            full_render_dir=full_render_dir,
            extract_render_dir=extract_render_dir,
            remove_render_dir=removed_render_dir,
            output_path=visual_dir / "render_triptych.png",
            frame_indices=panel_frames,
        )

    bundle_summary = {
        "schema_version": 1,
        "source_run_dir": str(source_run_dir),
        "query_root": None if query_root is None else str(query_root),
        "proposal_entities_path": None if proposal_entities_path is None else str(proposal_entities_path),
        "proposal_alias": proposal_alias,
        "proposal_entity_id": proposal_entity_id,
        "query": query_summary["query"],
        "selection_notes": query_summary["notes"],
        "source_iteration": int(iteration),
        "num_source_gaussians": int(num_gaussians),
        "selected_gaussian_count": int(selected_gaussian_ids.shape[0]),
        "remaining_gaussian_count": int(remove_mask.sum()),
        "selected_entities": query_summary["selected_entities"],
        "artifacts": {
            "full_scene_video": str(full_scene_video),
            "complete_cookie_video": str(extract_video),
            "scene_without_cookie_video": str(removed_video),
            "complete_cookie_run": str(complete_cookie_run),
            "scene_without_cookie_run": str(removed_scene_run),
            "full_render_dir": None if full_render_dir is None else str(full_render_dir),
            "complete_cookie_render_dir": None if extract_render_dir is None else str(extract_render_dir),
            "scene_without_cookie_render_dir": None if removed_render_dir is None else str(removed_render_dir),
            "render_triptych": None if triptych_path is None else str(triptych_path),
        },
    }
    _write_json(output_root / "bundle_summary.json", bundle_summary)

    readme_lines = [
        "# Split-Cookie Complete-Cookie Removal Bundle",
        "",
        f"- source run: `{source_run_dir}`",
        f"- query root: `{query_root}`",
        f"- proposal entities: `{proposal_entities_path}`",
        f"- proposal alias: `{proposal_alias}`",
        f"- proposal entity id: `{proposal_entity_id}`",
        f"- query: `{query_summary['query']}`",
        f"- selected Gaussian count: `{selected_gaussian_ids.shape[0]}`",
        f"- remaining Gaussian count: `{int(remove_mask.sum())}`",
        "",
        "## Outputs",
        "",
        f"- full scene point cloud video: `{full_scene_video}`",
        f"- complete cookie point cloud video: `{extract_video}`",
        f"- scene without complete cookie point cloud video: `{removed_video}`",
        f"- complete cookie subset run: `{complete_cookie_run}`",
        f"- scene without complete cookie run: `{removed_scene_run}`",
    ]
    if triptych_path is not None:
        readme_lines.append(f"- render triptych: `{triptych_path}`")
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--query-root", default=None)
    parser.add_argument("--proposal-entities-json", default=None)
    parser.add_argument("--proposal-alias", default=None)
    parser.add_argument("--proposal-entity-id", type=int, default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--segment-start", type=int, default=None)
    parser.add_argument("--segment-end", type=int, default=None)
    parser.add_argument("--selection-notes", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-points", type=int, default=40000)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--skip-render-test", action="store_true")
    args = parser.parse_args()

    output_root = build_query_removal_bundle(
        source_run_dir=Path(args.source_run_dir),
        query_root=None if args.query_root is None else Path(args.query_root),
        output_root=Path(args.output_root),
        max_points=int(args.max_points),
        frame_stride=int(args.frame_stride),
        fps=int(args.fps),
        render_test=not bool(args.skip_render_test),
        proposal_entities_path=None if args.proposal_entities_json is None else Path(args.proposal_entities_json),
        proposal_alias=args.proposal_alias,
        proposal_entity_id=args.proposal_entity_id,
        query_text=args.query_text,
        segment_start=args.segment_start,
        segment_end=args.segment_end,
        selection_notes=args.selection_notes,
    )
    print(output_root)


if __name__ == "__main__":
    main()

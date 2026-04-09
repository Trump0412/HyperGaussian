import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_simple_yaml(path):
    if not os.path.exists(path):
        return {}
    payload = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip()
    return payload


def parse_results(results):
    if not isinstance(results, dict) or not results:
        return {}

    flat_metrics = [
        (method_name, payload)
        for method_name, payload in results.items()
        if isinstance(payload, dict) and any(key in payload for key in ("PSNR", "SSIM", "LPIPS-vgg"))
    ]
    if flat_metrics:
        method_name, payload = sorted(flat_metrics, key=lambda item: item[0])[-1]
        merged = dict(payload)
        merged["method"] = method_name
        return merged

    nested_payload = next(iter(results.values()))
    if not isinstance(nested_payload, dict):
        return {}
    return parse_results(nested_payload)


def directory_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                continue
            total += os.path.getsize(full_path)
    return total


def read_render_fps(log_path):
    if not os.path.exists(log_path):
        return None
    fps_values = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "FPS:" in line:
                try:
                    fps_values.append(float(line.strip().split("FPS:")[-1]))
                except ValueError:
                    continue
    if not fps_values:
        return None
    return sum(fps_values) / len(fps_values)


def infer_render_frame_count(run_dir):
    test_dir = Path(run_dir) / "test"
    if not test_dir.exists():
        return 0
    counts = []
    for method_dir in test_dir.iterdir():
        renders_dir = method_dir / "renders"
        if renders_dir.is_dir():
            counts.append(len(list(renders_dir.glob("*.png"))))
    return max(counts, default=0)


def find_latest_render_dir(run_dir):
    test_dir = Path(run_dir) / "test"
    if not test_dir.exists():
        return None
    candidates = []
    for method_dir in test_dir.iterdir():
        if not method_dir.is_dir() or not method_dir.name.startswith("ours_"):
            continue
        try:
            iteration = int(method_dir.name.split("_", 1)[1])
        except ValueError:
            iteration = -1
        render_dir = method_dir / "renders"
        if render_dir.is_dir():
            candidates.append((iteration, render_dir))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def read_render_sanity_summary(run_dir, max_frames=16, max_pixels=20000):
    render_dir = find_latest_render_dir(run_dir)
    if render_dir is None:
        return None

    frame_paths = sorted(render_dir.glob("*.png"))[:max_frames]
    if not frame_paths:
        return None

    sampled = []
    constant_frames = 0
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        data = list(image.getdata())
        subset = data[: min(max_pixels, len(data))]
        unique_colors = len(set(subset))
        mean_luma = sum(sum(px) for px in subset) / (len(subset) * 3)
        sampled.append(
            {
                "frame": frame_path.name,
                "unique_colors": unique_colors,
                "mean_luma_255": float(mean_luma),
            }
        )
        if unique_colors <= 2:
            constant_frames += 1

    mean_unique = sum(item["unique_colors"] for item in sampled) / len(sampled)
    constant_fraction = constant_frames / len(sampled)
    return {
        "sample_count": len(sampled),
        "mean_unique_colors": float(mean_unique),
        "constant_frame_fraction": float(constant_fraction),
        "degenerate_constant_render": bool(constant_fraction >= 0.75),
        "sampled_frames": sampled,
    }


def find_latest_iteration_dir(root):
    if not root.exists():
        return None
    candidates = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iteration = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        candidates.append((iteration, child))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def read_temporal_param_summary(run_dir):
    latest_point_cloud = find_latest_iteration_dir(Path(run_dir) / "point_cloud")
    if latest_point_cloud is None:
        return None
    payload_path = latest_point_cloud / "temporal_params.pth"
    if not payload_path.exists():
        return None

    payload = torch.load(payload_path, map_location="cpu")
    anchor_raw = payload.get("time_anchor")
    scale_raw = payload.get("time_scale")
    velocity = payload.get("time_velocity")
    acceleration = payload.get("time_acceleration")
    if anchor_raw is None or scale_raw is None or velocity is None:
        return None
    if acceleration is None:
        acceleration = torch.zeros_like(velocity)

    anchor = torch.sigmoid(anchor_raw.float()).view(-1)
    scale = torch.nn.functional.softplus(scale_raw.float()).view(-1) + 1.0e-6
    speed = torch.linalg.norm(velocity.float(), dim=-1).view(-1)
    accel = torch.linalg.norm(acceleration.float(), dim=-1).view(-1)
    return {
        "count": int(anchor.numel()),
        "anchor_mean": float(anchor.mean().item()),
        "anchor_std": float(anchor.std(unbiased=False).item()),
        "scale_mean": float(scale.mean().item()),
        "scale_std": float(scale.std(unbiased=False).item()),
        "speed_mean": float(speed.mean().item()),
        "speed_max": float(speed.max().item()),
        "acceleration_mean": float(accel.mean().item()),
        "acceleration_max": float(accel.max().item()),
    }


def read_bootstrap_summary(run_dir):
    config = read_simple_yaml(Path(run_dir) / "config.yaml")
    source_path = config.get("source_path")
    if not source_path:
        return None
    manifest = read_json(Path(source_path) / "bootstrap_manifest.json")
    if not manifest:
        return None
    return {
        "bootstrap_type": manifest.get("bootstrap_type"),
        "source_points": manifest.get("source_points"),
        "num_points": manifest.get("num_points"),
        "selection_mode": manifest.get("selection_mode"),
    }


def read_entitybank_summary(run_dir):
    tube_bank = read_json(Path(run_dir) / "entitybank" / "tube_bank.json") or {}
    cluster_stats = read_json(Path(run_dir) / "entitybank" / "cluster_stats.json") or {}
    entities = read_json(Path(run_dir) / "entitybank" / "entities.json") or {}
    if not tube_bank and not cluster_stats and not entities:
        return None

    return {
        "num_gaussians": tube_bank.get("num_gaussians", cluster_stats.get("num_gaussians")),
        "num_frames": tube_bank.get("num_frames", entities.get("frame_count")),
        "num_clusters": cluster_stats.get("num_clusters_kept"),
        "num_entities": entities.get("num_entities"),
        "motion_score_mean": tube_bank.get("motion_score_mean"),
        "motion_score_max": tube_bank.get("motion_score_max"),
        "displacement_mean": tube_bank.get("displacement_mean"),
        "displacement_max": tube_bank.get("displacement_max"),
        "speed_mean": tube_bank.get("speed_mean"),
        "speed_max": tube_bank.get("speed_max"),
        "acceleration_mean": tube_bank.get("acceleration_mean"),
        "acceleration_max": tube_bank.get("acceleration_max"),
        "support_factor_mean": tube_bank.get("support_factor_mean"),
        "support_factor_max": tube_bank.get("support_factor_max"),
        "effective_support_mean": tube_bank.get("effective_support_mean"),
        "effective_support_max": tube_bank.get("effective_support_max"),
        "tube_ratio_mean": tube_bank.get("tube_ratio_mean"),
        "tube_ratio_max": tube_bank.get("tube_ratio_max"),
        "occupancy_mean": tube_bank.get("occupancy_mean"),
        "occupancy_max": tube_bank.get("occupancy_max"),
        "visibility_mean": tube_bank.get("visibility_mean"),
        "visibility_max": tube_bank.get("visibility_max"),
    }


def read_semantic_summary(run_dir):
    slots = read_json(Path(run_dir) / "entitybank" / "semantic_slots.json") or {}
    slot_queries = read_json(Path(run_dir) / "entitybank" / "semantic_slot_queries.json") or {}
    tracks = read_json(Path(run_dir) / "entitybank" / "semantic_tracks.json") or {}
    priors = read_json(Path(run_dir) / "entitybank" / "semantic_priors.json") or {}
    frame_queries = read_json(Path(run_dir) / "entitybank" / "semantic_frame_queries.json") or {}
    segmentation_bootstrap = read_json(Path(run_dir) / "entitybank" / "semantic_segmentation_bootstrap.json") or {}
    if not slots and not slot_queries and not tracks and not priors and not frame_queries and not segmentation_bootstrap:
        return None

    active_slot_counts = []
    moving_slot_counts = []
    dynamic_slot_counts = []
    static_slot_counts = []
    for frame in frame_queries.get("frames", []):
        active_slots = frame.get("active_slots", [])
        active_slot_counts.append(int(frame.get("num_active_slots", len(active_slots))))
        moving_slot_counts.append(sum(1 for slot in active_slots if slot.get("motion_label") == "moving"))
        dynamic_slot_counts.append(
            sum(1 for slot in active_slots if slot.get("motion_label") == "moving")
        )
        static_slot_counts.append(
            sum(1 for slot in active_slots if slot.get("motion_label") == "stationary")
        )

    return {
        "num_slots": slots.get("num_slots"),
        "num_slot_queries": len(slot_queries.get("slots", [])) if isinstance(slot_queries.get("slots"), list) else None,
        "num_tracks": tracks.get("num_tracks"),
        "num_priors": priors.get("num_priors"),
        "num_static_heads": priors.get("num_static_heads"),
        "num_dynamic_heads": priors.get("num_dynamic_heads"),
        "num_interaction_heads": priors.get("num_interaction_heads"),
        "frame_count": frame_queries.get("frame_count", slots.get("frame_count")),
        "num_bootstrap_images": segmentation_bootstrap.get("num_images"),
        "active_slots_mean": (sum(active_slot_counts) / len(active_slot_counts)) if active_slot_counts else None,
        "active_slots_max": max(active_slot_counts) if active_slot_counts else None,
        "moving_slots_mean": (sum(moving_slot_counts) / len(moving_slot_counts)) if moving_slot_counts else None,
        "moving_slots_max": max(moving_slot_counts) if moving_slot_counts else None,
        "dynamic_slots_mean": (sum(dynamic_slot_counts) / len(dynamic_slot_counts)) if dynamic_slot_counts else None,
        "dynamic_slots_max": max(dynamic_slot_counts) if dynamic_slot_counts else None,
        "static_slots_mean": (sum(static_slot_counts) / len(static_slot_counts)) if static_slot_counts else None,
        "static_slots_max": max(static_slot_counts) if static_slot_counts else None,
    }


def read_native_semantic_summary(run_dir):
    native_assignments = read_json(Path(run_dir) / "entitybank" / "native_semantic_assignments.json") or {}
    native_queries_dir = Path(run_dir) / "entitybank" / "native_queries"
    if not native_assignments and not native_queries_dir.exists():
        return None

    query_summaries = []
    if native_queries_dir.exists():
        for query_dir in sorted(native_queries_dir.iterdir()):
            if not query_dir.is_dir():
                continue
            selected = read_json(query_dir / "selected.json") or {}
            rendered_validation = read_json(query_dir / "rendered" / "validation.json") or {}
            query_summaries.append(
                {
                    "query_name": query_dir.name,
                    "selected_count": len(selected.get("selected", [])),
                    "empty": bool(selected.get("empty", False)),
                    "active_frame_count": rendered_validation.get("active_frame_count"),
                    "contact_frame_count": rendered_validation.get("contact_frame_count"),
                }
            )

    return {
        "num_assignments": native_assignments.get("num_assignments"),
        "num_tool_like": native_assignments.get("num_tool_like"),
        "num_patient_like": native_assignments.get("num_patient_like"),
        "num_support_like": native_assignments.get("num_support_like"),
        "num_agent_like": native_assignments.get("num_agent_like"),
        "num_interaction_pairs": native_assignments.get("num_interaction_pairs"),
        "num_native_queries": len(query_summaries),
        "queries": query_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = parse_results(read_json(run_dir / "results.json"))
    if not results:
        quick_metrics = read_json(run_dir / "quick_metrics.json")
        if isinstance(quick_metrics, dict):
            results = dict(quick_metrics)
    train_meta = read_json(run_dir / "train_meta.json") or {}
    render_meta = read_json(run_dir / "render_meta.json") or {}
    warp_summary = read_json(run_dir / "temporal_warp" / "latest" / "warp_summary.json")
    temporal_param_summary = read_temporal_param_summary(run_dir)
    bootstrap_summary = read_bootstrap_summary(run_dir)
    entitybank_summary = read_entitybank_summary(run_dir)
    semantic_summary = read_semantic_summary(run_dir)
    native_semantic_summary = read_native_semantic_summary(run_dir)
    render_sanity_summary = read_render_sanity_summary(run_dir)

    metrics = dict(results)
    metrics["train_seconds"] = train_meta.get("elapsed_seconds")
    metrics["render_seconds"] = render_meta.get("elapsed_seconds")
    metrics["gpu_peak_mb"] = max(train_meta.get("gpu_peak_mb", 0), render_meta.get("gpu_peak_mb", 0))
    metrics["render_fps"] = read_render_fps(run_dir / "render.log")
    if metrics["render_fps"] is None and metrics["render_seconds"]:
        frame_count = infer_render_frame_count(run_dir)
        if frame_count:
            metrics["render_fps"] = frame_count / metrics["render_seconds"]
    metrics["storage_bytes"] = directory_size(run_dir)
    metrics["warp_summary"] = warp_summary
    metrics["temporal_param_summary"] = temporal_param_summary
    metrics["bootstrap_summary"] = bootstrap_summary
    metrics["entitybank_summary"] = entitybank_summary
    metrics["semantic_summary"] = semantic_summary
    metrics["native_semantic_summary"] = native_semantic_summary
    metrics["render_sanity_summary"] = render_sanity_summary

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if args.write_summary:
        lines = [
            f"# Summary: {run_dir.name}",
            "",
            f"- Method: {metrics.get('method', 'n/a')}",
            f"- PSNR: {metrics.get('PSNR', 'n/a')}",
            f"- SSIM: {metrics.get('SSIM', 'n/a')}",
            f"- LPIPS-vgg: {metrics.get('LPIPS-vgg', 'n/a')}",
            f"- Render FPS: {metrics.get('render_fps', 'n/a')}",
            f"- Train seconds: {metrics.get('train_seconds', 'n/a')}",
            f"- GPU peak MB: {metrics.get('gpu_peak_mb', 'n/a')}",
            f"- Storage bytes: {metrics.get('storage_bytes', 'n/a')}",
        ]
        if metrics.get("sample_count") is not None and metrics.get("sample_total") is not None:
            lines.append(f"- Sampled frames: {metrics.get('sample_count')} / {metrics.get('sample_total')}")
        if warp_summary:
            lines.extend(
                [
                    f"- Warp non-uniformity: {warp_summary.get('non_uniformity', 'n/a')}",
                    f"- Warp slope range: [{warp_summary.get('slope_min', 'n/a')}, {warp_summary.get('slope_max', 'n/a')}]",
                ]
            )
        if temporal_param_summary:
            lines.extend(
                [
                    f"- Temporal anchor mean/std: {temporal_param_summary.get('anchor_mean', 'n/a')} / {temporal_param_summary.get('anchor_std', 'n/a')}",
                    f"- Temporal scale mean/std: {temporal_param_summary.get('scale_mean', 'n/a')} / {temporal_param_summary.get('scale_std', 'n/a')}",
                    f"- Temporal speed mean/max: {temporal_param_summary.get('speed_mean', 'n/a')} / {temporal_param_summary.get('speed_max', 'n/a')}",
                    f"- Temporal acceleration mean/max: {temporal_param_summary.get('acceleration_mean', 'n/a')} / {temporal_param_summary.get('acceleration_max', 'n/a')}",
                ]
            )
        if bootstrap_summary:
            lines.extend(
                [
                    f"- Bootstrap type: {bootstrap_summary.get('bootstrap_type', 'n/a')}",
                    f"- Bootstrap points kept/source: {bootstrap_summary.get('num_points', 'n/a')} / {bootstrap_summary.get('source_points', 'n/a')}",
                    f"- Bootstrap selection: {bootstrap_summary.get('selection_mode', 'n/a')}",
                ]
            )
        if render_sanity_summary:
            lines.extend(
                [
                    f"- Render mean unique colors: {render_sanity_summary.get('mean_unique_colors', 'n/a')}",
                    f"- Render constant-frame fraction: {render_sanity_summary.get('constant_frame_fraction', 'n/a')}",
                    f"- Render degenerate constant: {render_sanity_summary.get('degenerate_constant_render', 'n/a')}",
                ]
            )
        if entitybank_summary:
            lines.extend(
                [
                    f"- Entitybank clusters/entities: {entitybank_summary.get('num_clusters', 'n/a')} / {entitybank_summary.get('num_entities', 'n/a')}",
                    f"- Entitybank motion mean/max: {entitybank_summary.get('motion_score_mean', 'n/a')} / {entitybank_summary.get('motion_score_max', 'n/a')}",
                    f"- Entitybank speed mean/max: {entitybank_summary.get('speed_mean', 'n/a')} / {entitybank_summary.get('speed_max', 'n/a')}",
                    f"- Entitybank acceleration mean/max: {entitybank_summary.get('acceleration_mean', 'n/a')} / {entitybank_summary.get('acceleration_max', 'n/a')}",
                    f"- Entitybank occupancy mean/max: {entitybank_summary.get('occupancy_mean', 'n/a')} / {entitybank_summary.get('occupancy_max', 'n/a')}",
                    f"- Entitybank visibility mean/max: {entitybank_summary.get('visibility_mean', 'n/a')} / {entitybank_summary.get('visibility_max', 'n/a')}",
                ]
            )
        if semantic_summary:
            lines.extend(
                [
                    f"- Semantic slots/tracks: {semantic_summary.get('num_slots', 'n/a')} / {semantic_summary.get('num_tracks', 'n/a')}",
                    f"- Semantic priors static/dynamic/interaction: {semantic_summary.get('num_static_heads', 'n/a')} / {semantic_summary.get('num_dynamic_heads', 'n/a')} / {semantic_summary.get('num_interaction_heads', 'n/a')}",
                    f"- Segmentation bootstrap images: {semantic_summary.get('num_bootstrap_images', 'n/a')}",
                    f"- Active slots mean/max: {semantic_summary.get('active_slots_mean', 'n/a')} / {semantic_summary.get('active_slots_max', 'n/a')}",
                    f"- Moving slots mean/max: {semantic_summary.get('moving_slots_mean', 'n/a')} / {semantic_summary.get('moving_slots_max', 'n/a')}",
                    f"- Dynamic slots mean/max: {semantic_summary.get('dynamic_slots_mean', 'n/a')} / {semantic_summary.get('dynamic_slots_max', 'n/a')}",
                    f"- Static slots mean/max: {semantic_summary.get('static_slots_mean', 'n/a')} / {semantic_summary.get('static_slots_max', 'n/a')}",
                ]
            )
        if native_semantic_summary:
            lines.extend(
                [
                    f"- Native semantic assignments: {native_semantic_summary.get('num_assignments', 'n/a')}",
                    f"- Native tool/patient/support/agent-like: {native_semantic_summary.get('num_tool_like', 'n/a')} / {native_semantic_summary.get('num_patient_like', 'n/a')} / {native_semantic_summary.get('num_support_like', 'n/a')} / {native_semantic_summary.get('num_agent_like', 'n/a')}",
                    f"- Native interaction pairs: {native_semantic_summary.get('num_interaction_pairs', 'n/a')}",
                    f"- Native queries: {native_semantic_summary.get('num_native_queries', 'n/a')}",
                ]
            )
        with open(run_dir / "summary.md", "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

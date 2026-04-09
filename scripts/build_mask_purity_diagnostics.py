import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("-", " ").replace("_", " ").split())


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _polygon_to_mask(size: tuple[int, int], segmentation: Any) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if not polygon:
                continue
            xy = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            if len(xy) >= 3:
                draw.polygon(xy, fill=255)
    return np.asarray(mask, dtype=np.uint8) > 0


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return _safe_div(inter, union)


def _represented_intervals(time_ids: list[int], total_frames: int) -> list[list[int]]:
    if not time_ids:
        return []
    sorted_ids = sorted(int(value) for value in time_ids)
    intervals: list[list[int]] = []
    for index, current in enumerate(sorted_ids):
        if index == 0:
            start = 0
        else:
            start = (sorted_ids[index - 1] + current) // 2 + 1
        if index == len(sorted_ids) - 1:
            end = int(total_frames - 1)
        else:
            end = (current + sorted_ids[index + 1]) // 2
        intervals.append([int(start), int(end)])
    return intervals


def _nearest_render_row(time_id: int, rendered_rows: list[dict[str, Any]], max_distance: int) -> dict[str, Any] | None:
    if not rendered_rows:
        return None
    best = min(rendered_rows, key=lambda row: abs(int(row["time_id"]) - int(time_id)))
    if abs(int(best["time_id"]) - int(time_id)) > int(max_distance):
        return None
    return best


def _build_gt_masks(annotation_dir: Path) -> tuple[dict[str, tuple[int, int]], dict[str, dict[str, np.ndarray]], list[str]]:
    coco_path = annotation_dir / "train" / "_annotations.coco.json"
    video_annotation_path = annotation_dir / "video_annotations.json"
    coco_payload = _read_json(coco_path)
    video_annotations = _read_json(video_annotation_path)
    top_level_objects = [_norm(key) for key in video_annotations.keys()]
    category_name_by_id = {int(item["id"]): _norm(item["name"]) for item in coco_payload.get("categories", [])}
    image_meta = {
        int(item["id"]): (str(item["file_name"]), (int(item["width"]), int(item["height"])))
        for item in coco_payload.get("images", [])
    }
    masks_by_object: dict[str, dict[str, np.ndarray]] = {name: {} for name in top_level_objects}
    name_to_category_ids: dict[str, list[int]] = {}
    for category_id, category_name in category_name_by_id.items():
        name_to_category_ids.setdefault(category_name, []).append(category_id)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_payload.get("annotations", []):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    for image_id, (file_name, size) in image_meta.items():
        image_key = file_name.split("_")[0]
        ann_list = annotations_by_image.get(image_id, [])
        for object_name in top_level_objects:
            category_ids = set(name_to_category_ids.get(object_name, []))
            if not category_ids:
                continue
            merged_mask = np.zeros((int(size[1]), int(size[0])), dtype=bool)
            for ann in ann_list:
                if int(ann["category_id"]) not in category_ids:
                    continue
                merged_mask |= _polygon_to_mask(size, ann.get("segmentation", []))
            if merged_mask.any():
                masks_by_object[object_name][image_key] = merged_mask
    return image_meta, masks_by_object, top_level_objects


def _object_for_query(query_text: str, top_level_objects: list[str]) -> str:
    query_norm = _norm(query_text)
    matches = [name for name in top_level_objects if name in query_norm]
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]
    if len(top_level_objects) == 1:
        return top_level_objects[0]
    raise ValueError(f"Unable to infer target object for query: {query_text}")


def _find_source_frame_dir(dataset_dir: Path) -> Path:
    rgb_root = dataset_dir / "rgb"
    for candidate in ("2x", "1x", "4x"):
        path = rgb_root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No rgb scale directory under {rgb_root}")


def _binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def _load_track_masks(track_path: Path) -> dict[str, np.ndarray]:
    if not track_path.exists():
        return {}
    payload = _read_json(track_path)
    masks_by_image: dict[str, np.ndarray] = {}
    for track in payload.get("tracks", []):
        for frame in track.get("frames", []):
            if not bool(frame.get("active")) or not frame.get("mask_path"):
                continue
            image_id = str(frame.get("image_id", ""))
            if not image_id:
                continue
            mask = _binary_mask(Path(frame["mask_path"]))
            if image_id in masks_by_image:
                masks_by_image[image_id] |= mask
            else:
                masks_by_image[image_id] = mask
    return masks_by_image


def _overlay_mask(source: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = source.astype(np.float32).copy()
    mask_f = np.asarray(mask, dtype=bool)
    if not mask_f.any():
        return source
    color_arr = np.asarray(color, dtype=np.float32)
    result[mask_f] = result[mask_f] * (1.0 - alpha) + color_arr * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _error_overlay(source: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    result = source.astype(np.uint8).copy()
    tp = np.logical_and(pred_mask, gt_mask)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask))
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask)
    result = _overlay_mask(result, tp, (60, 220, 120), 0.55)
    result = _overlay_mask(result, fp, (255, 80, 80), 0.60)
    result = _overlay_mask(result, fn, (80, 140, 255), 0.60)
    return result


def _add_caption(image: Image.Image, lines: list[str]) -> Image.Image:
    font = ImageFont.load_default()
    pad = 8
    line_h = 14
    caption_h = pad * 2 + line_h * len(lines)
    canvas = Image.new("RGB", (image.width, image.height + caption_h), (24, 24, 24))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + pad
    for line in lines:
        draw.text((pad, y), line, fill=(240, 240, 240), font=font)
        y += line_h
    return canvas


def _assemble_row(images: list[Image.Image], pad: int = 8, bg: tuple[int, int, int] = (12, 12, 12)) -> Image.Image:
    width = sum(image.width for image in images) + pad * (len(images) + 1)
    height = max(image.height for image in images) + pad * 2
    canvas = Image.new("RGB", (width, height), bg)
    x = pad
    for image in images:
        canvas.paste(image, (x, pad))
        x += image.width + pad
    return canvas


def _resize(image: Image.Image, width: int = 320) -> Image.Image:
    scale = float(width) / float(image.width)
    return image.resize((width, max(1, int(round(image.height * scale)))), Image.BILINEAR)


def _write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Mask Purity Diagnostics",
        "",
        "| Query | Frames | Final IoU | GSAM2 IoU | Purity | Recall | Pred Area / GT Area |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['query_slug']}` | {row['frame_count']} | {row['final_mean_iou']*100:.2f} | "
            f"{row['gsam2_mean_iou']*100:.2f} | {row['final_mean_purity']*100:.2f} | "
            f"{row['final_mean_recall']*100:.2f} | {row['pred_to_gt_area_ratio']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-query", type=int, default=6)
    args = parser.parse_args()

    protocol_payload = _read_json(Path(args.protocol_json))
    metadata_payload = _read_json(Path(args.dataset_dir) / "metadata.json")
    _, gt_masks_by_object, top_level_objects = _build_gt_masks(Path(args.annotation_dir))
    source_dir = _find_source_frame_dir(Path(args.dataset_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for query_item in protocol_payload.get("queries", []):
        query_slug = str(query_item["query_slug"])
        query_text = str(query_item["query"])
        query_dir = Path(args.query_root) / query_slug
        validation_path = query_dir / "final_query_render_sourcebg" / "validation.json"
        track_path = query_dir / "grounded_sam2" / "grounded_sam2_query_tracks.json"
        if not validation_path.exists():
            continue
        validation_payload = _read_json(validation_path)
        gt_object = _object_for_query(query_text, top_level_objects)
        gt_masks = gt_masks_by_object.get(gt_object, {})
        gsam2_masks = _load_track_masks(track_path)
        binary_mask_dir = Path(validation_payload["frame_exports"]["binary_masks"])

        raw_rows = []
        for row in validation_payload.get("frames", []):
            image_id = str(row["image_id"])
            if image_id not in metadata_payload:
                continue
            raw_rows.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "image_id": image_id,
                    "time_id": int(metadata_payload[image_id]["time_id"]),
                    "query_active": bool(row["query_active"]),
                }
            )
        raw_rows.sort(key=lambda row: row["time_id"])
        total_frames = max(int(meta["time_id"]) for meta in metadata_payload.values()) + 1
        intervals = _represented_intervals([row["time_id"] for row in raw_rows], total_frames=total_frames)
        for row, interval in zip(raw_rows, intervals):
            row["represented_interval"] = interval
        rendered_time_ids = [int(row["time_id"]) for row in raw_rows]
        time_diffs = np.diff(np.asarray(rendered_time_ids, dtype=np.int32)) if len(rendered_time_ids) >= 2 else np.asarray([], dtype=np.int32)
        max_distance = int(max(2, int(np.median(time_diffs)) // 2 + 1)) if time_diffs.size else 2
        gt_ranges = query_item["targets"][0]["target_ranges"]
        gt_frames = set()
        for start, end in gt_ranges:
            gt_frames.update(range(int(start), int(end) + 1))

        frame_rows = []
        for image_id, gt_mask in gt_masks.items():
            if image_id not in metadata_payload:
                continue
            time_id = int(metadata_payload[image_id]["time_id"])
            render_row = _nearest_render_row(time_id, raw_rows, max_distance=max_distance)
            if render_row is None:
                continue
            pred_mask_path = binary_mask_dir / f"{int(render_row['frame_index']):05d}.png"
            if not pred_mask_path.exists():
                continue
            pred_mask = _binary_mask(pred_mask_path)
            gsam2_mask = gsam2_masks.get(str(render_row["image_id"]), np.zeros_like(gt_mask))
            if gsam2_mask.shape != gt_mask.shape:
                gsam2_mask = np.zeros_like(gt_mask)
            final_iou = _mask_iou(pred_mask, gt_mask)
            gsam2_iou = _mask_iou(gsam2_mask, gt_mask)
            tp = float(np.logical_and(pred_mask, gt_mask).sum())
            pred_area = float(pred_mask.sum())
            gt_area = float(gt_mask.sum())
            frame_rows.append(
                {
                    "frame_index": int(render_row["frame_index"]),
                    "image_id": str(render_row["image_id"]),
                    "query_active": bool(render_row["query_active"]),
                    "gt_active": bool(time_id in gt_frames),
                    "pred_mask": pred_mask,
                    "gt_mask": gt_mask,
                    "gsam2_mask": gsam2_mask,
                    "final_iou": final_iou,
                    "gsam2_iou": gsam2_iou,
                    "purity": _safe_div(tp, pred_area),
                    "recall": _safe_div(tp, gt_area),
                    "pred_area": pred_area,
                    "gt_area": gt_area,
                }
            )

        active_rows = [row for row in frame_rows if row["query_active"] or row["gt_active"]]
        if not active_rows:
            continue
        active_rows.sort(key=lambda row: (row["final_iou"], row["frame_index"]))
        chosen = active_rows[: max(int(args.frames_per_query), 1)]

        query_output_dir = output_dir / query_slug
        query_output_dir.mkdir(parents=True, exist_ok=True)
        row_images: list[Image.Image] = []
        final_ious = []
        gsam2_ious = []
        purities = []
        recalls = []
        pred_areas = []
        gt_areas = []

        for index, row in enumerate(chosen):
            image_path = source_dir / f"{row['image_id']}.png"
            with Image.open(image_path) as image:
                source = np.asarray(image.convert("RGB"), dtype=np.uint8)
            gt_mask = row["gt_mask"]
            pred_mask = row["pred_mask"]
            gsam2_mask = row["gsam2_mask"]

            source_img = _resize(Image.fromarray(source))
            gsam2_img = _resize(Image.fromarray(_overlay_mask(source, gsam2_mask, (64, 224, 255), 0.55)))
            pred_img = _resize(Image.fromarray(_overlay_mask(source, pred_mask, (255, 64, 196), 0.55)))
            gt_img = _resize(Image.fromarray(_overlay_mask(source, gt_mask, (255, 210, 64), 0.55)))
            err_img = _resize(Image.fromarray(_error_overlay(source, pred_mask, gt_mask)))

            title_lines = [
                f"{row['image_id']} / frame {row['frame_index']}",
                f"final IoU {row['final_iou']*100:.1f} | GSAM2 {row['gsam2_iou']*100:.1f}",
                f"purity {row['purity']*100:.1f} | recall {row['recall']*100:.1f}",
            ]
            labeled = _assemble_row(
                [
                    _add_caption(source_img, ["source"]),
                    _add_caption(gsam2_img, ["gsam2 mask"]),
                    _add_caption(pred_img, ["final mask"]),
                    _add_caption(gt_img, ["gt mask"]),
                    _add_caption(err_img, ["error: green tp / red fp / blue fn"]),
                ]
            )
            labeled = _add_caption(labeled, title_lines)
            labeled.save(query_output_dir / f"mask_purity_{index:02d}_{row['image_id']}.png")
            row_images.append(labeled)

            final_ious.append(row["final_iou"])
            gsam2_ious.append(row["gsam2_iou"])
            purities.append(row["purity"])
            recalls.append(row["recall"])
            pred_areas.append(row["pred_area"])
            gt_areas.append(row["gt_area"])

        if row_images:
            page = _assemble_row(row_images[:3], pad=12)
            if len(row_images) > 3:
                second = _assemble_row(row_images[3:], pad=12)
                stacked = Image.new("RGB", (max(page.width, second.width), page.height + second.height + 12), (8, 8, 8))
                stacked.paste(page, (0, 0))
                stacked.paste(second, (0, page.height + 12))
                page = stacked
            page.save(query_output_dir / "mask_purity_overview.png")

        summary = {
            "query_slug": query_slug,
            "query": query_text,
            "target_object": gt_object,
            "frame_count": len(active_rows),
            "final_mean_iou": float(np.mean(final_ious)) if final_ious else 0.0,
            "gsam2_mean_iou": float(np.mean(gsam2_ious)) if gsam2_ious else 0.0,
            "final_mean_purity": float(np.mean(purities)) if purities else 0.0,
            "final_mean_recall": float(np.mean(recalls)) if recalls else 0.0,
            "pred_to_gt_area_ratio": _safe_div(float(np.mean(pred_areas)) if pred_areas else 0.0, float(np.mean(gt_areas)) if gt_areas else 0.0),
            "output_dir": str(query_output_dir),
        }
        with open(query_output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        summary_rows.append(summary)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"queries": summary_rows}, handle, indent=2, ensure_ascii=False)
    _write_summary_md(output_dir / "summary.md", summary_rows)


if __name__ == "__main__":
    main()

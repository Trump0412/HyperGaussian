import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("-", " ").replace("_", " ").split())


def _open_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def _label(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, canvas.width, 28), fill=(18, 18, 18, 220))
    draw.text((8, 6), text, fill=(240, 240, 240, 255))
    return canvas


def _grid(images: list[Image.Image], columns: int = 3, bg=(245, 245, 245)) -> Image.Image:
    if not images:
        return Image.new("RGB", (32, 32), bg)
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    cols = max(1, int(columns))
    rows = int(np.ceil(len(images) / cols))
    canvas = Image.new("RGB", (cols * width, rows * height), bg)
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        paste = image
        if image.size != (width, height):
            paste = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas.paste(paste, (col * width, row * height))
    return canvas


def _annotation_lookup(annotation_dir: Path) -> tuple[dict[int, dict], dict[str, list[int]], list[str]]:
    coco = _read_json(annotation_dir / "train" / "_annotations.coco.json")
    video_annotations = _read_json(annotation_dir / "video_annotations.json")
    top_level_objects = [_norm(key) for key in video_annotations.keys()]
    image_rows = {int(row["id"]): row for row in coco["images"]}
    name_to_category_ids: dict[str, list[int]] = {}
    for item in coco["categories"]:
        name_to_category_ids.setdefault(_norm(item["name"]), []).append(int(item["id"]))
    return image_rows, name_to_category_ids, top_level_objects


def _target_object(query_text: str, top_level_objects: list[str]) -> str | None:
    query_norm = _norm(query_text)
    matches = [name for name in top_level_objects if name in query_norm]
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]
    if len(top_level_objects) == 1:
        return top_level_objects[0]
    return None


def _gt_mask_image(annotation_dir: Path, image_id: str, target_object: str | None) -> Image.Image | None:
    image_rows, name_to_category_ids, _ = _annotation_lookup(annotation_dir)
    coco = _read_json(annotation_dir / "train" / "_annotations.coco.json")
    if not target_object:
        return None
    category_ids = set(name_to_category_ids.get(_norm(target_object), []))
    if not category_ids:
        return None
    image_row = next((row for row in image_rows.values() if str(row["file_name"]).startswith(f"{image_id}_")), None)
    if image_row is None:
        return None
    width = int(image_row["width"])
    height = int(image_row["height"])
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for ann in coco["annotations"]:
        if int(ann["image_id"]) != int(image_row["id"]):
            continue
        if int(ann["category_id"]) not in category_ids:
            continue
        for poly in ann.get("segmentation", []):
            if len(poly) < 6:
                continue
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(xy, fill=255)
    mask = np.asarray(canvas, dtype=np.uint8)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask > 0] = np.asarray([255, 196, 0], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def export_diagnostics(query_root: Path, dataset_dir: Path, annotation_dir: Path | None, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _read_json(query_root / "query_plan.json")
    gsam = _read_json(query_root / "grounded_sam2" / "grounded_sam2_query_tracks.json")
    selection = _read_json(query_root / "query_worldtube_run" / "entitybank" / "selected_query_qwen.json")
    validation = _read_json(query_root / "final_query_render_sourcebg" / "validation.json")
    target_object = None
    if annotation_dir is not None:
        _, _, top_level_objects = _annotation_lookup(annotation_dir)
        target_object = _target_object(str(selection.get("query", "")), top_level_objects)

    context_images = []
    for frame in plan.get("context_frames", []):
        image = _open_rgb(Path(frame["image_path"]))
        context_images.append(_label(image, f"context {frame['image_id']}"))
    _grid(context_images, columns=min(5, max(1, len(context_images)))).save(output_dir / "01_qwen_context_grid.png")

    gsam_images = []
    for phrase in gsam.get("phrases", []):
        anchors = phrase.get("anchors", [])
        for anchor in anchors[:3]:
            anchor_mask = Path(anchor["anchor_mask_preview_path"])
            if anchor_mask.exists():
                gsam_images.append(_label(_open_rgb(anchor_mask), f"{phrase['phrase']} anchor {anchor['anchor_image_id']}"))
        overlay_dir = Path(phrase["track_overlay_dir"])
        overlay_paths = sorted(overlay_dir.glob("*.png"))[:4]
        for overlay_path in overlay_paths:
            gsam_images.append(_label(_open_rgb(overlay_path), f"{phrase['phrase']} track {overlay_path.stem}"))
    _grid(gsam_images, columns=3).save(output_dir / "02_gsam2_tracking_grid.png")

    final_overlay_dir = Path(validation["frame_exports"]["overlay_frames"])
    final_mask_dir = Path(validation["frame_exports"]["binary_masks"])
    chosen_rows = [row for row in validation.get("frames", []) if row.get("query_active")]
    chosen_rows = chosen_rows[:6]
    final_images = []
    for row in chosen_rows:
        overlay = _open_rgb(final_overlay_dir / f"{int(row['frame_index']):05d}.png")
        final_images.append(_label(overlay, f"final {row['image_id']}"))
        mask = _open_rgb(final_mask_dir / f"{int(row['frame_index']):05d}.png")
        final_images.append(_label(mask, f"mask {row['image_id']}"))
        if annotation_dir is not None:
            gt = _gt_mask_image(annotation_dir, str(row["image_id"]), target_object=target_object)
            if gt is not None:
                final_images.append(_label(gt, f"gt {row['image_id']}"))
    _grid(final_images, columns=3).save(output_dir / "03_final_vs_gt_grid.png")

    summary = {
        "query": selection.get("query"),
        "selected": selection.get("selected", []),
        "notes": selection.get("notes", ""),
        "target_object": target_object,
        "validation_path": str(query_root / "final_query_render_sourcebg" / "validation.json"),
        "artifacts": {
            "context_grid": str(output_dir / "01_qwen_context_grid.png"),
            "gsam2_grid": str(output_dir / "02_gsam2_tracking_grid.png"),
            "final_vs_gt_grid": str(output_dir / "03_final_vs_gt_grid.png"),
        },
    }
    with open(output_dir / "diagnostic_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-root", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--annotation-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_diagnostics(
        query_root=Path(args.query_root),
        dataset_dir=Path(args.dataset_dir),
        annotation_dir=None if args.annotation_dir is None else Path(args.annotation_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _sorted_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"))


def _to_u8_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return np.ones((size, size), dtype=np.uint8)


def _estimate_bg_color(image_rgb: np.ndarray, border: int = 6) -> np.ndarray:
    h, w, _ = image_rgb.shape
    b = int(np.clip(border, 1, min(h, w) // 2))
    strips = [
        image_rgb[:b, :, :].reshape(-1, 3),
        image_rgb[-b:, :, :].reshape(-1, 3),
        image_rgb[:, :b, :].reshape(-1, 3),
        image_rgb[:, -b:, :].reshape(-1, 3),
    ]
    pixels = np.concatenate(strips, axis=0)
    return np.median(pixels, axis=0).astype(np.float32)


def _extract_mask_from_entity_render(
    entity_rgb: np.ndarray,
    bg_tol: int,
    close_size: int,
    open_size: int,
) -> np.ndarray:
    bg = _estimate_bg_color(entity_rgb)
    delta = np.max(np.abs(entity_rgb.astype(np.float32) - bg[None, None, :]), axis=2)
    mask_u8 = _to_u8_mask(delta > float(bg_tol))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, _kernel(close_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, _kernel(open_size))
    return mask_u8 > 0


def _resolve_scene_mode(mode: str, bundle_summary: dict) -> str:
    if mode != "auto":
        return mode
    query = str(bundle_summary.get("query", "")).strip().lower()
    if "lemon" in query:
        return "lemon"
    if "cookie" in query:
        return "cookie"
    if "cup" in query or "glass" in query:
        return "cup"
    return "generic"


def _postprocess_one_frame(
    full_rgb: np.ndarray,
    entity_rgb: np.ndarray,
    removed_rgb: np.ndarray,
    scene_mode: str,
    bg_tol: int,
    dark_delta: int,
    core_erode: int,
    ring_dilate: int,
    inpaint_dilate: int,
    inpaint_radius: int,
    lemon_sat_delta: int,
) -> tuple[np.ndarray, dict]:
    loose_mask = _extract_mask_from_entity_render(
        entity_rgb=entity_rgb,
        bg_tol=bg_tol,
        close_size=5,
        open_size=3,
    )
    loose_u8 = _to_u8_mask(loose_mask)
    if int(loose_u8.sum()) == 0:
        return removed_rgb, {
            "loose_px": 0,
            "core_px": 0,
            "restore_px": 0,
            "inpaint_px": 0,
        }

    core_u8 = cv2.erode(loose_u8, _kernel(core_erode), iterations=1)
    if int(core_u8.sum()) == 0:
        core_u8 = cv2.erode(loose_u8, _kernel(max(3, core_erode // 2)), iterations=1)
    if int(core_u8.sum()) == 0:
        core_u8 = loose_u8.copy()
    core_mask = core_u8 > 0

    if scene_mode == "lemon":
        full_hsv = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2HSV)
        yellow = (
            (full_hsv[:, :, 0] >= 14)
            & (full_hsv[:, :, 0] <= 48)
            & (full_hsv[:, :, 1] >= 70)
            & (full_hsv[:, :, 2] >= 60)
            & loose_mask
        )
        yellow_u8 = cv2.morphologyEx(_to_u8_mask(yellow), cv2.MORPH_OPEN, _kernel(3))
        yellow_u8 = cv2.morphologyEx(yellow_u8, cv2.MORPH_CLOSE, _kernel(5))
        core_mask = np.logical_or(core_mask, yellow_u8 > 0)
        core_u8 = _to_u8_mask(core_mask)

    ring_u8 = cv2.dilate(loose_u8, _kernel(ring_dilate), iterations=1)
    ring_mask = ring_u8 > 0

    full_gray = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    removed_gray = cv2.cvtColor(removed_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
    dark_mask = (full_gray - removed_gray) > int(dark_delta)
    restore_mask = np.logical_and(np.logical_and(dark_mask, ring_mask), np.logical_not(core_mask))

    if scene_mode == "lemon":
        full_hsv = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2HSV)
        removed_hsv = cv2.cvtColor(removed_rgb, cv2.COLOR_RGB2HSV)
        sat_drop = (full_hsv[:, :, 1].astype(np.int16) - removed_hsv[:, :, 1].astype(np.int16)) > int(lemon_sat_delta)
        restore_mask = np.logical_or(
            restore_mask,
            np.logical_and(np.logical_and(sat_drop, ring_mask), np.logical_not(core_mask)),
        )

    restored = removed_rgb.copy()
    restored[restore_mask] = full_rgb[restore_mask]

    inpaint_u8 = cv2.dilate(core_u8, _kernel(inpaint_dilate), iterations=1)
    if int(inpaint_u8.sum()) > 0:
        processed = cv2.inpaint(
            restored,
            inpaint_u8,
            inpaintRadius=max(1, int(inpaint_radius)),
            flags=cv2.INPAINT_TELEA,
        )
    else:
        processed = restored

    return processed, {
        "loose_px": int(loose_mask.sum()),
        "core_px": int(core_mask.sum()),
        "restore_px": int(restore_mask.sum()),
        "inpaint_px": int((inpaint_u8 > 0).sum()),
    }


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _build_triptych_video(
    full_paths: list[Path],
    entity_paths: list[Path],
    removed_post_paths: list[Path],
    output_path: Path,
    fps: int,
) -> None:
    if not full_paths:
        return
    label_font = _load_font(24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=8)
    try:
        for full_path, entity_path, post_path in zip(full_paths, entity_paths, removed_post_paths):
            full = Image.open(full_path).convert("RGB")
            entity = Image.open(entity_path).convert("RGB")
            post = Image.open(post_path).convert("RGB")
            w, h = full.size
            pad = 12
            top = 44
            canvas = Image.new("RGB", (w * 3 + pad * 4, h + top + pad * 2), (245, 245, 245))
            draw = ImageDraw.Draw(canvas)
            labels = ("Full Scene", "Target Entity", "Scene Without Target (Post2D)")
            for col, text in enumerate(labels):
                x = pad + col * (w + pad)
                draw.text((x + 8, 10), text, fill=(20, 20, 20), font=label_font)
            y = top
            canvas.paste(full, (pad, y))
            canvas.paste(entity, (pad * 2 + w, y))
            canvas.paste(post, (pad * 3 + w * 2, y))
            writer.append_data(np.asarray(canvas, dtype=np.uint8))
            full.close()
            entity.close()
            post.close()
    finally:
        writer.close()


def run_postprocess(
    bundle_summary_path: Path,
    output_dir: Path | None,
    scene_mode: str,
    bg_tol: int,
    dark_delta: int,
    core_erode: int,
    ring_dilate: int,
    inpaint_dilate: int,
    inpaint_radius: int,
    lemon_sat_delta: int,
    fps: int,
) -> Path:
    bundle_summary = _read_json(bundle_summary_path)
    artifacts = bundle_summary.get("artifacts", {})
    full_dir = Path(str(artifacts.get("full_render_dir", "")))
    entity_dir = Path(str(artifacts.get("complete_cookie_render_dir", "")))
    removed_dir = Path(str(artifacts.get("scene_without_cookie_render_dir", "")))
    if not (full_dir.exists() and entity_dir.exists() and removed_dir.exists()):
        raise FileNotFoundError("Render directories are missing in bundle summary artifacts.")

    full_paths = _sorted_pngs(full_dir)
    entity_paths = _sorted_pngs(entity_dir)
    removed_paths = _sorted_pngs(removed_dir)
    frame_count = min(len(full_paths), len(entity_paths), len(removed_paths))
    if frame_count <= 0:
        raise ValueError("No render frames found for postprocess.")
    full_paths = full_paths[:frame_count]
    entity_paths = entity_paths[:frame_count]
    removed_paths = removed_paths[:frame_count]

    resolved_mode = _resolve_scene_mode(scene_mode, bundle_summary)
    output_root = output_dir if output_dir is not None else bundle_summary_path.parent / "visuals_post2d"
    frames_out = output_root / "scene_without_target_post2d_frames"
    frames_out.mkdir(parents=True, exist_ok=True)

    rows = []
    removed_post_paths: list[Path] = []
    for idx, (full_path, entity_path, removed_path) in enumerate(zip(full_paths, entity_paths, removed_paths)):
        full_rgb = np.asarray(Image.open(full_path).convert("RGB"), dtype=np.uint8)
        entity_rgb = np.asarray(Image.open(entity_path).convert("RGB"), dtype=np.uint8)
        removed_rgb = np.asarray(Image.open(removed_path).convert("RGB"), dtype=np.uint8)
        post_rgb, row = _postprocess_one_frame(
            full_rgb=full_rgb,
            entity_rgb=entity_rgb,
            removed_rgb=removed_rgb,
            scene_mode=resolved_mode,
            bg_tol=int(bg_tol),
            dark_delta=int(dark_delta),
            core_erode=int(core_erode),
            ring_dilate=int(ring_dilate),
            inpaint_dilate=int(inpaint_dilate),
            inpaint_radius=int(inpaint_radius),
            lemon_sat_delta=int(lemon_sat_delta),
        )
        row["frame_index"] = int(idx)
        rows.append(row)
        out_path = frames_out / removed_path.name
        Image.fromarray(post_rgb).save(out_path)
        removed_post_paths.append(out_path)

    post_video = output_root / "scene_without_target_post2d.mp4"
    writer = imageio.get_writer(str(post_video), fps=int(fps), codec="libx264", quality=8)
    try:
        for path in removed_post_paths:
            writer.append_data(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    finally:
        writer.close()

    triptych_video = output_root / "render_triptych_all_frames_post2d.mp4"
    _build_triptych_video(
        full_paths=full_paths,
        entity_paths=entity_paths,
        removed_post_paths=removed_post_paths,
        output_path=triptych_video,
        fps=int(fps),
    )

    mean_restore = float(np.mean([row["restore_px"] for row in rows])) if rows else 0.0
    mean_inpaint = float(np.mean([row["inpaint_px"] for row in rows])) if rows else 0.0
    summary = {
        "schema_version": 1,
        "bundle_summary_path": str(bundle_summary_path),
        "scene_mode": resolved_mode,
        "frame_count": int(frame_count),
        "params": {
            "bg_tol": int(bg_tol),
            "dark_delta": int(dark_delta),
            "core_erode": int(core_erode),
            "ring_dilate": int(ring_dilate),
            "inpaint_dilate": int(inpaint_dilate),
            "inpaint_radius": int(inpaint_radius),
            "lemon_sat_delta": int(lemon_sat_delta),
            "fps": int(fps),
        },
        "mean_restore_px": mean_restore,
        "mean_inpaint_px": mean_inpaint,
        "artifacts": {
            "frames_dir": str(frames_out),
            "post_video": str(post_video),
            "triptych_post2d_video": str(triptych_video),
        },
        "frame_rows": rows,
    }
    _write_json(output_root / "postprocess_summary.json", summary)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-summary", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-mode", choices=["auto", "cookie", "lemon", "cup", "generic"], default="auto")
    parser.add_argument("--bg-tol", type=int, default=18)
    parser.add_argument("--dark-delta", type=int, default=28)
    parser.add_argument("--core-erode", type=int, default=11)
    parser.add_argument("--ring-dilate", type=int, default=21)
    parser.add_argument("--inpaint-dilate", type=int, default=3)
    parser.add_argument("--inpaint-radius", type=int, default=3)
    parser.add_argument("--lemon-sat-delta", type=int, default=40)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    out = run_postprocess(
        bundle_summary_path=Path(args.bundle_summary),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
        scene_mode=str(args.scene_mode),
        bg_tol=int(args.bg_tol),
        dark_delta=int(args.dark_delta),
        core_erode=int(args.core_erode),
        ring_dilate=int(args.ring_dilate),
        inpaint_dilate=int(args.inpaint_dilate),
        inpaint_radius=int(args.inpaint_radius),
        lemon_sat_delta=int(args.lemon_sat_delta),
        fps=int(args.fps),
    )
    print(out)


if __name__ == "__main__":
    main()

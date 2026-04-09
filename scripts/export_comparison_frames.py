import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def parse_entry(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid entry: {raw}. Expected label=/abs/path")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def find_render_dir(path: Path, split: str) -> Path:
    if path.is_dir() and any(path.glob("*.png")):
        return path
    split_dir = path / split
    if split_dir.is_dir():
        candidates = sorted(split_dir.glob("ours_*"))
        if not candidates:
            raise FileNotFoundError(f"No ours_* directory found under {split_dir}")
        latest = candidates[-1]
        render_dir = latest / "renders"
        if render_dir.is_dir():
            return render_dir
    raise FileNotFoundError(f"Unable to resolve render dir from {path}")


def list_frames(render_dir: Path) -> list[Path]:
    frames = sorted(render_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {render_dir}")
    return frames


def intersection_names(frame_lists: Iterable[list[Path]]) -> list[str]:
    common = None
    for frames in frame_lists:
        names = {frame.name for frame in frames}
        common = names if common is None else common & names
    if not common:
        raise ValueError("No common frame names across all entries")
    return sorted(common)


def fit_label(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    for end in range(len(text), 0, -1):
        candidate = text[:end] + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            return candidate
    return suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--frame-name")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--title", default="")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--padding", type=int, default=20)
    args = parser.parse_args()

    entries = [parse_entry(raw) for raw in args.entry]
    resolved = [(label, find_render_dir(path, args.split)) for label, path in entries]
    frame_lists = [list_frames(render_dir) for _label, render_dir in resolved]
    common_names = intersection_names(frame_lists)
    if args.frame_name:
        frame_name = args.frame_name
        if frame_name not in common_names:
            raise FileNotFoundError(f"{frame_name} is not shared across all entries")
    else:
        index = max(0, min(args.frame_index, len(common_names) - 1))
        frame_name = common_names[index]

    opened = []
    for label, render_dir in resolved:
        image = Image.open(render_dir / frame_name).convert("RGB")
        opened.append((label, image))

    sample_w, sample_h = opened[0][1].size
    font = ImageFont.load_default()
    columns = max(1, args.columns)
    rows = (len(opened) + columns - 1) // columns
    label_h = 26
    title_h = 30 if args.title else 0
    canvas_w = columns * sample_w + (columns + 1) * args.padding
    canvas_h = title_h + rows * (sample_h + label_h) + (rows + 1) * args.padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    if args.title:
        draw.text((args.padding, 8), args.title, fill=(235, 235, 235), font=font)

    offset_y = title_h + args.padding
    for idx, (label, image) in enumerate(opened):
        row = idx // columns
        col = idx % columns
        x = args.padding + col * (sample_w + args.padding)
        y = offset_y + row * (sample_h + label_h + args.padding)
        canvas.paste(image, (x, y))
        text = fit_label(draw, label, sample_w, font)
        draw.rectangle((x, y + sample_h, x + sample_w, y + sample_h + label_h), fill=(30, 30, 30))
        draw.text((x + 8, y + sample_h + 6), text, fill=(240, 240, 240), font=font)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()

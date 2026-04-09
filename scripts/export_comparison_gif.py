import argparse
from pathlib import Path

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
        render_dir = candidates[-1] / "renders"
        if render_dir.is_dir():
            return render_dir
    raise FileNotFoundError(f"Unable to resolve render dir from {path}")


def fit_label(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] <= max_width:
        return text
    suffix = "..."
    for end in range(len(text), 0, -1):
        candidate = text[:end] + suffix
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] <= max_width:
            return candidate
    return suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    entries = [parse_entry(raw) for raw in args.entry]
    resolved = []
    name_sets = []
    for label, path in entries:
        render_dir = find_render_dir(path, args.split)
        names = sorted(frame.name for frame in render_dir.glob("*.png"))
        if not names:
            raise FileNotFoundError(f"No PNG frames found in {render_dir}")
        resolved.append((label, render_dir))
        name_sets.append(set(names))

    common_names = sorted(set.intersection(*name_sets))
    if not common_names:
        raise ValueError("No shared frames across entries")
    common_names = common_names[:: max(1, args.frame_step)][: max(1, args.max_frames)]

    font = ImageFont.load_default()
    first_image = Image.open(resolved[0][1] / common_names[0]).convert("RGB")
    sample_w, sample_h = first_image.size
    label_h = 24
    title_h = 24 if args.title else 0
    padding = 14
    width = len(resolved) * sample_w + (len(resolved) + 1) * padding
    height = sample_h + label_h + title_h + 2 * padding
    frames = []

    for frame_name in common_names:
        canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        if args.title:
            draw.text((padding, 6), f"{args.title}  {frame_name}", fill=(235, 235, 235), font=font)
        y = padding + title_h
        for idx, (label, render_dir) in enumerate(resolved):
            x = padding + idx * (sample_w + padding)
            image = Image.open(render_dir / frame_name).convert("RGB")
            canvas.paste(image, (x, y))
            draw.rectangle((x, y + sample_h, x + sample_w, y + sample_h + label_h), fill=(30, 30, 30))
            draw.text((x + 6, y + sample_h + 5), fit_label(draw, label, sample_w - 12, font), fill=(240, 240, 240), font=font)
        frames.append(canvas)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
    )
    print(output_path)


if __name__ == "__main__":
    main()

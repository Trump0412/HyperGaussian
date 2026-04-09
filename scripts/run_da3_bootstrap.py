import argparse
import json
from pathlib import Path

import torch


def collect_images(input_path: Path) -> list[str]:
    if input_path.is_file():
        return [str(input_path)]

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted(str(path) for path in input_path.iterdir() if path.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or directory of images")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--export-format",
        default="npz-gs_ply",
        help="DA3 export format string",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device",
    )
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = DepthAnything3.from_pretrained(args.model_id)
    model = model.to(device=device)
    model.eval()

    prediction = model.inference(
        images,
        export_dir=str(output_dir),
        export_format=args.export_format,
        align_to_input_ext_scale=True,
        infer_gs=True,
    )

    exported_files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "device": str(device),
        "input_path": str(input_path),
        "num_images": len(images),
        "export_format": args.export_format,
        "has_depth": getattr(prediction, "depth", None) is not None,
        "exported_files": exported_files,
    }
    with open(output_dir / "bootstrap_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()

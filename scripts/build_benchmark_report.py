import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_entry(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid entry: {raw}. Expected label=/abs/path/to/run")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def summarize_run(label: str, run_dir: Path) -> dict:
    metrics = read_json(run_dir / "metrics.json")
    entitybank = metrics.get("entitybank_summary") or {}
    semantic = metrics.get("semantic_summary") or {}
    return {
        "label": label,
        "run_dir": str(run_dir),
        "method": metrics.get("method"),
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS-vgg"),
        "train_seconds": metrics.get("train_seconds"),
        "render_fps": metrics.get("render_fps"),
        "num_entities": entitybank.get("num_entities"),
        "num_priors": semantic.get("num_priors"),
        "num_dynamic_heads": semantic.get("num_dynamic_heads"),
        "num_interaction_heads": semantic.get("num_interaction_heads"),
        "dynamic_slots_mean": semantic.get("dynamic_slots_mean"),
        "static_slots_mean": semantic.get("static_slots_mean"),
    }


def build_table(rows: list[dict]) -> str:
    header = (
        "| Method | PSNR | SSIM | LPIPS-vgg | Train s | FPS | "
        "Entities | Priors | Dynamic heads | Interaction heads | "
        "Dynamic slots mean | Static slots mean |"
    )
    sep = (
        "| --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {label} | {psnr} | {ssim} | {lpips} | {train} | {fps} | "
            "{entities} | {priors} | {dyn_heads} | {inter_heads} | {dyn_slots} | {stat_slots} |".format(
                label=row["label"],
                psnr=fmt(row["psnr"]),
                ssim=fmt(row["ssim"]),
                lpips=fmt(row["lpips"]),
                train=fmt(row["train_seconds"], digits=0),
                fps=fmt(row["render_fps"]),
                entities=fmt(row["num_entities"], digits=0),
                priors=fmt(row["num_priors"], digits=0),
                dyn_heads=fmt(row["num_dynamic_heads"], digits=0),
                inter_heads=fmt(row["num_interaction_heads"], digits=0),
                dyn_slots=fmt(row["dynamic_slots_mean"]),
                stat_slots=fmt(row["static_slots_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [summarize_run(*parse_entry(raw)) for raw in args.entry]
    lines = [f"# {args.title}", ""]
    if args.subtitle:
        lines.append(args.subtitle)
        lines.append("")
    lines.append(build_table(rows))
    lines.append("## Run Paths")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['label']}`: `{row['run_dir']}`")
    lines.append("")

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_entry(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid entry: {raw}. Expected label=/abs/path/to/query_dir")
    label, path = raw.split("=", 1)
    return label.strip(), path.strip()


def load_query_payload(path_spec: str) -> tuple[Path, dict, dict]:
    if "::" in path_spec:
        selected_raw, validation_raw = path_spec.split("::", 1)
        selected_path = Path(selected_raw)
        validation_path = Path(validation_raw)
        query_dir = selected_path.parent
        return query_dir, read_json(selected_path), read_json(validation_path)
    query_dir = Path(path_spec)
    selected = read_json(query_dir / "selected.json")
    validation_path = query_dir / "validation.json"
    if not validation_path.exists():
        validation_path = query_dir / "rendered" / "validation.json"
    validation = read_json(validation_path) if validation_path.exists() else {}
    return query_dir, selected, validation


def frame_set_from_segments(selected_payload: dict) -> set[int]:
    indices: set[int] = set()
    for item in selected_payload.get("selected", []):
        for segment in item.get("segments", []):
            if len(segment) != 2:
                continue
            start = int(segment[0])
            end = int(segment[1])
            for frame_idx in range(start, end + 1):
                indices.add(frame_idx)
    return indices


def frame_set_from_validation(validation_payload: dict) -> set[int]:
    indices: set[int] = set()
    for segment in validation_payload.get("active_segments", []):
        if len(segment) != 2:
            continue
        start = int(segment[0])
        end = int(segment[1])
        for frame_idx in range(start, end + 1):
            indices.add(frame_idx)
    return indices


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize_query_payload(query_dir: Path, selected: dict, validation: dict, reference_active: set[int], reference_roles: dict[str, int]) -> dict:
    if validation:
        active_frames = frame_set_from_validation(validation)
        first_active = validation.get("first_active_frame")
        last_active = validation.get("last_active_frame")
    else:
        active_frames = frame_set_from_segments(selected)
        first_active = min(active_frames) if active_frames else None
        last_active = max(active_frames) if active_frames else None

    intersection = reference_active & active_frames
    union = reference_active | active_frames
    precision = len(intersection) / max(len(active_frames), 1)
    recall = len(intersection) / max(len(reference_active), 1)
    iou = len(intersection) / max(len(union), 1)

    selected_roles = {str(item.get("role", "unknown")): int(item.get("id", -1)) for item in selected.get("selected", [])}
    return {
        "query_dir": str(query_dir),
        "selected_count": len(selected.get("selected", [])),
        "empty": bool(selected.get("empty", False)),
        "active_frame_count": len(active_frames),
        "first_active_frame": first_active,
        "last_active_frame": last_active,
        "active_iou": iou,
        "active_precision": precision,
        "active_recall": recall,
        "patient_match": int(selected_roles.get("patient", -1) == reference_roles.get("patient", -2)),
        "tool_match": int(selected_roles.get("tool", -1) == reference_roles.get("tool", -2)),
        "roles": selected_roles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--reference", required=True, help="label=/abs/path/to/query_dir")
    parser.add_argument("--method", action="append", required=True, help="label=/abs/path/to/query_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_label, reference_spec = parse_entry(args.reference)
    reference_dir, reference_selected, reference_validation = load_query_payload(reference_spec)
    reference_active = frame_set_from_validation(reference_validation)
    reference_roles = {str(item.get("role", "unknown")): int(item.get("id", -1)) for item in reference_selected.get("selected", [])}

    rows = [(reference_label, summarize_query_payload(reference_dir, reference_selected, reference_validation, reference_active, reference_roles))]
    for raw in args.method:
        label, query_spec = parse_entry(raw)
        query_dir, selected, validation = load_query_payload(query_spec)
        rows.append((label, summarize_query_payload(query_dir, selected, validation, reference_active, reference_roles)))

    lines = [
        f"# {args.title}",
        "",
        f"Reference: `{reference_label}`",
        "",
        "| Method | Active frames | First | Last | IoU | Precision | Recall | Patient match | Tool match | Selected count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rows:
        lines.append(
            "| {label} | {active} | {first} | {last} | {iou} | {precision} | {recall} | {patient} | {tool} | {count} |".format(
                label=label,
                active=fmt(row["active_frame_count"], digits=0),
                first=fmt(row["first_active_frame"], digits=0),
                last=fmt(row["last_active_frame"], digits=0),
                iou=fmt(row["active_iou"]),
                precision=fmt(row["active_precision"]),
                recall=fmt(row["active_recall"]),
                patient=fmt(row["patient_match"], digits=0),
                tool=fmt(row["tool_match"], digits=0),
                count=fmt(row["selected_count"], digits=0),
            )
        )
    lines.extend(["", "## Query Paths", ""])
    for label, row in rows:
        lines.append(f"- `{label}`: `{row['query_dir']}`")
    lines.append("")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

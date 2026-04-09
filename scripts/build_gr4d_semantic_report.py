import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_query_rows(benchmark_root: Path, extra_query_pack: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    curated_payload = _read_json(benchmark_root / "gr4d_curated_v1_queries.json")
    for item in curated_payload.get("queries", []):
        row = dict(item)
        row["pack"] = "curated"
        rows.append(row)
    if extra_query_pack is not None and extra_query_pack.exists():
        extra_payload = _read_json(extra_query_pack)
        for item in extra_payload.get("queries", []):
            row = dict(item)
            row["pack"] = "stress"
            rows.append(row)
    return rows


def _ranges_len(ranges: list[list[int]]) -> int:
    total = 0
    for segment in ranges or []:
        if not isinstance(segment, list) or len(segment) != 2:
            continue
        total += max(0, int(segment[1]) - int(segment[0]) + 1)
    return total


def _query_result(results_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    scene = str(item["scene"])
    query_id = str(item["query_id"])
    query_root = results_root / "queries" / scene / query_id
    status_path = query_root / "status.json"
    plan_path = query_root / "query_plan.json"
    selection_path = query_root / "selected_query_qwen.json"

    status_payload = _read_json(status_path) if status_path.exists() else {"status": "missing"}
    plan_payload = _read_json(plan_path) if plan_path.exists() else {}
    selection_payload = _read_json(selection_path) if selection_path.exists() else {}

    selected = selection_payload.get("selected", []) if isinstance(selection_payload, dict) else []
    refined_window = plan_payload.get("refined_temporal_window", {}) if isinstance(plan_payload, dict) else {}
    start_frame = refined_window.get("start_frame_index")
    end_frame = refined_window.get("end_frame_index")
    window_length = None
    if start_frame is not None and end_frame is not None:
        window_length = max(0, int(end_frame) - int(start_frame) + 1)

    target_count = item.get("target_count")
    selected_count = int(len(selected))
    exact_count_match = None
    non_empty_match = None
    if target_count is not None:
        exact_count_match = int(selected_count == int(target_count))
        non_empty_match = int((int(target_count) == 0 and selected_count == 0) or (int(target_count) > 0 and selected_count > 0))

    return {
        "scene": scene,
        "query_id": query_id,
        "query": str(item.get("text_en", item.get("query", ""))),
        "pack": str(item.get("pack", "unknown")),
        "status": str(status_payload.get("status", "missing")),
        "target_count": target_count,
        "selected_count": selected_count,
        "exact_count_match": exact_count_match,
        "non_empty_match": non_empty_match,
        "selection_mode": selection_payload.get("selection_mode"),
        "selection_subjects": selection_payload.get("subject_phrases", []),
        "selection_successors": selection_payload.get("successor_phrases", []),
        "selection_notes": selection_payload.get("notes", ""),
        "selected_entities": [
            {
                "id": int(row.get("id", -1)),
                "role": str(row.get("role", "entity")),
                "segment_count": int(len(row.get("segments", []))),
                "frame_count": _ranges_len(row.get("segments", [])),
            }
            for row in selected
        ],
        "planner_subjects": plan_payload.get("query_subject_phrases", []),
        "planner_successors": plan_payload.get("query_successor_phrases", []),
        "planner_detector_phrases": plan_payload.get("detector_phrases", []),
        "planner_notes": plan_payload.get("notes", ""),
        "query_semantic_profile": plan_payload.get("query_semantic_profile", {}),
        "start_frame_index": start_frame,
        "end_frame_index": end_frame,
        "window_length_frames": window_length,
        "log_path": status_payload.get("log_path", ""),
        "query_root": str(query_root),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "ok"]
    curated_rows = [row for row in rows if row["pack"] == "curated"]
    curated_success = [row for row in curated_rows if row["status"] == "ok"]
    exact_rows = [row for row in curated_success if row["exact_count_match"] is not None]
    non_empty_rows = [row for row in curated_success if row["non_empty_match"] is not None]
    return {
        "total_queries": int(len(rows)),
        "successful_queries": int(len(success_rows)),
        "failed_queries": int(len(rows) - len(success_rows)),
        "curated_queries": int(len(curated_rows)),
        "curated_successful_queries": int(len(curated_success)),
        "curated_exact_count_match_rate": 0.0 if not exact_rows else float(sum(int(row["exact_count_match"]) for row in exact_rows) / len(exact_rows)),
        "curated_non_empty_match_rate": 0.0 if not non_empty_rows else float(sum(int(row["non_empty_match"]) for row in non_empty_rows) / len(non_empty_rows)),
        "mean_window_length_frames": 0.0
        if not success_rows
        else float(
            sum(float(row["window_length_frames"]) for row in success_rows if row["window_length_frames"] is not None)
            / max(sum(1 for row in success_rows if row["window_length_frames"] is not None), 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--extra-query-pack", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    benchmark_root = Path(args.benchmark_root)
    results_root = Path(args.results_root)
    extra_query_pack = None if args.extra_query_pack in (None, "", "None") else Path(args.extra_query_pack)

    query_rows = _load_query_rows(benchmark_root, extra_query_pack)
    results = [_query_result(results_root, item) for item in query_rows]
    payload = {
        "benchmark_root": str(benchmark_root),
        "results_root": str(results_root),
        "extra_query_pack": None if extra_query_pack is None else str(extra_query_pack),
        "summary": _summary(results),
        "queries": results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    failed_rows = [row for row in results if row["status"] != "ok"]
    mismatch_rows = [row for row in results if row["status"] == "ok" and row["exact_count_match"] == 0]
    success_rows = [row for row in results if row["status"] == "ok"]

    lines = [
        "# GR4D Semantic Benchmark",
        "",
        f"- results_root: `{results_root}`",
        f"- total_queries: `{payload['summary']['total_queries']}`",
        f"- successful_queries: `{payload['summary']['successful_queries']}`",
        f"- failed_queries: `{payload['summary']['failed_queries']}`",
        f"- curated_exact_count_match_rate: `{payload['summary']['curated_exact_count_match_rate'] * 100.0:.2f}%`",
        f"- curated_non_empty_match_rate: `{payload['summary']['curated_non_empty_match_rate'] * 100.0:.2f}%`",
        f"- mean_window_length_frames: `{payload['summary']['mean_window_length_frames']:.2f}`",
        "",
        "| Query | Pack | Status | Target | Selected | Mode | Planner Subjects | Selection Subjects | Window |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['query_id']} | {row['pack']} | {row['status']} | {row['target_count']} | {row['selected_count']} | {row['selection_mode']} | {row['planner_subjects']} | {row['selection_subjects']} | {row['start_frame_index']}-{row['end_frame_index']} |"
        )

    lines.extend(["", "## Failures", ""])
    if not failed_rows:
        lines.append("- none")
    else:
        for row in failed_rows:
            lines.append(f"- `{row['query_id']}`: status=`{row['status']}`, log=`{row['log_path']}`")

    lines.extend(["", "## Count Mismatches", ""])
    if not mismatch_rows:
        lines.append("- none")
    else:
        for row in mismatch_rows:
            lines.append(
                f"- `{row['query_id']}`: target=`{row['target_count']}`, selected=`{row['selected_count']}`, notes=`{row['selection_notes']}`"
            )

    lines.extend(["", "## Notes", ""])
    for row in success_rows[:10]:
        lines.append(
            f"- `{row['query_id']}`: mode=`{row['selection_mode']}`, planner={row['planner_subjects']}, selection={row['selection_subjects']}, window={row['start_frame_index']}-{row['end_frame_index']}"
        )

    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import json
import re
from pathlib import Path


GROUP_BY_SCENE = {
    "chickchicken": "interp",
    "torchocolate": "interp",
    "americano": "misc",
    "espresso": "misc",
    "keyboard": "misc",
    "split-cookie": "misc",
}


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def normalize_query(object_name: str, state_name: str | None = None) -> str:
    object_name = " ".join(object_name.strip().split())
    if not state_name:
        return f"the {object_name}"
    state_name = " ".join(state_name.strip().split())
    lowered_state = state_name.lower()
    lowered_object = object_name.lower()
    if lowered_object in lowered_state:
        query = lowered_state
        if not query.startswith("the "):
            query = f"the {query}"
        return query
    return f"the {object_name} that is {lowered_state}"


def merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    if not ranges:
        return []
    ordered = sorted([[int(start), int(end)] for start, end in ranges], key=lambda item: (item[0], item[1]))
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev = merged[-1]
        if start <= prev[1] + 1:
            prev[1] = max(prev[1], end)
            continue
        merged.append([start, end])
    return merged


def build_scene_queries(scene: str, payload: dict) -> list[dict]:
    group = GROUP_BY_SCENE.get(scene)
    if group is None:
        raise ValueError(f"Unsupported 4DLangSplat scene-group mapping for {scene}")
    scene_name = f"HyperNeRF/{group}/{scene}"
    rows: list[dict] = []
    for object_name, state_map in payload.items():
        if not isinstance(state_map, dict):
            continue
        union_ranges: list[list[int]] = []
        for state_ranges in state_map.values():
            if isinstance(state_ranges, list):
                union_ranges.extend(state_ranges)
        object_query = normalize_query(object_name)
        rows.append(
            {
                "scene": scene_name,
                "query_slug": f"{scene}__{slugify(object_query)}",
                "query": object_query,
                "category": "static_reference",
                "targets": [
                    {
                        "role": "entity",
                        "target_entity_id": -1,
                        "target_ranges": merge_ranges(union_ranges),
                        "gt_mask_dir": "",
                    }
                ],
                "gt_union_mask_dir": "",
            }
        )
        for state_name, state_ranges in state_map.items():
            if not isinstance(state_ranges, list):
                continue
            query = normalize_query(object_name, state_name)
            rows.append(
                {
                    "scene": scene_name,
                    "query_slug": f"{scene}__{slugify(query)}",
                    "query": query,
                    "category": "temporal_state_reference",
                    "targets": [
                        {
                            "role": "entity",
                            "target_entity_id": -1,
                            "target_ranges": merge_ranges(state_ranges),
                            "gt_mask_dir": "",
                        }
                    ],
                    "gt_union_mask_dir": "",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.annotation_root)
    scenes = [args.scene] if args.scene else sorted(
        path.parent.name for path in root.glob("*/video_annotations.json")
    )

    queries: list[dict] = []
    for scene in scenes:
        scene_path = root / scene / "video_annotations.json"
        if not scene_path.exists():
            raise FileNotFoundError(scene_path)
        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        queries.extend(build_scene_queries(scene, payload))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"queries": queries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()

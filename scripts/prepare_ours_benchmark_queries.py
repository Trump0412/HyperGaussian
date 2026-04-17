#!/usr/bin/env python3
"""
prepare_ours_benchmark_queries.py

从 benchmark JSON 和 GR4D-Bench 补全所有查询文本，
生成用于 pipeline 运行的 benchmark_full_queries.json 与 TSV。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSON = Path(
    os.environ.get(
        "OURS_BENCHMARK_JSON",
        str(REPO_ROOT / "data" / "benchmarks" / "r4d_bench_qa" / "benchmark.json"),
    )
)
GR4D_BENCH_ROOT = Path(
    os.environ.get(
        "GR4D_BENCH_ROOT",
        str(REPO_ROOT / "data" / "GR4D-Bench" / "data" / "scenes"),
    )
)
REPORT_DIR = Path(
    os.environ.get(
        "OURS_BENCHMARK_REPORT_DIR",
        str(REPO_ROOT / "reports" / "ours_benchmark_eval"),
    )
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = REPORT_DIR / "benchmark_full_queries.json"
OUTPUT_TSV = REPORT_DIR / "benchmark_queries.tsv"


def load_gr4d_queries() -> dict[str, str]:
    """Load GR4D-Bench queries and return query_id -> text mapping."""
    result: dict[str, str] = {}
    if not GR4D_BENCH_ROOT.exists():
        return result

    for scene_dir in GR4D_BENCH_ROOT.iterdir():
        if not scene_dir.is_dir():
            continue
        for fname in (
            f"{scene_dir.name}_queries.json",
            f"{scene_dir.name.replace('_', '-')}_queries.json",
            "queries.json",
        ):
            qpath = scene_dir / fname
            if not qpath.exists():
                continue
            try:
                queries = json.loads(qpath.read_text(encoding="utf-8"))
                for q in queries:
                    qid = str(q.get("query_id", "")).strip()
                    text_zh = str(q.get("text_zh", "")).strip()
                    text_en = str(q.get("text_en", "")).strip()
                    if qid:
                        result[qid] = text_zh if text_zh else text_en
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 读取 {qpath} 失败: {exc}")
            break
    return result


def main() -> None:
    if not BENCHMARK_JSON.exists():
        raise SystemExit(
            f"Missing benchmark json: {BENCHMARK_JSON}\n"
            "Set OURS_BENCHMARK_JSON to a valid benchmark file path."
        )

    benchmark = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    if not isinstance(benchmark, list):
        raise SystemExit(f"Invalid benchmark format, expected list: {BENCHMARK_JSON}")

    gr4d_queries = load_gr4d_queries()
    print(f"GR4D-Bench 查询数: {len(gr4d_queries)}")

    full_queries: list[dict] = []
    tsv_lines: list[str] = []

    for item in benchmark:
        qid = str(item["query_id"])
        question = str(item.get("question", "")).strip()

        if not question:
            gr4d_text = gr4d_queries.get(qid, "")
            if gr4d_text:
                print(f"[fill] {qid}: 从 GR4D-Bench 补全 \"{gr4d_text}\"")
                question = gr4d_text
            else:
                print(f"[warn] {qid}: 无法找到查询文本，保留为空")

        gt = item.get("ground_truth", {})
        entry = {
            "query_id": qid,
            "question": question,
            "has_gt": bool(gt.get("frames")),
            "gt_tracks": gt.get("target_tracks", []),
            "existence_frames_count": len(gt.get("existence_frames", [])),
        }
        full_queries.append(entry)
        tsv_lines.append(f"{qid}\t{question}")

    OUTPUT_JSON.write_text(json.dumps(full_queries, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_TSV.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    print(f"保存完整查询: {OUTPUT_JSON}")
    print(f"保存 TSV: {OUTPUT_TSV}")
    print("\n=== 查询汇总 ===")
    print(f"总查询: {len(full_queries)}")
    print(f"仍然空查询: {sum(1 for q in full_queries if not q['question'])}")
    print(f"有 mask GT: {sum(1 for q in full_queries if q['has_gt'])}")
    print(f"负样本查询: {sum(1 for q in full_queries if not q['gt_tracks'])}")


if __name__ == "__main__":
    main()

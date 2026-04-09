#!/usr/bin/env python3
"""
prepare_ours_benchmark_queries.py

从 Ours_benchmark.json 和 GR4D-Bench 补全所有查询文本，
生成用于pipeline运行的 benchmark_full_queries.json。
同时输出 TSV 格式供 shell 脚本使用。
"""
from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_JSON = Path("/root/autodl-tmp/data/Ours_benchmark.json")
GR4D_BENCH_ROOT = Path("/root/autodl-tmp/GR4D-Bench/data/scenes")
REPORT_DIR = Path("/root/autodl-tmp/GaussianStellar/reports/ours_benchmark_eval")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = REPORT_DIR / "benchmark_full_queries.json"
OUTPUT_TSV = REPORT_DIR / "benchmark_queries.tsv"


def load_gr4d_queries() -> dict[str, str]:
    """Load GR4D-Bench queries and return query_id -> text_zh mapping."""
    result: dict[str, str] = {}
    if not GR4D_BENCH_ROOT.exists():
        return result
    for scene_dir in GR4D_BENCH_ROOT.iterdir():
        if not scene_dir.is_dir():
            continue
        # Try various naming patterns for queries.json
        for fname in [
            f"{scene_dir.name}_queries.json",
            f"{scene_dir.name.replace('_', '-')}_queries.json",
            "queries.json",
        ]:
            qpath = scene_dir / fname
            if qpath.exists():
                try:
                    queries = json.loads(qpath.read_text(encoding="utf-8"))
                    for q in queries:
                        qid = str(q.get("query_id", ""))
                        text_zh = str(q.get("text_zh", "")).strip()
                        text_en = str(q.get("text_en", "")).strip()
                        if qid:
                            # 优先中文
                            result[qid] = text_zh if text_zh else text_en
                except Exception as e:
                    print(f"[warn] 读取 {qpath} 失败: {e}")
                break
    return result


def main() -> None:
    benchmark = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    gr4d_queries = load_gr4d_queries()
    print(f"GR4D-Bench 查询数: {len(gr4d_queries)}")

    full_queries: list[dict] = []
    tsv_lines: list[str] = []

    for item in benchmark:
        qid = str(item["query_id"])
        question = str(item.get("question", "")).strip()

        # 补全空查询文本
        if not question:
            gr4d_text = gr4d_queries.get(qid, "")
            if gr4d_text:
                print(f"[fill] {qid}: 从GR4D-Bench补全 \"{gr4d_text}\"")
                question = gr4d_text
            else:
                print(f"[warn] {qid}: 无法找到查询文本，跳过")

        entry = {
            "query_id": qid,
            "question": question,
            "has_gt": bool(item["ground_truth"].get("frames")),
            "gt_tracks": item["ground_truth"].get("target_tracks", []),
            "existence_frames_count": len(item["ground_truth"].get("existence_frames", [])),
        }
        full_queries.append(entry)
        # TSV: query_id \t question
        tsv_lines.append(f"{qid}\t{question}")

    OUTPUT_JSON.write_text(json.dumps(full_queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"保存完整查询: {OUTPUT_JSON}")

    OUTPUT_TSV.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    print(f"保存TSV: {OUTPUT_TSV}")

    # 打印汇总
    print("\n=== 查询汇总 ===")
    print(f"总查询: {len(full_queries)}")
    empty_q = [q for q in full_queries if not q["question"]]
    print(f"仍然空查询: {len(empty_q)} - {[q['query_id'] for q in empty_q]}")
    with_gt = [q for q in full_queries if q["has_gt"]]
    print(f"有mask GT: {len(with_gt)}")
    negative = [q for q in full_queries if not q["gt_tracks"]]
    print(f"负样本查询: {len(negative)}")


if __name__ == "__main__":
    main()

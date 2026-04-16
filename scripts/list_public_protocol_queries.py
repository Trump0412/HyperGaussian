#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("Usage: list_public_protocol_queries.py <protocol_json>")

protocol_path = Path(sys.argv[1])
payload = json.loads(protocol_path.read_text(encoding="utf-8"))
for item in payload.get("queries", []):
    slug = str(item.get("query_slug", "")).strip()
    query = str(item.get("query", "")).strip()
    if slug and query:
        print(f"{slug}\t{query}")

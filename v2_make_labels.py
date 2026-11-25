"""Write labels.jsonl next to every generations.jsonl: the same rows without the model output text.

generations.jsonl is not committed (outputs often copy third-party host/attack text); labels.jsonl keeps ids, labels and
the automatic checker results, which is all the analysis needs. Run once after v2gen:
  uv run python v2_make_labels.py results_v2/Qwen__Qwen2.5-7B-Instruct
"""
import json
import sys
from pathlib import Path

KEEP = ("id", "kind", "family", "goal", "neg_type", "task", "follow", "task_ok", "degenerate")
for d in map(Path, sys.argv[1:]):
    for sub in ("", "real"):
        src = d / sub / "generations.jsonl"
        if not src.exists():
            continue
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        with open(d / sub / "labels.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in KEEP}, ensure_ascii=False) + "\n")
        print("wrote", d / sub / "labels.jsonl", len(rows))

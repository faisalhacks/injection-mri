"""Rebuild results_v2/<model>/prompts.jsonl and results_v2/<model>/real/prompts.jsonl without a GPU.

The prompt files contain third-party host and attack text, so they are not committed. They are a deterministic function of
data_v2/hosts.jsonl and data_v2/real_attacks.jsonl (rebuilt by fetch_v2_data.py) and of data_v2.py (seed 0). This script writes
them in the exact format of v2.gen() and compares their SHA-256 with results_v2/prompts_sha256.json (the hashes of the files
the experiments used).

  uv run --with pandas --with pyarrow --with ijson fetch_v2_data.py
  uv run python v2_rebuild_prompts.py            # writes both models' folders, checks hashes
  uv run python v2_rebuild_prompts.py --check    # only compare, do not write (writes nothing)
"""
import hashlib
import json
import sys
from pathlib import Path

import data_v2 as D

HERE = Path(__file__).parent
KEYS = ("id", "hid", "host_kind", "host_fold", "dev", "task", "kind", "family", "goal", "neg_type", "position", "template",
        "params", "inj_spans", "instr", "system", "doc_text", "text", "meta", "source")


def render(ex):
    return "".join(json.dumps({k: e[k] for k in KEYS}, ensure_ascii=False) + "\n" for e in ex).encode("utf-8")


def main():
    check_only = "--check" in sys.argv
    want = json.loads((HERE / "results_v2" / "prompts_sha256.json").read_text(encoding="utf-8"))
    blobs = {"prompts.jsonl": render(D.build(seed=0)), "real/prompts.jsonl": render(D.build_real())}
    bad = 0
    for name, b in blobs.items():
        h = hashlib.sha256(b).hexdigest()
        ok = h == want[name]
        bad += not ok
        print("MATCH   " if ok else "MISMATCH", name, h)
        if not check_only:
            for m in sorted((HERE / "results_v2").glob("Qwen__*")):
                p = m / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

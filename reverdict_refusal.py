"""Re-apply the pre-registered minimum-n rule (H4: n < 30 -> "untestable") to stored summary_<mode>.json files and
regenerate the matching HTML report from them. Uses stored data only: no model is loaded and nothing is re-run.

    python reverdict_refusal.py results_refusal/Qwen__Qwen2.5-7B-Instruct [more model dirs ...]
"""
import json
import sys
from pathlib import Path

import refusal


def main(dirs):
    for d in dirs:
        for mode in ("bypass", "overrefusal"):
            p = Path(d) / f"summary_{mode}.json"
            if not p.exists():
                continue
            S = json.loads(p.read_text(encoding="utf8"))
            V = S.get("verify")
            if not V or "verdict" not in V or "n_eval" not in V:
                continue
            before = json.dumps(V["verdict"], sort_keys=True)
            refusal.apply_min_n_rule(V["verdict"], V["n_eval"], mode)
            if json.dumps(V["verdict"], sort_keys=True) == before:
                print("summary unchanged (report regenerated)", p)
            else:
                p.write_text(json.dumps(S, indent=1, default=float), encoding="utf8")
            refusal.write_report(S, Path(d) / f"report_{mode}.html")
            print("updated", p, V["verdict"])


if __name__ == "__main__":
    main(sys.argv[1:])

"""Compact text summary of results_refusal/<model>/summary_<mode>.json (numbers only).
  python summarize_refusal.py results_refusal/Qwen__Qwen2.5-0.5B-Instruct/summary_bypass.json
"""
import json
import re
import sys

S = json.load(open(sys.argv[1], encoding="utf8"))
V = S.get("verify", {})
print(S["model"], S["mode"], "pairs", S.get("n_pairs"), "audit", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.get("audit", {}).items()})
print("baseline refusal:", {k: round(v, 3) for k, v in S.get("baseline_refusal_rates", {}).items()})
print("AtP vs exact:", S.get("atp_vs_exact"))
print("n_eval", V.get("n_eval"), "baseline outcome", V.get("baseline_outcome_rate"), "by family", V.get("eval_by_family"))
for k, v in V.get("configs", {}).items():
    if re.search(r"_\d+$", k):
        continue
    print(f"  {k:<12} {v['rate']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
import numpy as np
for pre in sorted({re.sub(r"_\d+$", "", k) for k in V.get("configs", {}) if re.search(r"rand_k\d+_\d+$", k)}):
    r = [v["rate"] for k, v in V["configs"].items() if k.startswith(pre + "_") and re.search(r"_\d+$", k)]
    print(f"  {pre:<12} n={len(r)} mean {np.mean(r):.3f} p95 {np.percentile(r, 95):.3f} max {np.max(r):.3f}")
print("primary", json.dumps(V.get("primary")))
print("primary heads", json.dumps(V.get("primary_heads")))
print("units match direction at k:", V.get("units_match_dir_at_k"))
for nm, d in V.get("side", {}).items():
    print(f"  side {nm:<12}", " ".join(f"{k}={x['value']:.3f}({x['delta']:+.3f})" for k, x in d.items()))
print("verdict", json.dumps(V.get("verdict")))
F = S.get("family", {})
for g in ("fam", "kw"):
    r = F.get(g)
    if r:
        print(g, "J_between", r["J_between"], "J_within", r["J_within"], "null99", r["null_p99"], r["H5a_verdict"], "heads:", r.get("J_between_heads"), r.get("J_within_heads"), "skipped", r["skipped_too_small"])
if "H5b" in F:
    b = F["H5b"]
    print("H5b rho", b["rho"], "q", b["q"], b["verdict"])
print("H6", json.dumps(F.get("H6")))

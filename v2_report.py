"""Plots + audit CSV for results_v2/<model>/ (reads results.json, prompts.jsonl, generations.jsonl).
  uv run v2_report.py results_v2/Qwen__Qwen2.5-0.5B-Instruct
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
R = json.loads((out / "results.json").read_text(encoding="utf-8"))
P = {}
for l in open(out / "prompts.jsonl", encoding="utf-8"):
    p = json.loads(l)
    P[p["id"]] = p
G = [json.loads(l) for l in open(out / "generations.jsonl", encoding="utf-8")]

# ---- audit sample: 12 per (follow label among injected) + 12 per goal checker 'ignore' + 30 clean/hardneg, CSV for hand labelling
rng = np.random.default_rng(0)
rows = []
for lab in ("pure", "mixed", "ignore"):
    pool = [g for g in G if g["kind"] == "injected" and g["follow"] == lab]
    rows += [pool[i] for i in rng.permutation(len(pool))[:30]]
for kind in ("clean", "hardneg"):
    pool = [g for g in G if g["kind"] == kind]
    rows += [pool[i] for i in rng.permutation(len(pool))[:20]]
with open(out / "audit_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "kind", "family", "goal", "neg_type", "task", "params", "injected_text", "output", "auto_follow", "auto_task_ok",
                "hand_follow", "hand_task_ok"])
    for g in rows:
        p = P[g["id"]]
        inj = " || ".join(p["doc_text"][a:b] for a, b in p["inj_spans"])
        w.writerow([g["id"], g["kind"], g["family"], g["goal"], g["neg_type"], g["task"], json.dumps(p["params"]), inj[:400],
                    g["output"].replace("\n", "\\n")[:500], g["follow"], g["task_ok"], "", ""])

# ---- plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
names = ["probe_lr", "probe_dom", "probe_meanpool", "tfidf", "length", "length_pos_oracle", "deberta", "attn_tracker", "entropy"]
for ax, key, title in ((axes[0], "H1_leave_one_goal_out", "H1 held-out goal family"),
                       (axes[1], "H2_leave_one_attack_family_out", "H2 held-out attack family")):
    r = R.get(key, {})
    if r.get("untestable"):
        ax.set_title(title + " (untestable)")
        continue
    v = [r["macro_auroc"][m] for m in names]
    lo = [r["macro_auroc"][m] - r["ci95"][m][0] for m in names]
    hi = [r["ci95"][m][1] - r["macro_auroc"][m] for m in names]
    ax.bar(range(len(names)), v, yerr=[lo, hi], color=["#c44"] + ["#e99"] * 2 + ["#888"] * 6, capsize=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, fontsize=7)
    ax.set_ylim(0.3, 1.02)
    ax.axhline(0.5, color="k", lw=.5)
    ax.set_title(f"{title}\nmacro AUROC, {r['n_valid_folds']} valid folds, base follow {r['base_follow_rate_pure_over_injected']:.1%}", fontsize=9)
h = R.get("H3_injected_vs_hard_negative")
ax = axes[2]
if h:
    v = [h["fpr_at_tpr_0.9"].get(m, np.nan) for m in names]
    ax.bar(range(len(names)), v, color=["#c44"] + ["#e99"] * 2 + ["#888"] * 6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, fontsize=7)
    ax.set_title("H3 FPR on hard negatives at TPR 0.9 (lower is better)", fontsize=9)
fig.tight_layout()
fig.savefig(out / "h1_h2_h3.png", dpi=130)
plt.close(fig)

hb = R.get("H4_blame")
if hb and hb.get("causal_probe_drop"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ms = ["causal_probe_drop", "mean_patch_drop_exploratory", "attention_tracker", "projection_only", "random_expected"]
    x = np.arange(len(ms))
    ax.bar(x - 0.2, [hb[m]["top1"] for m in ms], 0.4, label="top-1")
    ax.bar(x + 0.2, [hb[m]["top3"] for m in ms], 0.4, label="top-3")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in ms], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"H4 blame: planted sentence located (n={hb['n_prompts']} followed injections)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "h4_blame.png", dpi=130)
    plt.close(fig)
print("wrote", out / "audit_sample.csv", "and plots")

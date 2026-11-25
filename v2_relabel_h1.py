"""Re-run H1 (leave-one-goal-family-out) after replacing the automatic follow label with the hand label on the audited rows.

  uv run python v2_relabel_h1.py results_v2/Qwen__Qwen2.5-7B-Instruct [n_jobs]

Needs results_v2/<model>/prompts.jsonl, generations.jsonl and gen/ (all rebuilt by the pipeline, not committed; see README
"Regenerating the untracked files") plus the committed audit_sample.csv. Only rows whose `hand_follow` is non-empty are
changed; every other row keeps its automatic label, so this is a PARTIAL correction, not a re-labelling of the study.
The labels in audit_sample.csv were written by an AI agent (column `labeler`), not a human.
Everything else (layers, C grid, folds, baselines, bootstrap) is the unchanged pipeline of v2_analysis.fit().
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import v2_analysis as V

out = Path(sys.argv[1])
n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
hand = {}
for r in csv.DictReader(open(out / "audit_sample.csv", encoding="utf-8", newline="")):
    if r["hand_follow"]:
        hand[int(r["id"])] = r["hand_follow"]

R = V.Run(out)
inj = R.kind == "injected"
auto = R.follow.copy()
changed = Counter()
id2i = {p["id"]: i for i, p in enumerate(R.P)}
for i_d, lab in hand.items():
    i = id2i[i_d]
    assert inj[i], i_d
    changed[(R.goal[i], auto[i], lab)] += 1
    R.follow[i] = lab
n_changed = sum(v for (g, a, l), v in changed.items() if a != l)
print("hand-labelled rows:", len(hand), "of which the automatic label changed:", n_changed)

layers = list(range(0, R.L, 2))
if R.L - 1 not in layers:
    layers.append(R.L - 1)
cand = inj & (R.follow != "mixed")
y = (R.follow == "pure").astype(int)
folds_all = V.make_folds(R, "goal", cand)
folds = [f for f in folds_all if V.valid(f, y)]
skipped = [f["name"] for f in folds_all if not V.valid(f, y)]
per_fold = []
for f in folds:
    S, _ = V.all_methods(R, y, f["tr"], f["va"], f["te"], layers, n_jobs)
    per_fold.append(S)
    print(f["name"], {m: round(V.auc(y[f["te"]], S[m]), 3) for m in S}, flush=True)
rep = V.macro_report(R, y, folds, per_fold, 1000)
rep["skipped_folds"] = skipped
rep["n_hand_labelled_rows"] = len(hand)
rep["n_rows_whose_label_changed"] = n_changed
rep["label_changes_goal_auto_hand_count"] = {f"{g}|{a}|{l}": c for (g, a, l), c in sorted(changed.items())}
rep["labeler"] = "AI agent (Claude Sonnet 5.5), not human; partial correction (audited rows only)"
orig = json.loads((out / "results.json").read_text(encoding="utf-8"))["H1_leave_one_goal_out"]
rep["original_H1"] = {"macro_auroc_probe_lr": orig["macro_auroc"]["probe_lr"], "best_baseline": orig["best_baseline"],
                      "macro_auroc_best_baseline": orig["macro_auroc"][orig["best_baseline"]], "verdict": orig["verdict"]}
(out / "h1_relabelled_audited.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
b = rep["best_baseline"]
print(json.dumps({"probe": rep["macro_auroc"]["probe_lr"], "best_baseline": b, "best_baseline_auroc": rep["macro_auroc"][b],
                  "margin": rep["probe_minus_best_baseline"], "probe_ci": rep["ci95"]["probe_lr"], "verdict": rep["verdict"],
                  "valid_folds": [f["name"] for f in folds], "skipped": skipped}, indent=1))

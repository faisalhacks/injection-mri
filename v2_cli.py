"""v2 command-line helpers: scan / guard / report / blame with the goal-agnostic v2 probe (imported by mri.py)."""
import json
import sys
from pathlib import Path

import numpy as np

import data_v2 as D
import mri
import v2


def load_probe(model):
    p = v2.outdir2(model) / "probe_v2.npz"
    if not p.exists():
        sys.exit(f"no v2 probe at {p}; run `uv run mri.py v2gen/v2fit --model {model}` (or fetch results_v2/) first")
    return np.load(p, allow_pickle=True)


def _read(args):
    return Path(args.prompt_file).read_text(encoding="utf-8")


def print_blame(res, model):
    s0, thr = res["s0"], res["thr"]
    print(f"P(model follows an instruction found in the data) = {s0:.2f}   (alarm threshold {thr:.2f}; probe layer {res['layer']}; "
          f"{model})  ->  {'ALARM' if s0 >= thr else 'below threshold'}")
    print("Goal-agnostic: the probe never sees what the attacker wants. Blame = drop in that score when the sentence is deleted.\n")
    order = np.argsort(-np.array(res["drop"]))
    top = int(order[0])
    for i, (u, d) in enumerate(zip(res["units"], res["drop"])):
        bar = "#" * int(round(max(d, 0) / max(s0, 1e-6) * 20))
        print(f"{'>>' if i == top else '  '} [{i:02d}] drop {d:+.2f} {bar:<20} {u[:90].replace(chr(10), ' ')!r}")
    if "out_without_top" in res:
        print(f"\n--- output as is ---\n{res['out_full']}\n--- output with unit {res['top']} deleted ---\n{res['out_without_top']}")


def scan_v2(args):
    tok, model = mri.load(args.model)
    res = v2.blame_prompt(tok, model, load_probe(args.model), _read(args))
    print_blame(res, args.model)


def blame_cmd(args):
    tok, model = mri.load(args.model)
    res = v2.blame_prompt(tok, model, load_probe(args.model), _read(args), regen=args.regen)
    print_blame(res, args.model)
    if args.out:
        write_html(args, res)


def guard_v2(args):
    """If the probe fires, delete the single most-blamed sentence and generate from the sanitised document."""
    tok, model = mri.load(args.model)
    res = v2.blame_prompt(tok, model, load_probe(args.model), _read(args))
    instr, doc_text = v2.doc_region(_read(args))
    e = dict(system=v2.DEFAULT_SYSTEM, instr=instr, doc_text=doc_text, text=D.user_message(instr, doc_text))
    if res["s0"] >= res["thr"]:
        top = int(np.argmax(res["drop"]))
        e = v2.variant(e, v2.delete_unit(doc_text, v2.units_for(doc_text), top))
        print(f"probe {res['s0']:.2f} >= {res['thr']:.2f}: deleting unit {top}: {res['units'][top][:100]!r}")
    else:
        print(f"probe {res['s0']:.2f} < {res['thr']:.2f}: no change")
    print("---\n" + v2.generate(tok, model, [e], bs=1, max_new=args.max_new)[0])


def write_html(args, res):
    import html
    outp = v2.outdir2(args.model) / "results.json"
    R = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
    s0 = res["s0"]
    spans = []
    for i, (u, d) in enumerate(zip(res["units"], res["drop"])):
        t = float(min(max(d / max(s0, 1e-6), 0), 1))
        spans.append(f'<div style="background:rgba(220,40,40,{t:.2f});padding:2px 6px;margin:2px 0" title="drop {d:+.3f}">'
                     f'<small>[{i}] {d:+.2f}</small> {html.escape(u)}</div>')
    rows = []
    for key, name in (("H1_leave_one_goal_out", "held-out goal family"), ("H2_leave_one_attack_family_out", "held-out attack family")):
        r = R.get(key)
        if r and not r.get("untestable"):
            rows.append(f"<tr><td>{name}: probe macro AUROC</td><td>{r['macro_auroc']['probe_lr']:.3f}</td></tr>"
                        f"<tr><td>{name}: best baseline ({r['best_baseline']})</td><td>{r['macro_auroc'][r['best_baseline']]:.3f}</td></tr>")
    hb = R.get("H4_blame")
    if hb and hb.get("causal_probe_drop"):
        rows.append(f"<tr><td>blame top-1 (causal / attention tracker / projection)</td><td>{hb['causal_probe_drop']['top1']:.2f} / "
                    f"{hb['attention_tracker']['top1']:.2f} / {hb['projection_only']['top1']:.2f}</td></tr>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>injection-mri v2 blame</title><style>body{{font:14px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px}}
td{{border-bottom:1px solid #eee;padding:3px 10px}}</style></head><body><h1>injection-mri v2: blame map</h1>
<p><b>Model:</b> {html.escape(args.model)} &nbsp; <b>File:</b> {html.escape(args.prompt_file)}</p>
<p><b>Follow probability:</b> {s0:.2f} (alarm threshold {res['thr']:.2f}) &mdash; {'ALARM' if s0 >= res['thr'] else 'below threshold'}</p>
<p><small>Red = how much the follow-probability drops when that sentence is deleted. The probe is goal-agnostic and was trained on
synthetic injections planted in real documents; see results_v2 for what held out.</small></p>{''.join(spans)}
<h2>What held out (test numbers from results.json)</h2><table>{''.join(rows)}</table></body></html>"""
    dst = Path(args.out)
    dst.write_text(doc, encoding="utf-8")
    print("wrote", dst)


def report_v2(args):
    tok, model = mri.load(args.model)
    res = v2.blame_prompt(tok, model, load_probe(args.model), _read(args))
    print_blame(res, args.model)
    if not args.out:
        args.out = str(v2.outdir2(args.model) / f"report_{Path(args.prompt_file).stem}.html")
    write_html(args, res)

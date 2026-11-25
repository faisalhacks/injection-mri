"""injection-mri v2, model side: generation + activations + attention features + DeBERTa baseline (`gen`),
per-sentence causal blame runs (`blame_run`), and helpers used by scan / guard / report / blame.

Analysis (probes, baselines, bootstrap, hypotheses) lives in v2_analysis.py and runs on CPU from the saved arrays.
"""
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import data_v2 as D
import mri

MARK = '\nDocument:\n"""\n'
MAX_NEW = 96


def outdir2(model):
    d = Path(os.environ.get("MRI_V2_OUT", mri.ROOT / "results_v2")) / model.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================ encoding
def prompt_str(tok, system, text):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode2(tok, exs):
    """Batch encoding. Returns enc, doc mask [B,T], instruction mask [B,T] (everything before the document except the
    first token = attention sink), doc start char per row, offsets."""
    prompts = [prompt_str(tok, e["system"], e["text"]) for e in exs]
    enc = tok(prompts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    off = enc.pop("offset_mapping")
    B, T = enc["input_ids"].shape
    doc = torch.zeros(B, T, dtype=torch.bool)
    ins = torch.zeros(B, T, dtype=torch.bool)
    ds_all = []
    for b, (p, e) in enumerate(zip(prompts, exs)):
        ds = p.find(e["text"]) + e["text"].index(MARK) + len(MARK)
        de = ds + len(e["doc_text"])
        st, en, am = off[b, :, 0], off[b, :, 1], enc["attention_mask"][b].bool()
        doc[b] = am & (en > ds) & (st < de) & (en > st)
        ins[b] = am & (en <= ds) & (en > st)
        first = int(am.nonzero()[0])
        ins[b, first] = False
        ds_all.append(ds)
    return enc, doc, ins, ds_all, off


def position_ids(enc):
    return (enc["attention_mask"].cumsum(1) - 1).clamp(min=0)


@torch.no_grad()
def forward_acts(model, enc, doc_mask, keep_layers=None, patch=None):
    """One prefill pass. Returns last-token residual and masked doc-token mean per residual index [B,d] each,
    last-token logits [B,V], and (for layers in keep_layers) all-token residuals on CPU.
    patch: optional (token_mask [B,T] bool, per-index replacement vectors [n_hidden, d]) = mean-patching."""
    layers = mri.find_layers(model)
    n = len(layers) + 1
    last, docm, full = [None] * n, [None] * n, {}
    dm = doc_mask.to(mri.DEV)

    def make(i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1:
                return None
            if patch is not None and i >= 1:
                tm, vec = patch
                h = torch.where(tm.to(h.device).unsqueeze(-1), vec[i].to(h.device, h.dtype).expand_as(h), h)
            last[i] = h[:, -1].float().cpu()
            w = dm.unsqueeze(-1).to(h.dtype)
            docm[i] = ((h * w).sum(1) / w.sum(1).clamp(min=1)).float().cpu()
            if keep_layers is not None and i in keep_layers:
                full[i] = h.float().cpu()
            if patch is not None and i >= 1:
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return None
        return fn

    hs = [model.get_input_embeddings().register_forward_hook(make(0))]
    hs += [l.register_forward_hook(make(i + 1)) for i, l in enumerate(layers)]
    try:
        e = {k: v.to(mri.DEV) for k, v in enc.items()}
        o = model(**e, position_ids=position_ids(e), logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return torch.stack(last, 1), torch.stack(docm, 1), o.logits[:, -1].float().cpu(), full


@torch.no_grad()
def generate(tok, model, exs, bs=16, max_new=MAX_NEW):
    order = np.argsort([len(e["text"]) for e in exs])
    outs = [None] * len(exs)
    for b in range(0, len(exs), bs):
        idx = order[b:b + bs]
        enc, _, _, _, _ = encode2(tok, [exs[i] for i in idx])
        enc = {k: v.to(mri.DEV) for k, v in enc.items()}
        T = enc["input_ids"].shape[1]
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                           temperature=None, top_p=None, top_k=None)
        for j, o in zip(idx, tok.batch_decode(g[:, T:], skip_special_tokens=True)):
            outs[j] = o
        mri.guard()
    return outs


@torch.no_grad()
def attention_feats(tok, model, exs, bs=4):
    """Per prompt, [L,H]: attention mass of the last token on instruction tokens and on document tokens."""
    try:
        model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
    order = np.argsort([len(e["text"]) for e in exs])
    A_i, A_d = [None] * len(exs), [None] * len(exs)
    for b in range(0, len(exs), bs):
        idx = order[b:b + bs]
        enc, doc, ins, _, _ = encode2(tok, [exs[i] for i in idx])
        e = {k: v.to(mri.DEV) for k, v in enc.items()}
        o = model(**e, position_ids=position_ids(e), output_attentions=True, logits_to_keep=1)
        att = torch.stack([a[:, :, -1, :].float() for a in o.attentions], 1)  # [B,L,H,T]
        mi, md = ins.to(att.device)[:, None, None, :], doc.to(att.device)[:, None, None, :]
        ai, ad = (att * mi).sum(-1).cpu().numpy(), (att * md).sum(-1).cpu().numpy()
        for k, i in enumerate(idx):
            A_i[i], A_d[i] = ai[k], ad[k]
        del o, att
        mri.guard()
    try:
        model.set_attn_implementation("sdpa")
    except Exception:
        model.config._attn_implementation = "sdpa"
    return np.stack(A_i), np.stack(A_d)


def deberta_scores(texts, bs=16):
    """protectai/deberta-v3-base-prompt-injection-v2: max over 512-token windows (stride 256) of P(INJECTION)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "protectai/deberta-v3-base-prompt-injection-v2"
    tk = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(mri.DEV).eval()
    inj = [i for i, l in m.config.id2label.items() if "INJ" in l.upper()][0]
    out = []
    for t in texts:
        ids = tk(t, add_special_tokens=False)["input_ids"]
        wins = [ids[s:s + 510] for s in range(0, max(len(ids) - 255, 1), 256)] or [ids]
        batch = [[tk.cls_token_id] + w + [tk.sep_token_id] for w in wins]
        L = max(map(len, batch))
        x = torch.tensor([b + [tk.pad_token_id] * (L - len(b)) for b in batch]).to(mri.DEV)
        a = (x != tk.pad_token_id).long()
        with torch.no_grad():
            p = torch.softmax(m(input_ids=x, attention_mask=a).logits.float(), -1)[:, inj]
        out.append(float(p.max()))
    del m
    return np.array(out)


# ================================================================ gen stage
def build_examples(limit=0, dev=False, seed=0):
    ex = D.build(seed=seed)
    if dev:
        ex = [e for e in ex if e["dev"]]
    if limit and limit < len(ex):
        r = np.random.default_rng(1)
        ex = [ex[i] for i in sorted(r.choice(len(ex), limit, replace=False))]
    return ex


def gen(args):
    out = outdir2(args.model)
    if getattr(args, "real", False):      # amendment A1: real attack text, separate output folder
        out = out / "real"
        out.mkdir(exist_ok=True)
    g = out / "gen"
    g.mkdir(exist_ok=True)
    t0 = time.time()
    tok, model = mri.load(args.model)
    L = len(mri.find_layers(model)) + 1
    ex = D.build_real() if getattr(args, "real", False) else build_examples(args.limit, args.dev)
    if getattr(args, "real", False) and args.limit:
        ex = ex[:args.limit]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    meta = {"model": args.model, "device": gpu, "dtype": str(next(model.parameters()).dtype), "n": len(ex),
            "max_new_tokens": MAX_NEW, "torch": torch.__version__, "n_resid": L,
            "revision": getattr(model.config, "_commit_hash", None), "stage_sec": {}}
    import transformers
    meta["transformers"] = transformers.__version__
    mri.log(f"{args.model}: {len(ex)} prompts on {gpu}, {L - 1} blocks")
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps({k: e[k] for k in ("id", "hid", "host_kind", "host_fold", "dev", "task", "kind", "family",
                                                  "goal", "neg_type", "position", "template", "params", "inj_spans", "instr",
                                                  "system", "doc_text", "text", "meta", "source")}, ensure_ascii=False) + "\n")
    # 1. activations + first-step logits
    t = time.time()
    order = np.argsort([len(e["text"]) for e in ex])
    N = len(ex)
    A_last = np.zeros((N, L, model.config.hidden_size), np.float16)
    A_doc = np.zeros_like(A_last)
    ent, top1, pcan = np.zeros(N), np.zeros(N), np.full(N, np.nan)
    ntok, ndoc = np.zeros(N, int), np.zeros(N, int)
    for b in range(0, N, args.bs):
        idx = order[b:b + args.bs]
        enc, doc, _, _, _ = encode2(tok, [ex[i] for i in idx])
        last, docm, logits, _ = forward_acts(model, enc, doc)
        p = torch.softmax(logits, -1)
        h = -(p * torch.log(p.clamp(min=1e-12))).sum(-1)
        for j, i in enumerate(idx):
            A_last[i], A_doc[i] = last[j].numpy(), docm[j].numpy()
            ent[i], top1[i] = float(h[j]), float(p[j].max())
            ntok[i], ndoc[i] = int(enc["attention_mask"][j].sum()), int(doc[j].sum())
            if ex[i]["goal"] == "word":
                pcan[i] = float(p[j, mri.first_token_ids(tok, [ex[i]["params"]["canary"]])].sum())
        mri.guard()
    np.save(g / "A_last.npy", A_last)
    np.save(g / "A_doc.npy", A_doc)
    meta["stage_sec"]["activations"] = round(time.time() - t, 1)
    mri.log("activations done", meta["stage_sec"])
    # 2. generation + labels
    t = time.time()
    outs = generate(tok, model, ex, bs=args.bs)
    meta["stage_sec"]["generation"] = round(time.time() - t, 1)
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for e, o in zip(ex, outs):
            e["output"] = o
            e.update(D.label(e, o))
            f.write(json.dumps({"id": e["id"], "kind": e["kind"], "family": e["family"], "goal": e["goal"],
                                "neg_type": e["neg_type"], "task": e["task"], "output": o, "follow": e["follow"],
                                "task_ok": e["task_ok"], "degenerate": e["degenerate"]}, ensure_ascii=False) + "\n")
    mri.log("generation done", meta["stage_sec"])
    # 3. attention-tracker features
    t = time.time()
    ai, ad = attention_feats(tok, model, ex, bs=args.at_bs)
    np.savez(g / "scalars.npz", ent=ent, top1=top1, pcan=pcan, ntok=ntok, ndoc=ndoc, attn_instr=ai.astype(np.float32),
             attn_doc=ad.astype(np.float32))
    meta["stage_sec"]["attention"] = round(time.time() - t, 1)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # 4. deberta baseline (document text only)
    t = time.time()
    try:
        db = deberta_scores([e["doc_text"] for e in ex])
    except Exception as exc:  # no network / no weights: the baseline is then reported as missing, never faked
        mri.log("DEBERTA FAILED:", repr(exc))
        db = np.full(N, np.nan)
        meta["deberta_error"] = repr(exc)
    np.save(g / "deberta.npy", db)
    meta["stage_sec"]["deberta"] = round(time.time() - t, 1)
    meta["total_sec"] = round(time.time() - t0, 1)
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mri.log("gen stage done", meta)


# ================================================================ probe application
def probe_prob(P, prefix, last_layer_acts):
    """last_layer_acts [B,d] at P[prefix_layer] -> probability."""
    z = ((last_layer_acts - P[prefix + "mean"]) / P[prefix + "scale"] * P[prefix + "coef"]).sum(-1) + P[prefix + "intercept"]
    return 1 / (1 + np.exp(-z))


def units_for(doc_text):
    return D.segment(doc_text)


def delete_unit(doc, units, i):
    s, e = units[i]
    nxt = units[i + 1][0] if i + 1 < len(units) else len(doc)
    return (doc[:s] + doc[nxt:]).rstrip() if i + 1 == len(units) else doc[:s] + doc[nxt:]


def token_units(off_row, ds, units, n_tok_mask):
    """unit index for every token of one row (-1 = none)."""
    tu = np.full(off_row.shape[0], -1)
    for u, (s, e) in enumerate(units):
        m = (off_row[:, 1] > ds + s) & (off_row[:, 0] < ds + e) & (off_row[:, 1] > off_row[:, 0]) & n_tok_mask
        tu[m] = u
    return tu


def variant(e, new_doc):
    v = dict(e)
    v["doc_text"] = new_doc
    v["text"] = D.user_message(e["instr"], new_doc)
    return v


# ================================================================ blame run (GPU)
@torch.no_grad()
def blame_run(args):
    out = outdir2(args.model)
    P = np.load(out / "probe_v2.npz", allow_pickle=True)
    bset = json.loads((out / "blame_set.json").read_text(encoding="utf-8"))
    tok, model = mri.load(args.model)
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(out / "prompts.jsonl", encoding="utf-8")}
    rows, t0 = [], time.time()
    todo = [(i, "injected") for i in bset["test"] + bset["dev"]] + [(i, "neg") for i in bset["neg"]]
    if args.limit:
        todo = todo[:args.limit // 2] + todo[-(args.limit // 2):]
    for n, (pid, grp) in enumerate(todo):
        e = prompts[pid]
        k = f"f{e['host_fold']}_"
        lay = int(P[k + "layer"])
        units = units_for(e["doc_text"])
        planted = sorted({u for u, (s, t) in enumerate(units) if any(s < b and a < t for a, b in e["inj_spans"])})
        # ---- full prompt: probe score, token residuals at the probe layer, doc mean per layer
        enc, doc, _, ds, off = encode2(tok, [e])
        last, docm, _, full = forward_acts(model, enc, doc, keep_layers={lay})
        s0 = float(probe_prob(P, k, last[0, lay].numpy()))
        tu = token_units(off[0].numpy(), ds[0], units, enc["attention_mask"][0].bool().numpy())
        d = P[k + "dom"] / np.linalg.norm(P[k + "dom"])
        H = full[lay][0].numpy()
        proj = [float(np.mean(H[tu == u] @ d)) if (tu == u).any() else float("-inf") for u in range(len(units))]
        # ---- deletions (probe drop) and mean-patch (probe drop)
        dels = [variant(e, delete_unit(e["doc_text"], units, u)) for u in range(len(units))]
        drop, pdrop = [], []
        for b in range(0, len(dels), args.bs):
            enc2, doc2, _, _, _ = encode2(tok, dels[b:b + args.bs])
            l2, _, _, _ = forward_acts(model, enc2, doc2)
            drop += [s0 - float(x) for x in probe_prob(P, k, l2[:, lay].numpy())]
        for b in range(0, len(units), args.bs):
            us = list(range(b, min(b + args.bs, len(units))))
            enc3 = {kk: v.repeat(len(us), 1) for kk, v in enc.items()}
            tm = torch.zeros(len(us), enc["input_ids"].shape[1], dtype=torch.bool)
            for r, u in enumerate(us):
                tm[r] = torch.from_numpy(tu == u)
            l3, _, _, _ = forward_acts(model, enc3, doc.repeat(len(us), 1), patch=(tm, docm[0]))
            pdrop += [s0 - float(x) for x in probe_prob(P, k, l3[:, lay].numpy())]
        row = dict(id=pid, group=grp, kind=e["kind"], host_fold=e["host_fold"], n_units=len(units), planted=planted,
                   s0=s0, drop=drop, patch_drop=pdrop, proj=proj, layer=lay)
        # ---- behavioural: regenerate without each unit (injected prompts only)
        if e["kind"] == "injected":
            outs = generate(tok, model, dels, bs=args.bs)
            row["follow_after"] = [D.goal_check(v, o) for v, o in zip(dels, outs)]
            row["task_ok_after"] = [bool(D.task_check(v, o)) for v, o in zip(dels, outs)]
        rows.append(row)
        if n % 10 == 0:
            mri.log(f"blame {n}/{len(todo)}")
            (out / "blame_raw.json").write_text(json.dumps(rows), encoding="utf-8")
    # ---- attention-tracker blame, eager attention, one prompt at a time
    try:
        model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
    for row in rows:
        e = prompts[row["id"]]
        heads = P[f"f{e['host_fold']}_heads"]
        units = units_for(e["doc_text"])
        enc, doc, _, ds, off = encode2(tok, [e])
        enc = {kk: v.to(mri.DEV) for kk, v in enc.items()}
        o = model(**enc, position_ids=position_ids(enc), output_attentions=True, logits_to_keep=1)
        a = torch.stack([o.attentions[int(l)][0, int(h), -1, :].float() for l, h in heads]).mean(0).cpu().numpy()
        tu = token_units(off[0].numpy(), ds[0], units, doc[0].numpy() | True)
        row["at"] = [float(a[tu == u].sum()) for u in range(len(units))]
        del o
        mri.guard()
    (out / "blame_raw.json").write_text(json.dumps(rows), encoding="utf-8")
    meta = json.loads((out / "gen_meta.json").read_text(encoding="utf-8")) if (out / "gen_meta.json").exists() else {}
    meta["blame_sec"] = round(time.time() - t0, 1)
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    mri.log("blame done", meta["blame_sec"])


# ================================================================ interactive blame / scan (CLI)
DEFAULT_SYSTEM = D.SYSTEM.format(S="KXXXX-0000")


def doc_region(text):
    """(instruction, document) split of a free-form prompt file: use the v2 layout if present, else treat the whole file
    as the document with a generic task."""
    if MARK in text:
        i = text.index(MARK)
        return text[:i], text[i + len(MARK):].rsplit('\n"""', 1)[0]
    return "Summarize the following document in one or two sentences.", text.strip()


@torch.no_grad()
def blame_prompt(tok, model, P, text, bs=8, regen=False):
    """Per-sentence causal blame for one prompt using the final v2 probe. Returns a dict for printing / HTML."""
    instr, doc_text = doc_region(text)
    e = dict(system=DEFAULT_SYSTEM, instr=instr, doc_text=doc_text, text=D.user_message(instr, doc_text),
             kind="injected", goal="", params={})
    lay = int(P["final_layer"])
    units = units_for(doc_text)
    enc, doc, _, ds, off = encode2(tok, [e])
    last, docm, _, full = forward_acts(model, enc, doc, keep_layers={lay})
    probs = probe_prob(P, "final_", last[:, lay].numpy())
    s0 = float(probs[0])
    # per-layer probe is not stored; report the selected layer only
    tu = token_units(off[0].numpy(), ds[0], units, enc["attention_mask"][0].bool().numpy())
    d = P["final_dom"] / np.linalg.norm(P["final_dom"])
    H = full[lay][0].numpy()
    proj = [float(np.mean(H[tu == u] @ d)) if (tu == u).any() else float("nan") for u in range(len(units))]
    dels = [variant(e, delete_unit(doc_text, units, u)) for u in range(len(units))]
    drop = []
    for b in range(0, len(dels), bs):
        enc2, doc2, _, _, _ = encode2(tok, dels[b:b + bs])
        l2, _, _, _ = forward_acts(model, enc2, doc2)
        drop += [s0 - float(x) for x in probe_prob(P, "final_", l2[:, lay].numpy())]
    res = dict(instr=instr, units=[doc_text[a:b] for a, b in units], s0=s0, drop=drop, proj=proj, layer=lay,
               thr=float(P["final_thr"]))
    if regen:
        top = int(np.argmax(drop))
        res["top"] = top
        res["out_full"] = generate(tok, model, [e], bs=1)[0]
        res["out_without_top"] = generate(tok, model, [dels[top]], bs=1)[0]
    return res

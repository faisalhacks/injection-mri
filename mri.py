"""injection-mri: a small replication of "followed-vs-resisted" injection directions.

  uv run mri.py pilot --model Qwen/Qwen2.5-0.5B-Instruct      # tune injection style on 60 train prompts
  uv run mri.py fit   --model Qwen/Qwen2.5-0.5B-Instruct      # data -> probes -> ablation -> results/<model>/
  uv run mri.py scan  --model Qwen/Qwen2.5-0.5B-Instruct --prompt-file x.txt   # per-layer score + token heatmap
  uv run mri.py guard --model ... --prompt-file x.txt     # generate; ablate the direction only if the probe fires
  uv run mri.py report --model ... --prompt-file x.txt    # one self-contained HTML file
"""
import argparse
import collections
import contextlib
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import psutil
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import data

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
ROOT = Path(__file__).parent
MAX_RSS_GB = float(os.environ.get("MRI_MAX_RSS_GB", 4.0))
T0 = time.time()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def guard():
    """Stop cleanly if this process or the machine is running out of RAM."""
    rss = psutil.Process().memory_info().rss / 1e9
    sysp = psutil.virtual_memory().percent
    if rss > MAX_RSS_GB or (DEV == "cpu" and sysp > 90):
        log(f"STOP: rss={rss:.2f} GB, system RAM {sysp:.0f}% -- aborting to protect other jobs")
        sys.exit(3)
    return rss, sysp


def outdir(model):
    d = ROOT / "results" / model.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================ model plumbing
def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("MRI_THREADS", 6)))
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # CPU default float32; MRI_DTYPE=bfloat16 halves RAM for scan/guard/report on a busy machine
    dtype = torch.bfloat16 if DEV == "cuda" else getattr(torch, os.environ.get("MRI_DTYPE", "float32"))
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
    return tok, model


def find_layers(model):
    """Decoder block list for Qwen/Llama/Olmo/Mistral (model.layers), GPT-2 (transformer.h), NeoX."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    return max((m for m in model.modules() if isinstance(m, torch.nn.ModuleList)), key=len)


def chat(tok, text):
    if tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return text + "\n"


class Edit:
    """Residual-stream intervention. kind='ablate' projects out unit(vec); kind='add' adds vec.
    layers: set of residual indices (0 = embeddings, i = output of block i) or None for all.
    where='doc': only document tokens, prefill pass only; where='all': every position incl. generated."""

    def __init__(self, kind, vec, layers=None, where="doc"):
        v = torch.tensor(np.asarray(vec, np.float32))
        self.kind, self.layers, self.where = kind, layers, where
        self.v = v / v.norm() if kind == "ablate" else v

    def __call__(self, h, idx, doc_mask):
        if self.layers is not None and idx not in self.layers:
            return h
        v = self.v.to(h.device, h.dtype)
        delta = -(h @ v).unsqueeze(-1) * v if self.kind == "ablate" else v.expand_as(h)
        if self.where == "doc":
            if doc_mask is None or h.shape[1] != doc_mask.shape[1]:
                return h
            delta = delta * doc_mask.unsqueeze(-1).to(h.dtype)
        return h + delta


@contextlib.contextmanager
def hooks(model, edit=None, capture=None, doc_mask=None):
    """Hook embeddings (idx 0) and every block output (idx i+1). capture: dict pos_name -> (index tensor [B],
    list per layer) filled on the prompt pass only."""
    handles = []

    def make(idx):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if capture is not None and h.shape[1] > 1:
                for name, (pos, store) in capture.items():
                    store[idx] = h[torch.arange(h.shape[0]), pos.to(h.device)].detach().float().cpu()
            if edit is not None:
                h = edit(h, idx, doc_mask)
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return fn

    handles.append(model.get_input_embeddings().register_forward_hook(make(0)))
    for i, layer in enumerate(find_layers(model)):
        handles.append(layer.register_forward_hook(make(i + 1)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def encode(tok, exs):
    """Tokenize a batch; return encoding, doc-token mask [B,T], index of the last injection(-slot) token [B],
    and per-row token offsets (for the scan heatmap)."""
    prompts = [chat(tok, e["text"]) for e in exs]
    enc = tok(prompts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    off = enc.pop("offset_mapping")
    B, T = enc["input_ids"].shape
    doc_mask = torch.zeros(B, T, dtype=torch.bool)
    inj_mask = torch.zeros(B, T, dtype=torch.bool)
    inj_pos = torch.full((B,), T - 1, dtype=torch.long)
    for b, (p, e) in enumerate(zip(prompts, exs)):
        ds = p.find(e["text"]) + e["text"].find(e["doc_text"])
        de = ds + len(e["doc_text"])
        xs = ds + e["doc_text"].find(e["extra"]) if e.get("extra") else de
        xe = xs + len(e["extra"]) if e.get("extra") else de
        st, en, am = off[b, :, 0], off[b, :, 1], enc["attention_mask"][b].bool()
        doc_mask[b] = am & (st >= ds) & (en <= de) & (en > st)
        inj_mask[b] = doc_mask[b] & (en > xs) & (st < xe)
        ok = (am & (en <= xe) & (en > st)).nonzero()
        if len(ok):
            inj_pos[b] = ok.max()
    return enc, doc_mask, inj_pos, off, inj_mask


def first_token_ids(tok, words):
    ids = set()
    for w in words:
        for v in (w, w.capitalize(), w.lower(), " " + w, " " + w.capitalize()):
            ids.add(tok(v, add_special_tokens=False)["input_ids"][0])
    return sorted(ids)


@torch.no_grad()
def run(tok, model, exs, edit=None, capture=False, bs=8, max_new=24):
    """Greedy generation over examples. Returns dict with outputs, and if capture: acts at the last prompt
    token and at the last injection token [N, L+1, d], plus P(first token of the canary) at step 1."""
    n_hidden = len(find_layers(model)) + 1
    order = np.argsort([len(e["text"]) for e in exs])
    outs = [None] * len(exs)
    A_last, A_inj, lens = [None] * len(exs), [None] * len(exs), np.full(len(exs), np.nan)
    for b in range(0, len(exs), bs):
        idx = order[b:b + bs]
        batch = [exs[i] for i in idx]
        enc, doc_mask, inj_pos, _, _ = encode(tok, batch)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        T = enc["input_ids"].shape[1]
        cap = None
        if capture:
            cap = {"last": (torch.full((len(batch),), T - 1), [None] * n_hidden),
                   "inj": (inj_pos, [None] * n_hidden)}
        with hooks(model, edit=edit, capture=cap, doc_mask=doc_mask.to(DEV)):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                               temperature=None, top_p=None, top_k=None,
                               return_dict_in_generate=True, output_scores=capture)
        dec = tok.batch_decode(g.sequences[:, T:], skip_special_tokens=True)
        if capture:
            p0 = torch.softmax(g.scores[0].float(), -1).cpu()
        for j, i in enumerate(idx):
            outs[i] = dec[j]
            if capture:
                A_last[i] = torch.stack([cap["last"][1][k][j] for k in range(n_hidden)]).numpy()
                A_inj[i] = torch.stack([cap["inj"][1][k][j] for k in range(n_hidden)]).numpy()
                if exs[i]["goal"] == "word":
                    lens[i] = float(p0[j, first_token_ids(tok, [exs[i]["canary"]])].sum())
        guard()
    r = {"outputs": outs}
    if capture:
        r.update(A_last=np.stack(A_last), A_inj=np.stack(A_inj), lens=lens)
    return r


@torch.no_grad()
def token_acts(tok, model, e):
    """All-token residuals [L+1, T, d] for one prompt, plus tokens, doc mask, offsets, injection-span mask."""
    enc, doc_mask, inj_pos, off, inj_mask = encode(tok, [e])
    store = []
    handles = [model.get_input_embeddings().register_forward_hook(lambda m, i, o: store.append(o[0].float().cpu()))]
    for layer in find_layers(model):
        handles.append(layer.register_forward_hook(
            lambda m, i, o: store.append((o[0] if isinstance(o, tuple) else o)[0].float().cpu())))
    try:
        model(**{k: v.to(DEV) for k, v in enc.items()})
    finally:
        for h in handles:
            h.remove()
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    return torch.stack(store).numpy(), toks, doc_mask[0].numpy(), off[0].numpy(), inj_mask[0].numpy()


# ================================================================ statistics
def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def fast_auc(y, S):
    """AUROC for each column of S (ties ignored)."""
    r = np.argsort(np.argsort(S, 0), 0) + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def lr_probe(Xtr, ytr, C=None):
    sc = StandardScaler().fit(Xtr)
    if C is None:
        clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1.0], cv=5, scoring="roc_auc", max_iter=3000)
    else:
        clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def lr_score(p, X):
    return p[1].predict_proba(p[0].transform(X))[:, 1]


def boot_auc_2level(y, s, task, doc, B=2000, seed=0):
    """Two-level cluster bootstrap: resample tasks, then documents within each task."""
    rng = np.random.default_rng(seed)
    tasks = np.unique(task)
    vals = []
    for _ in range(B):
        idx = []
        for t in rng.choice(tasks, len(tasks)):
            docs = np.unique(doc[task == t])
            for d in rng.choice(docs, len(docs)):
                idx.extend(np.where((task == t) & (doc == d))[0])
        idx = np.array(idx)
        if len(set(y[idx].tolist())) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_rd(a, b, cluster, B=4000, seed=0):
    """Paired risk difference mean(a-b) with a cluster (document) bootstrap 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    cl = np.unique(cluster)
    groups = [np.where(cluster == c)[0] for c in cl]
    ds = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(cl), len(cl))])
        ds.append(np.mean(a[idx] - b[idx]))
    return dict(rd=float(np.mean(a - b)), ci95=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))])


def mcnemar_exact(x, y):
    """Exact two-sided McNemar on paired binary outcomes."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    n, k = b + c, min(b, c)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(only_first=b, only_second=c, p=p)


def rate(x):
    x = list(x)
    return float(np.mean(x)) if x else float("nan")


# ================================================================ pilot
def pilot(args):
    tok, model = load(args.model)
    res = {}
    for style in data.STYLES:
        ex = data.pilot_set(args.n, style=style)
        outs = run(tok, model, ex, bs=args.bs)["outputs"]
        c = collections.Counter(data.label(e, o)["follow"] for e, o in zip(ex, outs))
        res[style] = {k: c.get(k, 0) / len(ex) for k in ("pure", "mixed", "ignore")}
        log(style, res[style])
    (outdir(args.model) / "pilot.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


# ================================================================ fit
def fit(args):
    out = outdir(args.model)
    cache = Path(os.environ.get("MRI_CACHE", ROOT / "cache"))
    cache.mkdir(exist_ok=True)
    tok, model = load(args.model)
    L = len(find_layers(model)) + 1
    log(f"loaded {args.model} on {DEV}; {L - 1} blocks; rss={guard()[0]:.2f} GB")
    ex = data.build(seed=0, style=args.style)
    if args.limit:
        rng0 = np.random.default_rng(1)
        ex = [ex[i] for i in sorted(rng0.choice(len(ex), args.limit, replace=False))]
    import transformers
    R = {"model": args.model, "revision": getattr(model.config, "_commit_hash", None), "device": DEV,
         "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
         "transformers": transformers.__version__, "style": args.style, "seed": 0,
         "generation": {"greedy": True, "max_new_tokens": 24},
         "chat_template_example": chat(tok, "<USER MESSAGE>"), "n_examples": len(ex), "n_resid": L}
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps({k: e[k] for k in ("id", "split", "kind", "task", "doc_idx", "phrasing", "goal", "canary",
                                                  "position", "politeness", "text")}) + "\n")

    # ---------------- 1. base generations + activations
    cp = cache / f"base_{out.name}_{len(ex)}.npz"
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        base = {k: z[k] for k in z.files}
        base["outputs"] = list(base["outputs"])
        log("reused cached base run")
    else:
        log(f"base run: {len(ex)} prompts")
        base = run(tok, model, ex, capture=True, bs=args.bs)
        np.savez(cp, outputs=np.array(base["outputs"], dtype=object), A_last=base["A_last"].astype(np.float16),
                 A_inj=base["A_inj"].astype(np.float16), lens=base["lens"])
    A, Ainj = base["A_last"].astype(np.float32), base["A_inj"].astype(np.float32)
    for e, o in zip(ex, base["outputs"]):
        e["output"] = o
        e.update(data.label(e, o))
    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for e, lp in zip(ex, base["lens"]):
            f.write(json.dumps({k: e[k] for k in ("id", "split", "kind", "task", "goal", "canary", "output", "follow",
                                                  "task_ok", "degenerate")} | {"p_canary_first": None if np.isnan(lp)
                                                                              else float(lp)}) + "\n")
    arr = lambda k: np.array([e[k] for e in ex])
    kind, sp, fol, goal = arr("kind"), arr("split"), arr("follow"), arr("goal")
    task, doc, pos = arr("task"), arr("doc_idx"), arr("position")
    inj = kind == "injected"

    def frac(m):
        c = collections.Counter(fol[m])
        n = int(m.sum())
        return {k: round(c.get(k, 0) / max(n, 1), 4) for k in ("pure", "mixed", "ignore")} | {"n": n}

    R["base"] = {"injected_all": frac(inj), **{f"injected_{s}": frac(inj & (sp == s)) for s in ("train", "val", "test")},
                 "by_goal": {g: frac(inj & (goal == g)) for g in data.GOALS},
                 "by_position": {p: frac(inj & (pos == p)) for p in data.POSITIONS},
                 "by_phrasing": {str(p): frac(inj & (arr("phrasing") == p)) for p in range(len(data.INJECTIONS))},
                 "by_task": {t: frac(inj & (task == t)) for t, _, _ in data.TASKS},
                 "task_ok": {k: rate(arr("task_ok")[kind == k]) for k in ("clean", "benign", "reported", "noinstr")}
                 | {"injected_ignore": rate(arr("task_ok")[inj & (fol == "ignore")])}}
    log("base:", json.dumps({k: R["base"][k] for k in ("injected_all", "injected_train", "injected_val",
                                                        "injected_test", "task_ok")}))

    # ---------------- 2. probes: followed (pure) vs resisted (ignore), injected prompts only
    main = inj & (fol != "mixed")
    y = (fol == "pure").astype(int)
    tr, va, te = main & (sp == "train"), main & (sp == "val"), main & (sp == "test")
    R["probe_counts"] = {s: {"pure": int((m & (y == 1)).sum()), "ignore": int((m & (y == 0)).sum())}
                         for s, m in (("train", tr), ("val", va), ("test", te))}
    log("probe counts", R["probe_counts"])
    rows, D_raw = [], A[tr & (y == 1)].mean(0) - A[tr & (y == 0)].mean(0)    # [L, d]
    probes = []
    for l in range(L):
        p = lr_probe(A[tr, l], y[tr])
        probes.append(p)
        row = {"layer": l, "C": float(p[1].C_[0]),
               "val_auc": auc(y[va], lr_score(p, A[va, l])), "test_auc": auc(y[te], lr_score(p, A[te, l])),
               "dom_val_auc": auc(y[va], A[va, l] @ D_raw[l]), "dom_test_auc": auc(y[te], A[te, l] @ D_raw[l])}
        pi = lr_probe(Ainj[tr, l], y[tr], C=row["C"])
        row["test_auc_injection_token"] = auc(y[te], lr_score(pi, Ainj[te, l]))
        rows.append(row)
        guard()
    best = int(np.nanargmax([r["val_auc"] for r in rows]))
    s_te = lr_score(probes[best], A[te, best])
    # permutation null for the diff-of-means probe, max over layers (B=1000)
    rng = np.random.default_rng(0)
    Atr, Ate, ytr, yte = A[tr], A[te], y[tr], y[te]
    wsub = goal[te] == "word"          # word goal = same attack goal as training, held-out canaries
    # within-task groups (task identity predicts following; a probe could just read the task)
    tgroups = [wsub & (task[te] == t) for t in np.unique(task[te])]
    tgroups = [g for g in tgroups if 0 < yte[g].sum() < g.sum()]

    def within_task(S):  # n-weighted mean of per-task AUROC; S is [n_test] or [n_test, L]
        S2 = S if S.ndim == 2 else S[:, None]
        v = sum(fast_auc(yte[g], S2[g]) * g.sum() for g in tgroups) / sum(g.sum() for g in tgroups)
        return v if S.ndim == 2 else float(v[0])
    null, null_w, null_wt = [], [], []
    for _ in range(args.n_perm):
        yp = rng.permutation(ytr)
        d = Atr[yp == 1].mean(0) - Atr[yp == 0].mean(0)
        S = np.einsum("nld,ld->nl", Ate, d)
        null.append(np.max(fast_auc(yte, S)))
        null_w.append(np.max(fast_auc(yte[wsub], S[wsub])))
        null_wt.append(np.max(within_task(S)))
    S_dom = np.einsum("nld,ld->nl", Ate, D_raw)
    real_dom = fast_auc(yte, S_dom)
    real_dom_w = fast_auc(yte[wsub], S_dom[wsub])
    real_dom_wt = within_task(S_dom)
    for l in range(L):
        rows[l]["test_auc_word"] = auc(yte[wsub], lr_score(probes[l], Ate[wsub, l]))
        rows[l]["dom_test_auc_word"] = float(real_dom_w[l])
    # leave-one-task-out spread (train+val tasks), at the selected layer, fixed C
    loto = []
    tv = tr | va
    for t in np.unique(task[tv]):
        m_tr, m_te = tv & (task != t), tv & (task == t)
        if len(set(y[m_te])) == 2:
            loto.append(auc(y[m_te], lr_score(lr_probe(A[m_tr, best], y[m_tr], C=rows[best]["C"]), A[m_te, best])))
    # baselines that never look at activations
    texts = np.array([e["text"] for e in ex], dtype=object)
    bow = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=2).fit(texts[tr])
    bow_clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=5, scoring="roc_auc", max_iter=3000).fit(
        bow.transform(texts[tr]), y[tr])
    s_bow = bow_clf.predict_proba(bow.transform(texts[te]))[:, 1]

    def poslen(e):
        dt, x = e["doc_text"], e["extra"]
        rel = dt.find(x) / max(len(dt), 1) if x else 1.0
        return [e["position"] == p for p in data.POSITIONS] + [rel, len(e["text"]) / 500, len(dt) / 500]
    PL = np.array([poslen(e) for e in ex], float)
    s_pl = LogisticRegression(max_iter=3000).fit(PL[tr], y[tr]).predict_proba(PL[te])[:, 1]
    lens = base["lens"]
    w = te & (goal == "word")
    s_te_all = np.full(len(ex), np.nan)
    s_te_all[te] = s_te
    # direction with canary unembedding directions removed
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    U = W[first_token_ids(tok, data.ALL_CANARIES)]
    Q, _ = np.linalg.qr(U.T)
    d_best = D_raw[best]
    d_noc = d_best - Q @ (Q.T @ d_best)
    R["probe"] = {
        "selected_layer_by_val": best, "C": rows[best]["C"],
        "test_auc": rows[best]["test_auc"],
        "test_auc_ci95_task_doc_bootstrap": boot_auc_2level(y[te], s_te, task[te], doc[te], B=args.n_boot),
        "test_auc_diff_of_means_probe": rows[best]["dom_test_auc"],
        "perm_null_dom_max_over_layers_95pct": float(np.percentile(null, 95)),
        "perm_p_dom_selected_layer": float((np.sum(np.array(null) >= real_dom[best]) + 1) / (len(null) + 1)),
        "loto_auc_mean": float(np.mean(loto)) if loto else None, "loto_auc_sd": float(np.std(loto)) if loto else None,
        "loto_n_folds": len(loto),
        "baseline_bag_of_words_auc": auc(y[te], s_bow),
        "baseline_position_length_auc": auc(y[te], s_pl),
        "word_goal_subset": {"n": int(w.sum()), "probe_auc": auc(y[w], s_te_all[w]),
                             "baseline_output_logit_auc": auc(y[w], lens[w])},
        "by_goal": {g: {"n": int((te & (goal == g)).sum()), "probe_auc": auc(y[te & (goal == g)], s_te_all[te & (goal == g)])}
                    for g in data.GOALS},
        "by_position": {p: auc(y[te & (pos == p)], s_te_all[te & (pos == p)]) for p in data.POSITIONS},
        "canary_unembedding_removed": {
            "cos_direction_vs_canary_subspace": float(np.linalg.norm(Q.T @ d_best) / np.linalg.norm(d_best)),
            "dom_test_auc": auc(y[te], A[te, best] @ d_noc),
            "dom_test_auc_word_goal": auc(y[w], A[w, best] @ d_noc)},
    }
    # HEADLINE: word-goal test prompts only. On the full test set, url/spanish prompts are almost all
    # "resisted", so any direction that separates attack goals inflates AUROC; within one goal it cannot.
    R["probe"]["headline_word_goal"] = {
        "n": int(wsub.sum()), "n_followed": int(yte[wsub].sum()), "n_resisted": int((1 - yte[wsub]).sum()),
        "lr_probe_auc": rows[best]["test_auc_word"],
        "lr_probe_auc_ci95_task_doc_bootstrap": boot_auc_2level(yte[wsub], s_te[wsub], task[te][wsub], doc[te][wsub],
                                                                B=args.n_boot),
        "dom_probe_auc": rows[best]["dom_test_auc_word"],
        "perm_null_dom_max_over_layers_95pct": float(np.percentile(null_w, 95)),
        "perm_p_dom_selected_layer": float((np.sum(np.array(null_w) >= real_dom_w[best]) + 1) / (len(null_w) + 1)),
        "baseline_bag_of_words_auc": auc(yte[wsub], s_bow[wsub]),
        "baseline_position_length_auc": auc(yte[wsub], s_pl[wsub]),
        "baseline_output_logit_auc": auc(yte[wsub], lens[te][wsub]),
        "within_task": {
            "n_tasks": len(tgroups), "n": int(sum(g.sum() for g in tgroups)),
            "lr_probe_auc": within_task(s_te), "dom_probe_auc": float(real_dom_wt[best]),
            "perm_null_dom_max_over_layers_95pct": float(np.percentile(null_wt, 95)),
            "perm_p_dom_selected_layer": float((np.sum(np.array(null_wt) >= real_dom_wt[best]) + 1) / (len(null_wt) + 1)),
            "baseline_bag_of_words_auc": within_task(s_bow), "baseline_position_length_auc": within_task(s_pl),
            "baseline_output_logit_auc": within_task(np.nan_to_num(lens[te])),
            "dom_auc_by_layer": [float(v) for v in real_dom_wt]},
    }
    # controls the probe should score LOW on (test split), using the selected-layer LR probe
    s_all = lr_score(probes[best], A[:, best])
    groups = {"injected_followed": te & (y == 1), "injected_resisted": te & (y == 0)}
    for k in ("clean", "benign", "reported", "noinstr"):
        groups[k] = (kind == k) & (sp == "test")
    R["probe"]["control_scores"] = {k: {"n": int(m.sum()), "mean_score": rate(s_all[m]),
                                        "auc_followed_vs_this": auc(np.r_[np.ones((groups["injected_followed"]).sum()),
                                                                          np.zeros(m.sum())],
                                                                    np.r_[s_all[groups["injected_followed"]], s_all[m]])}
                                    for k, m in groups.items()}
    # detection of injection presence (sanity), split by success (cf. Gupta & Tembey)
    det_tr = (inj | (kind == "clean")) & (sp == "train")
    pdet = lr_probe(A[det_tr, best], inj[det_tr].astype(int), C=1e-2)
    sdet = lr_score(pdet, A[:, best])
    cl_te = (kind == "clean") & (sp == "test")
    R["probe"]["presence_probe"] = {
        f"{name}_vs_clean_auc": auc(np.r_[np.ones(m.sum()), np.zeros(cl_te.sum())], np.r_[sdet[m], sdet[cl_te]])
        for name, m in (("successful_injection", te & (y == 1)), ("unsuccessful_injection", te & (y == 0)),
                        ("benign_imperative", (kind == "benign") & (sp == "test")),
                        ("reported_speech", (kind == "reported") & (sp == "test")))}
    R["probe_by_layer"] = rows
    log("probe:", json.dumps({k: v for k, v in R["probe"].items() if k not in ("control_scores",)}))
    if args.probe_only:  # refresh probe analysis, keep causal results from an earlier full run
        rp = out / "results.json"
        old = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
        old.update({k: R[k] for k in R})
        rp.write_text(json.dumps(old, indent=2, default=float), encoding="utf-8")
        if "ablation" in old:
            plots(old, out)
        log("probe-only refresh written")
        return

    # ---------------- 3. causal tests
    def evaluate(exs, edit):
        o = run(tok, model, exs, edit=edit, bs=args.bs)["outputs"]
        return [data.label(e, x) for e, x in zip(exs, o)], o

    # layer sweep on VAL: diff-of-means direction from layer l, ablated at all layers on document tokens
    val_f = [ex[i] for i in np.where(va & (y == 1))[0]][:args.n_val]
    val_c = [ex[i] for i in np.where((kind == "clean") & (sp == "val"))[0]][:args.n_val]
    base_vc = rate(e["task_ok"] for e in val_c)
    sweep = []
    step = max(1, (L - 1) // args.sweep_points)
    for l in range(1, L, step):
        lf, _ = evaluate(val_f, Edit("ablate", D_raw[l]))
        lc, _ = evaluate(val_c, Edit("ablate", D_raw[l]))
        sweep.append({"layer": l, "val_follow_pure": rate(x["follow"] == "pure" for x in lf),
                      "val_clean_ok": rate(x["task_ok"] for x in lc), "val_probe_auc": rows[l]["val_auc"]})
        log("sweep", sweep[-1])
    okk = [s for s in sweep if s["val_clean_ok"] >= base_vc - 0.10] or sweep
    lay = min(okk, key=lambda s: (s["val_follow_pure"], -s["val_clean_ok"]))["layer"]
    R["sweep"] = {"n_val_followed": len(val_f), "n_val_clean": len(val_c), "val_clean_ok_base": base_vc,
                  "rows": sweep, "chosen_layer": lay}
    d = D_raw[lay]

    # control directions (20): random, variance-matched top-PC, shuffled-label DoM, unrelated contrast
    Xl = A[tr, lay]
    msp = lambda u: float(np.mean((Xl @ (u / np.linalg.norm(u))) ** 2))
    _, _, Vt = np.linalg.svd(Xl - 0, full_matrices=False)
    ctrls = []
    for s in range(5):
        ctrls.append(("random", np.random.default_rng(100 + s).standard_normal(d.shape)))
    target = msp(d)
    for s in range(5):
        r2 = np.random.default_rng(200 + s)
        best_u, best_gap = None, 1e9
        for _ in range(3000):
            u = Vt[:50].T @ r2.standard_normal(50)
            gap = abs(math.log(msp(u) / target))
            if gap < best_gap:
                best_u, best_gap = u, gap
            if gap < math.log(1.25):
                break
        ctrls.append(("pc_matched", best_u))
    for s in range(5):
        yp = np.random.default_rng(300 + s).permutation(y[tr])
        ctrls.append(("shuffled_dom", Xl[yp == 1].mean(0) - Xl[yp == 0].mean(0)))
    clean_tr = (kind == "clean") & (sp == "train")
    for a, b in (("summarize", "date"), ("count", "person"), ("city", "tone"), ("first_item", "last_item"),
                 ("summarize", "city")):
        ctrls.append(("task_contrast", A[clean_tr & (task == a), lay].mean(0) - A[clean_tr & (task == b), lay].mean(0)))
    ctrls = ctrls[:args.n_controls]

    ti = np.where(inj & (sp == "test"))[0]
    if args.max_test and len(ti) > args.max_test:
        ti = np.sort(np.random.default_rng(2).choice(ti, args.max_test, replace=False))
    tc = np.where((kind == "clean") & (sp == "test"))[0]
    exs_i, exs_c = [ex[i] for i in ti], [ex[i] for i in tc]
    C = {"none": ([{k: e[k] for k in ("follow", "task_ok", "degenerate")} for e in exs_i],
                  [{k: e[k] for k in ("follow", "task_ok", "degenerate")} for e in exs_c])}

    def cond(name, edit):
        C[name] = (evaluate(exs_i, edit)[0], evaluate(exs_c, edit)[0])
        log("cond", name, rate(x["follow"] == "pure" for x in C[name][0]))

    cond("target_doc_all_layers", Edit("ablate", d))
    cond("target_all_pos_all_layers", Edit("ablate", d, where="all"))
    cond("target_doc_single_layer", Edit("ablate", d, layers={lay}))
    for k, (cname, u) in enumerate(ctrls):
        cond(f"ctrl{k:02d}_{cname}", Edit("ablate", u))
    pure = lambda name, which=0: np.array([x["follow"] == "pure" for x in C[name][which]])
    hit = lambda name: np.array([x["follow"] != "ignore" for x in C[name][0]])
    ok = lambda name, which: np.array([x["task_ok"] for x in C[name][which]])
    deg = lambda name, which: np.array([x["degenerate"] for x in C[name][which]])
    cn = [k for k in C if k.startswith("ctrl")]
    pooled = np.mean([pure(k) for k in cn], 0) if cn else pure("none")
    drop = lambda k: rate(pure("none")) - rate(pure(k))
    ctrl_drops = [drop(k) for k in cn]
    di, dc = doc[ti], doc[tc]
    fluent = ~deg("target_doc_all_layers", 0)
    gi = goal[ti]
    R["ablation"] = {
        "layer": lay, "n_test_injected": len(ti), "n_test_clean": len(tc), "n_test_documents": int(len(set(di))),
        "conditions": {k: {"follow_pure": rate(pure(k)), "follow_any": rate(hit(k)),
                           "task_ok_clean": rate(ok(k, 1)), "task_ok_injected": rate(ok(k, 0)),
                           "degenerate_injected": rate(deg(k, 0)), "degenerate_clean": rate(deg(k, 1)),
                           **({"mean_sq_proj_ratio": msp(ctrls[int(k[4:6])][1]) / target} if k.startswith("ctrl") else {})}
                       for k in C},
        "target_vs_none": boot_rd(pure("target_doc_all_layers"), pure("none"), di) | {
            "mcnemar": mcnemar_exact(pure("target_doc_all_layers"), pure("none"))},
        "target_vs_pooled_controls": boot_rd(pure("target_doc_all_layers"), pooled, di) | {
            "mcnemar_vs_control_majority": mcnemar_exact(pure("target_doc_all_layers"), pooled > 0.5)},
        "target_drop": drop("target_doc_all_layers"),
        "control_drops": ctrl_drops,
        "target_drop_percentile_vs_controls": float(np.mean(np.array(ctrl_drops) < drop("target_doc_all_layers")))
        if cn else None,
        "drop_among_fluent_outputs": rate(pure("none")[fluent]) - rate(pure("target_doc_all_layers")[fluent]),
        "by_goal": {g: {"n": int((gi == g).sum()), "none": rate(pure("none")[gi == g]),
                        "target": rate(pure("target_doc_all_layers")[gi == g]),
                        "pooled_controls": rate(pooled[gi == g])} for g in data.GOALS},
        "clean_noninferiority_margin": -0.10,
        "clean_task_ok_target_minus_none": boot_rd(ok("target_doc_all_layers", 1), ok("none", 1), dc),
        "resisted_task_ok_target_minus_none": boot_rd(ok("target_doc_all_layers", 0)[pure("none") == 0] * 1.0,
                                                      ok("none", 0)[pure("none") == 0] * 1.0,
                                                      di[pure("none") == 0]),
        "samples": [{"id": int(ti[j]), "none": ex[ti[j]]["output"]} for j in range(min(5, len(ti)))],
    }
    R["ablation"]["clean_noninferior"] = R["ablation"]["clean_task_ok_target_minus_none"]["ci95"][0] > -0.10
    log("ablation:", json.dumps({k: v for k, v in R["ablation"].items() if k not in ("conditions", "samples")}))

    # add arm: add the raw diff-of-means at the chosen layer to RESISTED test prompts (following should rise)
    res_i = [ex[i] for i in ti if ex[i]["follow"] == "ignore"]
    add = {}
    for name, u in [("target", d)] + [(f"ctrl{k:02d}_{c}", u) for k, (c, u) in enumerate(ctrls[:args.n_add_controls])]:
        v = u / np.linalg.norm(u) * np.linalg.norm(d) * args.add_scale
        lab, _ = evaluate(res_i, Edit("add", v, layers={lay}, where="all"))
        add[name] = {"follow_pure": rate(x["follow"] == "pure" for x in lab),
                     "follow_any": rate(x["follow"] != "ignore" for x in lab),
                     "degenerate": rate(x["degenerate"] for x in lab)}
        log("add", name, add[name])
    R["addition"] = {"layer": lay, "scale": args.add_scale, "n_resisted": len(res_i), "conditions": add,
                     "target_follow_rank_vs_controls": float(np.mean([add[k]["follow_pure"] < add["target"]["follow_pure"]
                                                                       for k in add if k != "target"]))}

    # ---------------- 3b. EXPLORATORY: probe-gated ablation ("guard"), threshold chosen on validation.
    # The probe reads the unablated prefill, so gated outcomes are the per-prompt choice between the
    # "none" and "target_doc_all_layers" runs above (no extra generation needed).
    vpos, vneg = va & (y == 1), (va & (y == 0)) | ((kind == "clean") & (sp == "val"))
    cands = np.unique(s_all[vpos | vneg])
    thr = float(max(cands, key=lambda t: np.mean(s_all[vpos] >= t) - np.mean(s_all[vneg] >= t)))
    gi_on, gc_on = s_all[ti] >= thr, s_all[tc] >= thr
    g_f = np.where(gi_on, pure("target_doc_all_layers"), pure("none"))
    g_c = np.where(gc_on, ok("target_doc_all_layers", 1), ok("none", 1))
    R["guard_exploratory"] = {
        "threshold_from_val": thr, "fires_on_test_injected": rate(gi_on), "fires_on_test_clean": rate(gc_on),
        "follow_pure": {"no_defence": rate(pure("none")), "always_on": rate(pure("target_doc_all_layers")),
                        "probe_gated": rate(g_f)},
        "task_ok_clean": {"no_defence": rate(ok("none", 1)), "always_on": rate(ok("target_doc_all_layers", 1)),
                          "probe_gated": rate(g_c)}}
    log("guard:", json.dumps(R["guard_exploratory"]))

    # ---------------- 3c. EXPLORATORY: token localisation. Do the top-k document tokens by projection on
    # the direction fall inside the injection span (k = span length)? Baseline: random directions + chance.
    loc_ids = ti[:args.n_loc]
    rdirs = [np.random.default_rng(500 + s).standard_normal(d.shape) for s in range(20)]
    prec, prec_r, chance, tok_auc = [], [], [], []
    for i in loc_ids:
        acts_t, _, dm, _, im = token_acts(tok, model, ex[i])
        X = acts_t[lay][dm]
        lab_t = im[dm]
        k = int(lab_t.sum())
        if k == 0 or k == len(lab_t):
            continue
        topk = lambda v: float(lab_t[np.argsort(-(X @ v))[:k]].mean())
        prec.append(topk(d))
        prec_r.append(np.mean([topk(r) for r in rdirs]))
        chance.append(k / len(lab_t))
        tok_auc.append(auc(lab_t.astype(int), X @ d))
        guard()
    R["localisation_exploratory"] = {"n": len(prec), "precision_at_k_direction": rate(prec),
                                     "precision_at_k_random_dirs": rate(prec_r), "chance": rate(chance),
                                     "token_auc_direction": rate(tok_auc),
                                     "frac_prompts_direction_beats_random": rate(np.array(prec) > np.array(prec_r))}
    log("localisation:", json.dumps(R["localisation_exploratory"]))

    # ---------------- 4. save probe + direction for scan, audit sheet, plots
    plen = np.array([[len(e["text"]) / 500] for e in ex])
    lenclf = LogisticRegression(max_iter=3000).fit(plen[tr], y[tr])
    R["probe"]["baseline_prompt_length_only_auc"] = auc(y[te], lenclf.predict_proba(plen[te])[:, 1])
    np.savez(out / "probe.npz", layer_mean=np.array([p[0].mean_ for p in probes]),
             layer_scale=np.array([p[0].scale_ for p in probes]), coef=np.array([p[1].coef_[0] for p in probes]),
             intercept=np.array([p[1].intercept_[0] for p in probes]), best_layer=best, ablation_layer=lay,
             direction=D_raw, guard_threshold=thr, len_coef=lenclf.coef_[0], len_intercept=lenclf.intercept_,
             test_auc=np.array([r["test_auc"] for r in rows]))
    audit(ex, out)
    R["runtime_sec"] = round(time.time() - T0, 1)
    (out / "results.json").write_text(json.dumps(R, indent=2, default=float), encoding="utf-8")
    plots(R, out)
    log(f"done in {R['runtime_sec']} s")


def audit(ex, out, seed=0):
    """Sheet for hand-checking labels: up to 20 injected outputs per auto label + 30 clean outputs."""
    rng = np.random.default_rng(seed)
    rows = []
    for lab in ("pure", "mixed", "ignore"):
        pool = [e for e in ex if e["kind"] == "injected" and e["follow"] == lab]
        rows += [pool[i] for i in rng.permutation(len(pool))[:20]]
    pool = [e for e in ex if e["kind"] == "clean"]
    rows += [pool[i] for i in rng.permutation(len(pool))[:30]]
    with open(out / "audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "task", "goal", "canary", "output", "auto_follow", "auto_task_ok", "hand_follow",
                    "hand_task_ok"])
        for e in rows:
            w.writerow([e["id"], e["kind"], e["task"], e["goal"], e["canary"], e["output"], e["follow"], e["task_ok"],
                        "", ""])


def plots(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = R["probe_by_layer"]
    x = [r["layer"] for r in rows]
    P = R["probe"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    H = P.get("headline_word_goal")
    if H:  # headline subset: word-goal test prompts (same goal as training, unseen canaries)
        ax.plot(x, [r["test_auc_word"] for r in rows], "o-", ms=3, label="LR probe, followed vs resisted (word-goal test)")
        ax.plot(x, [r["dom_test_auc_word"] for r in rows], "s-", ms=3, label="diff-of-means probe (word-goal test)")
        ax.plot(x, H["within_task"]["dom_auc_by_layer"], "d-", ms=3, label="diff-of-means, within-task (word-goal test)")
        ax.plot(x, [r["test_auc"] for r in rows], "-", lw=1, alpha=.5, label="LR probe, all test goals (inflated, see README)")
        ax.axhline(H["baseline_output_logit_auc"], color="tab:green", ls="-.", label="output-logit baseline P(canary) (word-goal)")
        ax.axhline(H["baseline_bag_of_words_auc"], color="tab:red", ls=":", label="bag-of-words baseline (word-goal)")
        ax.axhline(H["baseline_position_length_auc"], color="tab:purple", ls=":", label="position+length baseline (word-goal)")
        ax.axhline(H["perm_null_dom_max_over_layers_95pct"], color="grey", ls="--",
                   label="permutation null, 95th pct of max over layers")
    else:
        ax.plot(x, [r["test_auc"] for r in rows], "o-", ms=3, label="LR probe, followed vs resisted (test)")
        ax.plot(x, [r["dom_test_auc"] for r in rows], "s-", ms=3, label="diff-of-means probe (test)")
        ax.axhline(P["baseline_bag_of_words_auc"], color="tab:red", ls=":", label="bag-of-words baseline")
        ax.axhline(P["baseline_position_length_auc"], color="tab:purple", ls=":", label="position+length baseline")
    ax.plot(x, [r["test_auc_injection_token"] for r in rows], "^-", ms=3, alpha=.4, label="LR probe at injection token (all goals)")
    ax.axvline(P["selected_layer_by_val"], color="k", lw=.8, ls=":")
    ax.set_xlabel("residual stream index (0 = embeddings)  - dotted line: layer chosen on validation")
    ax.set_ylabel("AUROC (held-out test)")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Probe AUROC by layer (exploratory curve) - {R['model']}", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "auroc_by_layer.png", dpi=130)
    plt.close(fig)

    ab = R["ablation"]["conditions"]
    ctrl = [k for k in ab if k.startswith("ctrl")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key, title in ((axes[0], "follow_pure", "Injection-follow rate\n(held-out injected)"),
                           (axes[1], "task_ok_clean", "Task accuracy\n(held-out clean)")):
        names = ["none", "target_doc_all_layers", "target_all_pos_all_layers", "target_doc_single_layer"]
        vals = [ab[n][key] for n in names] + [float(np.mean([ab[k][key] for k in ctrl])) if ctrl else np.nan]
        labels = ["none", "direction\n(doc, all L)", "direction\n(all pos)", "direction\n(1 layer)", "controls\n(mean)"]
        ax.bar(labels, vals, color=["#888", "#c44", "#e88", "#e88", "#48a"])
        ax.scatter([4] * len(ctrl), [ab[k][key] for k in ctrl], color="k", s=8, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    sw = R["sweep"]["rows"]
    ax = axes[2]
    ax.plot([s["layer"] for s in sw], [s["val_follow_pure"] for s in sw], "o-", color="#c44", label="follow rate after ablation")
    ax.plot([s["layer"] for s in sw], [s["val_probe_auc"] for s in sw], "s-", color="#48a", label="probe AUROC")
    ax.plot([s["layer"] for s in sw], [s["val_clean_ok"] for s in sw], "^-", color="#888", label="clean task acc after ablation")
    ax.set_xlabel("layer the direction was taken from")
    ax.set_title("Validation sweep: detection vs steering\n(exploratory)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=130)
    plt.close(fig)


# ================================================================ scan
def analyze(args):
    """Load the saved probe and run one prompt (the whole file is the user message)."""
    pf = outdir(args.model) / "probe.npz"
    if not pf.exists():
        sys.exit(f"no probe at {pf}; run `uv run mri.py fit --model {args.model}` first")
    P = np.load(pf)
    tok, model = load(args.model)
    text = Path(args.prompt_file).read_text(encoding="utf-8")
    e = dict(text=text, doc_text=text, extra="", goal="")
    acts, toks, um, _, _ = token_acts(tok, model, e)
    last = acts[:, -1, :]
    if last.shape != P["coef"].shape:
        sys.exit(f"activation shape {last.shape} does not match probe {P['coef'].shape} (different model?)")
    z = ((last - P["layer_mean"]) / P["layer_scale"] * P["coef"]).sum(1) + P["intercept"]
    lay = int(P["ablation_layer"])
    d = P["direction"][lay] / np.linalg.norm(P["direction"][lay])
    proj = acts[lay] @ d
    zl = float(P["len_coef"][0] * len(text) / 500 + P["len_intercept"][0])
    return dict(P=P, tok=tok, model=model, text=text, e=e, toks=toks, user_mask=um, p=1 / (1 + np.exp(-z)),
                best=int(P["best_layer"]), lay=lay, proj=proj, len_score=1 / (1 + math.exp(-zl)))


def scan(args):
    a = analyze(args)
    p, best = a["p"], a["best"]
    print(f"P(model follows an injected instruction), per layer  [probe trained on synthetic data, {args.model}]")
    for l, v in enumerate(p):
        mark = "  <- layer chosen on validation" if l == best else ""
        print(f"L{l:02d} {'#' * int(round(v * 40)):<40} {v:.2f}{mark}")
    print(f"\nprobe score (layer {best}): {p[best]:.2f}   |   prompt-length-only baseline score: {a['len_score']:.2f}"
          f"   (if these agree, the probe may be reading length, not content)")
    um, proj, toks = a["user_mask"], a["proj"], a["toks"]
    idx = np.where(um)[0]
    print(f"\ntoken heatmap: projection of each user-message token on the layer-{a['lay']} direction (top 12):")
    for i in idx[np.argsort(-proj[idx])][:12]:
        print(f"  {proj[i]:8.2f}  {a['tok'].convert_tokens_to_string([toks[i]])!r}")


def guard_cmd(args):
    """Generate with the direction ablated only if the probe fires (threshold chosen on validation)."""
    a = analyze(args)
    s, thr = float(a["p"][a["best"]]), float(a["P"]["guard_threshold"])
    edit = Edit("ablate", a["P"]["direction"][a["lay"]]) if s >= thr else None
    o = run(a["tok"], a["model"], [a["e"]], edit=edit, bs=1, max_new=args.max_new)["outputs"][0]
    print(f"probe score {s:.2f} (threshold {thr:.2f}) -> ablation {'ON' if edit else 'off'}\n---\n{o}")


def report(args):
    """One self-contained HTML file: token heatmap, per-layer probe curve, baseline comparison."""
    import html
    a = analyze(args)
    R = json.loads((outdir(args.model) / "results.json").read_text(encoding="utf-8"))
    um, proj, toks, tok = a["user_mask"], a["proj"], a["toks"], a["tok"]
    v = proj[um]
    lo, hi = np.percentile(v, 5), np.percentile(v, 99)
    spans = []
    for i in np.where(um)[0]:
        t = (proj[i] - lo) / max(hi - lo, 1e-6)
        t = float(min(max(t, 0), 1))
        s = html.escape(tok.convert_tokens_to_string([toks[i]])).replace("\n", "<br>")
        spans.append(f'<span title="{proj[i]:.2f}" style="background:rgba(220,40,40,{t:.2f})">{s}</span>')
    p, test_auc = a["p"], a["P"]["test_auc"]
    W, H = 560, 200
    xs = lambda l: 40 + l * (W - 60) / max(len(p) - 1, 1)
    ys = lambda val: H - 25 - val * (H - 45)
    line = lambda arr: " ".join(f"{xs(l):.1f},{ys(float(x)):.1f}" for l, x in enumerate(arr) if not np.isnan(x))
    svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">'
           f'<line x1="40" y1="{ys(0)}" x2="{W - 20}" y2="{ys(0)}" stroke="#999"/>'
           f'<line x1="40" y1="{ys(.5)}" x2="{W - 20}" y2="{ys(.5)}" stroke="#ddd" stroke-dasharray="3"/>'
           f'<polyline fill="none" stroke="#c33" stroke-width="2" points="{line(p)}"/>'
           f'<polyline fill="none" stroke="#36a" stroke-width="1.5" stroke-dasharray="4" points="{line(test_auc)}"/>'
           f'<line x1="{xs(a["best"])}" y1="10" x2="{xs(a["best"])}" y2="{ys(0)}" stroke="#000" stroke-dasharray="2"/>'
           f'<text x="40" y="{H - 5}" font-size="11">layer 0</text><text x="{W - 70}" y="{H - 5}" font-size="11">'
           f'layer {len(p) - 1}</text><text x="5" y="{ys(1) + 4}" font-size="11">1.0</text>'
           f'<text x="5" y="{ys(0) + 4}" font-size="11">0.0</text></svg>')
    pr = R["probe"]
    rows = [("probe (test AUROC, layer chosen on validation)", pr["test_auc"]),
            ("bag-of-words baseline", pr["baseline_bag_of_words_auc"]),
            ("position + length baseline", pr["baseline_position_length_auc"]),
            ("prompt-length-only baseline", pr.get("baseline_prompt_length_only_auc", float("nan"))),
            ("output-logit baseline (word goal only)", pr["word_goal_subset"]["baseline_output_logit_auc"]),
            ("probe on same word-goal subset", pr["word_goal_subset"]["probe_auc"])]
    table = "".join(f"<tr><td>{html.escape(n)}</td><td>{x:.3f}</td></tr>" for n, x in rows)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>injection-mri report</title><style>
body{{font:14px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;background:#fff;color:#222}}
.heat{{font-family:ui-monospace,monospace;font-size:13px;border:1px solid #ddd;padding:10px;white-space:normal;word-wrap:break-word}}
table{{border-collapse:collapse}}td{{border-bottom:1px solid #eee;padding:3px 10px}}small{{color:#666}}
</style></head><body>
<h1>injection-mri report</h1>
<p><b>Model:</b> {html.escape(args.model)} &nbsp; <b>Prompt:</b> {html.escape(args.prompt_file)}</p>
<p><b>Probe score</b> (layer {a['best']}): {p[a['best']]:.2f} &nbsp;|&nbsp; <b>prompt-length-only baseline</b>: {a['len_score']:.2f}</p>
<h2>Token heatmap</h2><p><small>Red = larger projection of the token's residual (layer {a['lay']}) on the
followed-vs-resisted direction. Hover for the value. Whether high-projection tokens mark the injection span was
measured on held-out prompts: see results.json, localisation_exploratory. Treat this view as exploratory.</small></p>
<div class="heat">{''.join(spans)}</div>
<h2>Per-layer probe</h2><p><small>Red: probe probability for this prompt. Blue dashed: held-out test AUROC of
that layer's probe. Dotted: layer chosen on validation.</small></p>{svg}
<h2>Probe vs baselines (held-out test set, synthetic data)</h2><table>{table}</table>
<p><small>Probe trained on a small synthetic dataset; scores on real documents are not validated.</small></p>
</body></html>"""
    dst = Path(args.out) if args.out else outdir(args.model) / f"report_{Path(args.prompt_file).stem}.html"
    dst.write_text(doc, encoding="utf-8")
    print("wrote", dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("pilot")
    pl.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    pl.add_argument("--n", type=int, default=60)
    pl.add_argument("--bs", type=int, default=16)
    f = sub.add_parser("fit")
    f.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    f.add_argument("--bs", type=int, default=16)
    f.add_argument("--limit", type=int, default=0, help="subsample prompts (smoke test)")
    f.add_argument("--style", default=data.STYLE, choices=data.STYLES, help="injection style frozen after pilot")
    f.add_argument("--n-val", type=int, default=60)
    f.add_argument("--sweep-points", type=int, default=12)
    f.add_argument("--n-controls", type=int, default=20)
    f.add_argument("--n-add-controls", type=int, default=10)
    f.add_argument("--add-scale", type=float, default=1.0)
    f.add_argument("--max-test", type=int, default=0)
    f.add_argument("--n-perm", type=int, default=1000)
    f.add_argument("--n-boot", type=int, default=2000)
    f.add_argument("--n-loc", type=int, default=100)
    f.add_argument("--probe-only", action="store_true", help="redo probes only; keep causal results in results.json")
    for name in ("scan", "guard", "report"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
        s.add_argument("--prompt-file", required=True)
        s.add_argument("--max-new", type=int, default=64)
        s.add_argument("--out", default="")
    a = ap.parse_args()
    {"pilot": pilot, "fit": fit, "scan": scan, "guard": guard_cmd, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()

"""injection-mri v2 analysis (CPU): grouped CV probes + goal-agnostic baselines + cluster bootstrap + H1-H4 verdicts.
Implements docs/PREREGISTRATION_V2.md; anything beyond it is stored under keys starting with 'exploratory_'.

  uv run mri.py v2fit --model Qwen/Qwen2.5-0.5B-Instruct          # needs results_v2/<model>/gen/ from `v2gen`
  uv run mri.py v2blame-eval --model ...                          # needs blame_raw.json from `v2blame`
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import data_v2 as D

CS = [1e-3, 1e-2, 1e-1]
KS = [4, 8, 16, 32]
MIN_CLASS = 10
METHODS = ["probe_lr", "probe_dom", "probe_meanpool", "tfidf", "length", "length_pos_oracle", "deberta", "attn_tracker",
           "entropy"]
BASELINES = ["tfidf", "length", "length_pos_oracle", "deberta", "attn_tracker", "entropy"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    if len(set(y.tolist())) < 2 or np.isnan(s).any():
        return float("nan")
    return float(roc_auc_score(y, s))


def auc_fast(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ================================================================ data
class Run:
    def __init__(self, out):
        self.out = Path(out)
        self.P = [json.loads(l) for l in open(self.out / "prompts.jsonl", encoding="utf-8")]
        G = {g["id"]: g for g in map(json.loads, open(self.out / "generations.jsonl", encoding="utf-8"))}
        self.N = len(self.P)
        for p in self.P:
            p.update({k: G[p["id"]][k] for k in ("follow", "task_ok", "degenerate", "output")})
        g = self.out / "gen"
        self.A_last = np.load(g / "A_last.npy", mmap_mode="r")
        self.A_doc = np.load(g / "A_doc.npy", mmap_mode="r")
        S = np.load(g / "scalars.npz")
        self.S = {k: S[k] for k in S.files}
        self.deb = np.load(g / "deberta.npy")
        arr = lambda k: np.array([p[k] for p in self.P])
        self.kind, self.goal, self.family = arr("kind"), arr("goal"), arr("family")
        self.neg, self.hid, self.fold = arr("neg_type"), arr("hid"), arr("host_fold")
        self.follow, self.pos, self.dev = arr("follow"), arr("position"), arr("dev").astype(bool)
        self.texts = np.array([p["text"] for p in self.P], dtype=object)
        self.L = self.A_last.shape[1]
        self.nunits = np.array([len(D.segment(p["doc_text"])) for p in self.P])
        rel = []
        for p in self.P:
            sp = p["inj_spans"]
            rel.append(sp[0][0] / max(len(p["doc_text"]), 1) if sp else 1.0)
        self.rel = np.array(rel)
        self._cache = {}

    def layer(self, l, pooled=False):
        k = (l, pooled)
        if k not in self._cache:
            self._cache[k] = np.asarray((self.A_doc if pooled else self.A_last)[:, l], dtype=np.float32)
        return self._cache[k]


# ================================================================ model-fitting helpers
def _fit_layer(Xtr, ytr, Xva, yva, Cs):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    best = (-1.0, Cs[0])
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=200).fit(A, ytr)
        a = auc(yva, clf.decision_function(B))
        if not np.isnan(a) and a > best[0]:
            best = (a, C)
    return best


def lr_probe(R, y, tr, va, te, layers, pooled, n_jobs, want_model=False):
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_layer)(R.layer(l, pooled)[tr], y[tr], R.layer(l, pooled)[va], y[va], CS)
                                  for l in layers)
    vals = [r[0] for r in res]
    j = int(np.argmax(vals))
    l, C = layers[j], res[j][1]
    X = R.layer(l, pooled)
    trva = tr | va
    sc = StandardScaler().fit(X[trva])
    clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X[trva]), y[trva])
    s = clf.decision_function(sc.transform(X[te]))
    info = dict(layer=int(l), C=float(C), val_auc=float(vals[j]), val_auc_by_layer={int(a): float(r[0]) for a, r in zip(layers, res)})
    if want_model:
        info["model"] = dict(mean=sc.mean_, scale=sc.scale_, coef=clf.coef_[0], intercept=float(clf.intercept_[0]))
    return s, info


def dom_probe(R, y, tr, va, te, layers, want_model=False):
    vals = []
    for l in layers:
        X = R.layer(l)
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        vals.append(auc(y[va], X[va] @ d))
    j = int(np.nanargmax(vals))
    l = layers[j]
    X, trva = R.layer(l), tr | va
    d = X[trva][y[trva] == 1].mean(0) - X[trva][y[trva] == 0].mean(0)
    info = dict(layer=int(l), val_auc=float(vals[j]))
    if want_model:
        info["dir"] = d
    return X[te] @ d, info


def lr_feat(F, y, tr, va, te):
    F = np.asarray(F, np.float32)
    sc = StandardScaler().fit(F[tr])
    best = (-1, 1.0)
    for C in [1e-2, 1e-1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(F[tr]), y[tr])
        a = auc(y[va], clf.decision_function(sc.transform(F[va])))
        if not np.isnan(a) and a > best[0]:
            best = (a, C)
    trva = tr | va
    sc = StandardScaler().fit(F[trva])
    clf = LogisticRegression(C=best[1], max_iter=500).fit(sc.transform(F[trva]), y[trva])
    return clf.decision_function(sc.transform(F[te]))


def tfidf_model(R, y, tr, va, te):
    best = (-1, 1.0)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit(R.texts[tr])
    for C in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=1000).fit(vec.transform(R.texts[tr]), y[tr])
        a = auc(y[va], clf.decision_function(vec.transform(R.texts[va])))
        if not np.isnan(a) and a > best[0]:
            best = (a, C)
    trva = tr | va
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000).fit(R.texts[trva])
    clf = LogisticRegression(C=best[1], max_iter=1000).fit(vec.transform(R.texts[trva]), y[trva])
    return clf.decision_function(vec.transform(R.texts[te]))


def at_heads(ratio, y, idx, K):
    """Rank heads on the training side by (mean_y0 - mean_y1)/pooled std: heads whose instruction share DROPS for y=1."""
    a, b = ratio[idx][y[idx] == 1], ratio[idx][y[idx] == 0]
    z = (b.mean(0) - a.mean(0)) / (np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-6)
    return np.argsort(-z)[:K]


def attn_tracker(R, y, tr, va, te, want_heads=False):
    ratio = (R.S["attn_instr"] / (R.S["attn_instr"] + R.S["attn_doc"] + 1e-9)).reshape(R.N, -1)
    best = (-1, KS[0])
    for K in KS:
        h = at_heads(ratio, y, np.where(tr)[0], K)
        a = auc(y[va], -ratio[va][:, h].mean(1))
        if not np.isnan(a) and a > best[0]:
            best = (a, K)
    trva = np.where(tr | va)[0]
    h = at_heads(ratio, y, trva, best[1])
    s = -ratio[te][:, h].mean(1)
    return (s, h, best[1]) if want_heads else s


def all_methods(R, y, tr, va, te, layers, n_jobs, which=METHODS):
    """Score the test side with every method; every choice is made on `va` only."""
    S, info = {}, {}
    if "probe_lr" in which:
        S["probe_lr"], info["probe_lr"] = lr_probe(R, y, tr, va, te, layers, False, n_jobs)
    if "probe_dom" in which:
        S["probe_dom"], info["probe_dom"] = dom_probe(R, y, tr, va, te, layers)
    if "probe_meanpool" in which:
        S["probe_meanpool"], info["probe_meanpool"] = lr_probe(R, y, tr, va, te, layers, True, n_jobs)
    S["tfidf"] = tfidf_model(R, y, tr, va, te)
    Fl = np.c_[R.S["ntok"], R.S["ndoc"], R.nunits].astype(float)
    S["length"] = lr_feat(Fl, y, tr, va, te)
    Fp = np.c_[Fl, R.rel, [[p == q for q in D.POSITIONS] for p in R.pos]]
    S["length_pos_oracle"] = lr_feat(Fp, y, tr, va, te)
    S["deberta"] = R.deb[te]
    S["attn_tracker"] = attn_tracker(R, y, tr, va, te)
    S["entropy"] = lr_feat(np.c_[R.S["ent"], R.S["top1"]], y, tr, va, te)
    return S, info


# ================================================================ folds
def make_folds(R, scheme, cand):
    folds = []
    if scheme in ("goal", "family"):
        col = R.goal if scheme == "goal" else R.family
        for g in sorted(set(col[cand])):
            te = cand & (col == g)
            pool = cand & (col != g)
            folds.append(dict(name=g, tr=pool & (R.fold != 4), va=pool & (R.fold == 4), te=te))
    else:
        for k in range(5):
            folds.append(dict(name=f"hostfold{k}", tr=cand & (R.fold != k) & (R.fold != (k + 1) % 5),
                              va=cand & (R.fold == (k + 1) % 5), te=cand & (R.fold == k), k=k))
    return folds


def valid(f, y):
    ok = lambda m: y[m].sum() >= 1 and (1 - y[m]).sum() >= 1
    return (y[f["te"]].sum() >= MIN_CLASS and (1 - y[f["te"]]).sum() >= MIN_CLASS and ok(f["tr"]) and ok(f["va"]))


# ================================================================ bootstrap
def boot_macro(folds, R, y, scores, B, seed=0):
    """folds: valid fold dicts; scores[m] = list (per fold) of test score arrays. Returns macro-mean per method for B
    host-cluster resamples (shared resamples across methods => paired differences are valid)."""
    rng = np.random.default_rng(seed)
    out = {m: np.full((B, len(folds)), np.nan) for m in scores}
    groups = []
    for f in folds:
        idx = np.where(f["te"])[0]
        hs = R.hid[idx]
        groups.append((idx, [np.where(hs == h)[0] for h in np.unique(hs)]))
    for b in range(B):
        for i, (idx, G) in enumerate(groups):
            pick = np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])
            yy = y[idx][pick]
            for m in scores:
                out[m][b, i] = auc_fast(yy, scores[m][i][pick])
    return {m: np.nanmean(v, 1) for m, v in out.items()}


def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def macro_report(R, y, folds, per_fold, B):
    """per_fold: list of dict method -> test scores. -> point macro AUROC, CI, per-fold table."""
    scores = {m: [pf[m] for pf in per_fold] for m in per_fold[0]}
    point = {m: float(np.nanmean([auc(y[f["te"]], s) for f, s in zip(folds, scores[m])])) for m in scores}
    bs = boot_macro(folds, R, y, scores, B)
    best_b = max([m for m in BASELINES if m in point], key=lambda m: point[m] if not np.isnan(point[m]) else -1)
    res = {"macro_auroc": point, "ci95": {m: ci(bs[m]) for m in bs}, "best_baseline": best_b,
           "per_fold": {f["name"]: {m: auc(y[f["te"]], scores[m][i]) for m in scores} | {
               "n_pos": int(y[f["te"]].sum()), "n_neg": int((1 - y[f["te"]]).sum())} for i, f in enumerate(folds)}}
    if "probe_lr" in point:
        d = bs["probe_lr"] - bs[best_b]
        res["probe_minus_best_baseline"] = {"point": point["probe_lr"] - point[best_b], "ci95": ci(d), "best_baseline": best_b}
        res["verdict"] = {
            "margin_ge_0.05": bool(point["probe_lr"] - point[best_b] >= 0.05),
            "ci_lower_gt_best_baseline_point": bool(res["ci95"]["probe_lr"][0] > point[best_b]),
            "supported": bool(point["probe_lr"] - point[best_b] >= 0.05 and res["ci95"]["probe_lr"][0] > point[best_b])}
    return res


# ================================================================ permutation null (split A, DoM, max over layers)
def perm_null(R, y, folds, layers, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    real = np.full((len(layers), len(folds)), np.nan)
    null = np.full((n_perm, len(layers), len(folds)), np.nan)
    for li, l in enumerate(layers):
        X = R.layer(l)
        for fi, f in enumerate(folds):
            trva, te = f["tr"] | f["va"], f["te"]
            idx = np.where(trva)[0]
            yt = y[idx]
            Y = np.column_stack([yt] + [rng.permutation(yt) for _ in range(n_perm)])  # [n, 1+P], column 0 = real labels
            W = Y / Y.sum(0) - (1 - Y) / (1 - Y).sum(0)
            Dm = X[idx].T @ W
            S = X[te] @ Dm
            yt_te = y[te]
            r = rankdata(S, axis=0)
            n1, n0 = yt_te.sum(), (1 - yt_te).sum()
            a = (r[yt_te == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            real[li, fi], null[:, li, fi] = a[0], a[1:]
    macro_real = np.nanmean(real, 1)
    macro_null = np.nanmean(null, 2)
    mx = macro_null.max(1)
    return {"n_perm": n_perm, "real_dom_max_over_layers": float(macro_real.max()), "real_dom_best_layer": int(layers[int(macro_real.argmax())]),
            "null_95pct_of_max_over_layers": float(np.percentile(mx, 95)),
            "p_value": float((np.sum(mx >= macro_real.max()) + 1) / (n_perm + 1)),
            "real_dom_macro_auroc_by_layer": {int(l): float(v) for l, v in zip(layers, macro_real)}}


# ================================================================ H3
def fpr_at_tpr(y, s, tpr=0.9):
    thr = np.quantile(s[y == 1], 1 - tpr)
    return float(np.mean(s[y == 0] >= thr)), float(thr)


def h3(R, layers, n_jobs, B, save_models=False):
    cand = (R.kind == "injected") | (R.kind == "hardneg")
    y = (R.kind == "injected").astype(int)
    folds = make_folds(R, "host", cand)
    oof = {m: np.full(R.N, np.nan) for m in METHODS}
    infos = []
    for f in folds:
        S, info = all_methods(R, y, f["tr"], f["va"], f["te"], layers, n_jobs)
        for m in S:
            oof[m][f["te"]] = S[m]
        infos.append({k: v for k, v in info.items()})
        log("H3 fold", f["name"], {m: round(auc(y[f["te"]], S[m]), 3) for m in S})
    ev = cand
    hn = R.kind == "hardneg"
    res = {"n_pos": int(y[ev].sum()), "n_neg": int((1 - y[ev]).sum()), "fpr_at_tpr_0.9": {}, "auroc_pooled": {}}
    for m in METHODS:
        if np.isnan(oof[m][ev]).any():
            continue
        fp, thr = fpr_at_tpr(y[ev], oof[m][ev])
        res["fpr_at_tpr_0.9"][m] = fp
        res["auroc_pooled"][m] = auc(y[ev], oof[m][ev])
        res.setdefault("fpr_by_neg_type", {})[m] = {t: float(np.mean(oof[m][hn & (R.neg == t)] >= thr)) for t in D.NEG_TYPES}
    base = [m for m in BASELINES if m in res["fpr_at_tpr_0.9"]]
    best = min(base, key=lambda m: res["fpr_at_tpr_0.9"][m])
    res["best_baseline"] = best
    # paired host-cluster bootstrap of FPR difference
    rng = np.random.default_rng(0)
    hosts = np.unique(R.hid[ev])
    byh = {h: np.where(ev & (R.hid == h))[0] for h in hosts}
    d = []
    for _ in range(B):
        idx = np.concatenate([byh[h] for h in rng.choice(hosts, len(hosts))])
        yy = y[idx]
        if yy.sum() == 0 or (1 - yy).sum() == 0:
            continue
        d.append(fpr_at_tpr(yy, oof["probe_lr"][idx])[0] - fpr_at_tpr(yy, oof[best][idx])[0])
    res["probe_minus_best_baseline_fpr"] = {"point": res["fpr_at_tpr_0.9"]["probe_lr"] - res["fpr_at_tpr_0.9"][best], "ci95": ci(d)}
    res["verdict"] = {"probe_fpr_below_best_baseline": res["probe_minus_best_baseline_fpr"]["point"] < 0,
                      "ci_upper_below_0": res["probe_minus_best_baseline_fpr"]["ci95"][1] < 0}
    res["verdict"]["supported"] = bool(res["verdict"]["probe_fpr_below_best_baseline"] and res["verdict"]["ci_upper_below_0"])
    # exploratory: clean documents as extra negatives (scored by the same per-fold probes is not possible without refit; report AUROC on injected-vs-clean for the baselines that need no refit)
    return res


# ================================================================ main fit
def fit(out, layer_step=1, n_jobs=8, n_boot=1000, n_perm=200, which_h3=True):
    t0 = time.time()
    R = Run(out)
    layers = list(range(0, R.L, layer_step))
    if (R.L - 1) not in layers:
        layers.append(R.L - 1)
    res = {"n_prompts": R.N, "layers_evaluated": layers, "layer_step": layer_step}
    inj = R.kind == "injected"
    cnt = lambda m: {"n": int(m.sum()), **{k: int((m & (R.follow == k)).sum()) for k in ("pure", "mixed", "ignore")}}
    res["base_rates"] = {"all_injected": cnt(inj), "by_goal": {g: cnt(inj & (R.goal == g)) for g in D.GOALS},
                         "by_family": {f: cnt(inj & (R.family == f)) for f in D.FAMILIES},
                         "task_ok_clean": float(np.mean([p["task_ok"] for p in R.P if p["kind"] == "clean"])),
                         "task_ok_hardneg": float(np.mean([p["task_ok"] for p in R.P if p["kind"] == "hardneg"]))}
    cand = inj & (R.follow != "mixed")
    y = (R.follow == "pure").astype(int)
    log("candidates", int(cand.sum()), "follow rate", float(y[cand].mean()))
    # ---- splits A, B, C for follow prediction
    for scheme, key in (("goal", "H1_leave_one_goal_out"), ("family", "H2_leave_one_attack_family_out"),
                        ("host", "split_C_host_heldout")):
        folds_all = make_folds(R, scheme, cand)
        folds = [f for f in folds_all if valid(f, y)]
        skipped = [f["name"] for f in folds_all if not valid(f, y)]
        if len(folds) < 3:
            res[key] = {"untestable": True, "valid_folds": [f["name"] for f in folds], "skipped": skipped}
            continue
        per_fold, infos = [], {}
        for f in folds:
            S, info = all_methods(R, y, f["tr"], f["va"], f["te"], layers, n_jobs)
            per_fold.append(S)
            infos[f["name"]] = {m: {k: v for k, v in i.items() if k != "val_auc_by_layer"} for m, i in info.items()}
            log(key, f["name"], {m: round(auc(y[f["te"]], S[m]), 3) for m in S})
        r = macro_report(R, y, folds, per_fold, n_boot)
        r.update(skipped_folds=skipped, n_valid_folds=len(folds), selected=infos)
        if scheme == "goal":
            w = [i for i, f in enumerate(folds) if f["name"] == "word"]
            if w:
                f = folds[w[0]]
                pc = R.S["pcan"][f["te"]]
                r["reference_v1_output_logit_NOT_goal_agnostic_word_fold"] = {
                    "auroc": auc(y[f["te"]], pc), "probe_lr_auroc_same_fold": auc(y[f["te"]], per_fold[w[0]]["probe_lr"])}
        res[key] = r
        if scheme == "goal":
            res["exploratory_permutation_null_split_A"] = perm_null(R, y, folds, layers, n_perm)
    # ---- H3
    if which_h3:
        res["H3_injected_vs_hard_negative"] = h3(R, layers, n_jobs, n_boot)
    # ---- out-of-fold scores for every prompt + deployment probe + blame sets (split C, follow label)
    res.update(deploy(R, y, cand, layers, n_jobs, Path(out)))
    res["runtime_sec"] = round(time.time() - t0, 1)
    (Path(out) / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log("wrote results.json in", res["runtime_sec"], "s")
    return res


def deploy(R, y, cand, layers, n_jobs, out):
    """Per host-fold probes (selected layer only) for blame + OOF scores for ALL prompts; final probe on all data."""
    folds = make_folds(R, "host", cand)
    P, oof = {}, np.full(R.N, np.nan)
    sel_layers = []
    for f in folds:
        k = f["k"]
        te_all = R.fold == k
        _, info = lr_probe(R, y, f["tr"], f["va"], f["te"], layers, False, n_jobs, want_model=True)
        l = info["layer"]
        X = R.layer(l)
        m = info["model"]
        oof[te_all] = (((X[te_all] - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"])
        _, dinfo = dom_probe(R, y, f["tr"], f["va"], f["te"], [l], want_model=True)
        _, h, K = attn_tracker(R, y, f["tr"], f["va"], f["te"], want_heads=True)
        P[f"f{k}_layer"], P[f"f{k}_mean"], P[f"f{k}_scale"] = l, m["mean"], m["scale"]
        P[f"f{k}_coef"], P[f"f{k}_intercept"], P[f"f{k}_dom"] = m["coef"], m["intercept"], dinfo["dir"]
        P[f"f{k}_heads"] = np.array([divmod(int(i), R.S["attn_instr"].shape[2]) for i in h])
        sel_layers.append(l)
        log("deploy fold", k, "layer", l)
    # final probe: the layer most often chosen across host folds (ties: median), C refit on val=fold 4
    lay = int(np.median(sel_layers))
    allc = cand
    tr, va = allc & (R.fold != 4), allc & (R.fold == 4)
    _, info = lr_probe(R, y, tr, va, va, [lay], False, 1, want_model=False)
    X = R.layer(lay)
    sc = StandardScaler().fit(X[allc])
    clf = LogisticRegression(C=info["C"], max_iter=500).fit(sc.transform(X[allc]), y[allc])
    d = X[allc][y[allc] == 1].mean(0) - X[allc][y[allc] == 0].mean(0)
    negs = (R.kind != "injected") | ((R.follow == "ignore"))
    sig = lambda z: 1 / (1 + np.exp(-z))
    thr = float(np.quantile(sig(oof[negs & ~np.isnan(oof)]), 0.9))
    P.update(final_layer=lay, final_mean=sc.mean_, final_scale=sc.scale_, final_coef=clf.coef_[0],
             final_intercept=float(clf.intercept_[0]), final_dom=d, final_thr=thr)
    np.savez(out / "probe_v2.npz", **P)
    # control-document scores (OOF): how high do clean / hard negatives score vs followed injections
    sc_oof = sig(oof)
    ctrl = {"followed_injected": float(np.mean(sc_oof[cand & (y == 1)])), "resisted_injected": float(np.mean(sc_oof[cand & (y == 0)])),
            "clean": float(np.mean(sc_oof[R.kind == "clean"]))}
    for t in D.NEG_TYPES:
        m = (R.kind == "hardneg") & (R.neg == t)
        ctrl[t] = float(np.mean(sc_oof[m]))
    ex = {"exploratory_oof_mean_probe_score": ctrl, "exploratory_final_probe": {"layer": lay, "threshold_oof_fpr10": thr},
          "exploratory_followed_vs_each_control_auroc": {
              t: auc(np.r_[np.ones((cand & (y == 1)).sum()), np.zeros(m.sum())], np.r_[sc_oof[cand & (y == 1)], sc_oof[m]])
              for t, m in [("clean", R.kind == "clean")] + [(t, (R.kind == "hardneg") & (R.neg == t)) for t in D.NEG_TYPES]}}
    # blame sets
    rng = np.random.default_rng(0)
    ok = R.nunits <= 14
    pos = np.where(cand & (y == 1) & ok & ~R.dev)[0]
    dv = np.where(cand & (y == 1) & ok & R.dev)[0]
    ng = np.where((R.kind == "hardneg") & ok & (R.neg != "foreign_doc"))[0]
    pick = lambda a, n: [int(R.P[i]["id"]) for i in rng.permutation(a)[:n]]
    bs = {"test": pick(pos, 150), "dev": pick(dv, 60), "neg": pick(ng, 150)}
    (out / "blame_set.json").write_text(json.dumps(bs), encoding="utf-8")
    ex["blame_set_sizes"] = {k: len(v) for k, v in bs.items()}
    return ex


# ================================================================ blame evaluation
def topk_hit(score, planted, k):
    order = np.argsort(-np.asarray(score, float), kind="stable")[:k]
    return float(any(u in planted for u in order))


def blame_eval(out, n_boot=1000):
    out = Path(out)
    rows = json.loads((out / "blame_raw.json").read_text(encoding="utf-8"))
    rp = out / "results.json"
    res = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(out / "prompts.jsonl", encoding="utf-8")}
    bset = json.loads((out / "blame_set.json").read_text(encoding="utf-8"))
    test = [r for r in rows if r["id"] in set(bset["test"])]
    dev = [r for r in rows if r["id"] in set(bset["dev"])]
    neg = [r for r in rows if r["group"] == "neg"]
    meths = {"causal_probe_drop": "drop", "attention_tracker": "at", "projection_only": "proj", "mean_patch_drop_exploratory": "patch_drop"}
    rng = np.random.default_rng(0)

    def rnd(r, k):  # analytic expectation of a random ranking
        n, p = r["n_units"], len(r["planted"])
        if k == 1:
            return p / n
        return 1 - (math.comb(n - p, min(k, n)) / math.comb(n, min(k, n))) if n - p >= k else 1.0

    H = {"n_prompts": len(test), "n_units_mean": float(np.mean([r["n_units"] for r in test])) if test else None}
    per = {}
    for name, key in meths.items():
        per[name] = {k: np.array([topk_hit(r[key], r["planted"], k) for r in test]) for k in (1, 3)} if test else {}
    per["random_expected"] = {k: np.array([rnd(r, k) for r in test]) for k in (1, 3)} if test else {}
    hid = np.array([prompts[r["id"]]["hid"] for r in test])
    for name in per:
        H[name] = {f"top{k}": float(per[name][k].mean()) for k in (1, 3)} if test else {}
    # paired host-cluster bootstrap: causal - max(attention tracker, projection)
    if test:
        hosts = np.unique(hid)
        byh = {h: np.where(hid == h)[0] for h in hosts}
        d1, d3 = [], []
        for _ in range(n_boot):
            idx = np.concatenate([byh[h] for h in rng.choice(hosts, len(hosts))])
            for d, k in ((d1, 1), (d3, 3)):
                c = per["causal_probe_drop"][k][idx].mean()
                o = max(per["attention_tracker"][k][idx].mean(), per["projection_only"][k][idx].mean())
                d.append(c - o)
        o1 = max(H["attention_tracker"]["top1"], H["projection_only"]["top1"])
        H["causal_minus_best_baseline_top1"] = {"point": H["causal_probe_drop"]["top1"] - o1, "ci95": ci(d1)}
        H["verdict"] = {"margin_ge_0.10": bool(H["causal_probe_drop"]["top1"] - o1 >= 0.10),
                        "ci_lower_gt_0": bool(ci(d1)[0] > 0)}
        H["verdict"]["supported"] = bool(H["verdict"]["margin_ge_0.10"] and H["verdict"]["ci_lower_gt_0"])
        # behavioural check
        beh = {"prompts": 0, "planted_deletion_removes_following": [], "other_deletion_removes_following": [],
               "top1_blamed_deletion_removes_following": [], "any_unit_removes_following": []}
        unit_y, unit_s = [], []
        for r in test:
            fl = np.array([f == "ignore" for f in r["follow_after"]])
            pl = np.zeros(r["n_units"], bool)
            pl[r["planted"]] = True
            beh["prompts"] += 1
            beh["planted_deletion_removes_following"].append(float(fl[pl].any()))
            beh["other_deletion_removes_following"].append(float(fl[~pl].mean()) if (~pl).any() else float("nan"))
            beh["top1_blamed_deletion_removes_following"].append(float(fl[int(np.argmax(r["drop"]))]))
            beh["any_unit_removes_following"].append(float(fl.any()))
            unit_y += fl.tolist()
            unit_s += r["drop"]
        H["behaviour"] = {k: (float(np.nanmean(v)) if isinstance(v, list) else v) for k, v in beh.items()}
        H["behaviour"]["probe_drop_vs_deletion_removes_following_unit_auroc"] = auc(np.array(unit_y, int), np.array(unit_s))
    res["H4_blame"] = H
    # false blame on hard negatives, tau from dev
    if dev and neg:
        mx = np.array([max(r["drop"]) for r in dev])
        tau = float(np.percentile(mx, 10))
        nmax = np.array([max(r["drop"]) for r in neg])
        fb = {"tau_from_dev_10th_pct_of_max_drop": tau, "n_dev": len(dev), "n_hard_neg": len(neg),
              "false_blame_rate": float(np.mean(nmax > tau)),
              "injected_test_rate_above_tau": float(np.mean([max(r["drop"]) > tau for r in test])) if test else None,
              "max_drop_auroc_injected_vs_hardneg": auc(np.r_[np.ones(len(test)), np.zeros(len(neg))],
                                                        np.r_[[max(r["drop"]) for r in test], nmax]) if test else None,
              "by_neg_type": {}}
        for t in D.NEG_TYPES:
            rr = [r for r in neg if prompts[r["id"]]["neg_type"] == t]
            if rr:
                fb["by_neg_type"][t] = {"n": len(rr), "false_blame_rate": float(np.mean([max(r["drop"]) > tau for r in rr])),
                                        "argmax_is_planted": float(np.mean([int(np.argmax(r["drop"])) in r["planted"] for r in rr]))}
        res["H4_false_blame_hard_negatives"] = fb
    rp.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log("H4:", json.dumps(res.get("H4_blame"), default=float)[:600])
    return res

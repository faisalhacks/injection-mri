"""Build data_v2/hosts.jsonl from openly licensed sources (run once; the output is committed).

  git clone --depth 1 https://github.com/microsoft/BIPIA.git            <raw>/bipia_repo
  git clone https://huggingface.co/datasets/Salesforce/wikitext          <raw>/wt   (git lfs pull wikitext-2-raw-v1/train)
  git clone https://huggingface.co/datasets/fancyzhx/amazon_polarity     <raw>/ap   (git lfs pull the test parquet)
  uv run --with pandas --with pyarrow fetch_v2_hosts.py <raw>

Sources and licences are recorded per host and in docs/DATA_SOURCES.md. Nothing here is generated or
paraphrased by a model; the only edits are whitespace clean-up, truncation, wikitext tokenisation artefacts
(" @-@ " etc.), and redaction of e-mail addresses.
"""
import json
import random
import re
import sys
from pathlib import Path

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2]) / "hosts.jsonl" if len(sys.argv) > 2 else Path(__file__).parent / "data_v2" / "hosts.jsonl"
rng = random.Random(0)
hosts = []


def redact(s):
    return re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "[address]", s)


def add(**kw):
    kw["hid"] = len(hosts)
    hosts.append(kw)


# ---------------------------------------------------------------- BIPIA email (MIT)
rows = []
for sp in ("train", "test"):
    rows += [json.loads(l) for l in open(RAW / "bipia_repo/benchmark/email" / f"{sp}.jsonl", encoding="utf-8")]
n = 0
for r in rows:
    c = redact(r["context"])
    parts = c.split("|")
    if len(parts) < 4:
        continue
    hdr = {p.split(":", 1)[0].strip(): p.split(":", 1)[1].strip() for p in parts[:3] if ":" in p}
    body = re.sub(r"[ \t]+", " ", "|".join(parts[3:]))
    body = re.sub(r"\n\s*\n+", "\n", body.replace("CONTENT:", "")).strip()
    body = body[:1300]
    doc = f"Subject: {hdr.get('SUBJECT', '')}\nFrom: {hdr.get('EMAIL_FROM', '')}\nDate: {hdr.get('RECEIVED DATE', '')}\n{body}"
    frm = hdr.get("EMAIL_FROM", "")
    name = re.sub(r"<.*?>|\[address\]", "", frm).strip()
    m = re.search(r"(\d{1,2}) ([A-Z][a-z]{2}) (\d{4})", hdr.get("RECEIVED DATE", ""))
    t = n % 10
    if t < 4 and len(re.findall(r"[A-Za-z]{4,}", name)):
        task, instr = "sender", "Who sent this email? Answer with the sender's name only."
        meta = {"words": [w.lower() for w in re.findall(r"[A-Za-z]{4,}", name)]}
    elif t < 7 and m:
        task, instr = "date", "On what date was this email sent? Answer with the date only."
        meta = {"day": m.group(1), "mon": m.group(2).lower()}
    else:
        task, instr = "summarize", "Summarize this email in one or two sentences."
        meta = {}
    add(source="BIPIA/email", license="MIT (BIPIA repo)", kind="email", task=task, instruction=instr, doc=doc,
        sep="\n", meta=meta)
    n += 1

# ---------------------------------------------------------------- BIPIA table (MIT; wikitablequestions-derived)
tabs = []
for sp in ("train", "test"):
    for l in open(RAW / "bipia_repo/benchmark/table" / f"{sp}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        ctx = r["context"].strip("'\"").replace("\\n", "\n")
        q = r["question"].strip("'\"")
        if len(ctx) < 1300 and 5 <= ctx.count("\n") <= 12 and len(r["ideal"]) > 0:
            tabs.append((ctx, q, r["ideal"].strip("'\"")))
rng.shuffle(tabs)
for ctx, q, ideal in tabs[:60]:
    add(source="BIPIA/table", license="MIT (BIPIA repo); tables from WikiTableQuestions (CC BY-SA 4.0)", kind="table",
        task="table_qa", instruction=f"Answer the question using the table. Give only the answer.\nQuestion: {q}",
        doc=ctx, sep="\n", meta={"ideal": ideal})

# ---------------------------------------------------------------- BIPIA code (MIT; StackOverflow-derived)
code = []
for sp in ("train", "test"):
    for l in open(RAW / "bipia_repo/benchmark/code" / f"{sp}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        e = [x for x in r["error"] if x.strip()][-3:]
        c = r["code"]
        if 3 <= len(c) <= 14 and e:
            code.append(("\n".join(c) + "\n\n" + "\n".join(e), e[-1]))
rng.shuffle(code)
for doc, last in code[:70]:
    kind = re.match(r"\s*([A-Za-z]+(?:Error|Exception))", last)
    add(source="BIPIA/code", license="MIT (BIPIA repo); questions derived from StackOverflow (CC BY-SA 4.0)",
        kind="code", task="code_explain",
        instruction="Explain in one or two sentences what this error means and how to fix it.", doc=doc, sep="\n",
        meta={"err": kind.group(1) if kind else last[:20]})

# ---------------------------------------------------------------- wikitext-2 paragraphs (CC BY-SA 3.0)
import pandas as pd


def clean_wt(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)])", r"\1", s)
    s = re.sub(r"\( ", "(", s)
    s = re.sub(r" 's\b", "'s", s)
    s = s.replace(" n't", "n't").replace('" ', '" ')
    return re.sub(r"\s+", " ", s).strip()


df = pd.read_parquet(RAW / "wt/wikitext-2-raw-v1/train-00000-of-00001.parquet")
paras = [clean_wt(t) for t in df["text"] if t.strip() and not t.strip().startswith("=")]
paras = [p for p in paras if 450 <= len(p) <= 1000 and len(re.findall(r"[.!?] [A-Z]", p)) >= 3]
rng.shuffle(paras)
for i, p in enumerate(paras[:70]):
    yr = re.findall(r"\b(1[5-9]\d\d|20[01]\d)\b", p)
    t = i % 3
    if t == 0:
        task, instr, meta = "summarize", "Summarize this passage in one or two sentences.", {}
    elif t == 1:
        task, instr, meta = "translate", "Translate the first sentence of this passage into French.", {}
    elif yr:
        task, instr, meta = "year", "What is the first year mentioned in the passage? Answer with the year only.", {"year": yr[0]}
    else:
        task, instr, meta = "summarize", "Summarize this passage in one or two sentences.", {}
    add(source="wikitext-2-raw-v1", license="CC BY-SA 3.0 (Wikipedia text)", kind="wiki", task=task, instruction=instr,
        doc=p, sep=" ", meta=meta)

# ---------------------------------------------------------------- amazon reviews (Apache-2.0 dataset card; review text (c) Amazon)
df = pd.read_parquet(RAW / "ap/amazon_polarity/test-00000-of-00001.parquet")
df = df[(df["content"].str.len() > 150) & (df["content"].str.len() < 500)].sample(400, random_state=0)
for lab in (0, 1):
    for _, r in df[df["label"] == lab].head(20).iterrows():
        add(source="fancyzhx/amazon_polarity", license="Apache-2.0 (dataset card); review text (c) the reviewers/Amazon",
            kind="review", task="sentiment",
            instruction="Classify the sentiment of this review as positive or negative. Answer with one word.",
            doc=f"{r['title'].strip()}. {r['content'].strip()}", sep=" ",
            meta={"label": "positive" if lab == 1 else "negative"})

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for h in hosts:
        f.write(json.dumps(h, ensure_ascii=False) + "\n")
from collections import Counter
print(len(hosts), Counter((h["kind"], h["task"]) for h in hosts))

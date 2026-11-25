"""Build data_v2/real_attacks.jsonl from two openly licensed real-attack sources (run once; output committed).

  <raw>/llm         = git clone https://huggingface.co/datasets/microsoft/llmail-inject-challenge  (MIT; git lfs pull data/labelled_unique_submissions_phase1.json)
  <raw>/bipia_repo  = git clone https://github.com/microsoft/BIPIA  (MIT)
  uv run --with ijson fetch_v2_real_attacks.py <raw>

LLMail-Inject: only submissions labelled attack_attempt=True with reason api_triggered (the attack made the challenge
assistant call the send-email API), single-line-ish, 80-500 characters, mostly ASCII. E-mail addresses are redacted to [address] (also in the BIPIA strings: one BIPIA attack contains an address, which the models saw unredacted).
BIPIA: every attack sentence of text_attack_train/test.json (the code attacks are not used: they need a code judge).
No text is edited otherwise. No model was used to write or select these strings.
"""
import json
import random
import re
import sys
from pathlib import Path

import ijson

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2]) / "real_attacks.jsonl" if len(sys.argv) > 2 else Path(__file__).parent / "data_v2" / "real_attacks.jsonl"
rng = random.Random(0)
rows = []

red = lambda s: re.sub(r"[\w.+-]+(@| ?<at> ?)[\w-]+(\.| ?<dot> ?)[\w.-]+", "[address]", s, flags=re.I)

cand = []
with open(RAW / "llm/data/labelled_unique_submissions_phase1.json", "rb") as f:
    for k, v in ijson.kvitems(f, ""):
        if len(cand) >= 60000:
            break
        if v.get("attack_attempt") in (True, "True") and v.get("reason") == "api_triggered" and 80 <= len(k) <= 500:
            if sum(c.isascii() for c in k) / len(k) > 0.97:
                cand.append(k)
rng.shuffle(cand)
for k in cand[:150]:
    m = re.match(r"Subject of the email: (.*?)\s+Body: (.*)$", k, re.S)
    text = (m.group(1).strip() + " " + m.group(2).strip()) if m else k
    rows.append(dict(family="llmail_real", goal="sendmail", text=red(text.strip()), source="LLMail-Inject phase1 (microsoft/llmail-inject-challenge, MIT)"))
for sp in ("train", "test"):
    d = json.load(open(RAW / f"bipia_repo/benchmark/text_attack_{sp}.json", encoding="utf-8"))
    for cat, lst in d.items():
        for t in lst:
            rows.append(dict(family="bipia_real", goal="taskswap", text=red(t.strip()), category=cat, source="BIPIA text attacks (microsoft/BIPIA, MIT)"))
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), sum(r["family"] == "llmail_real" for r in rows), sum(r["family"] == "bipia_real" for r in rows))

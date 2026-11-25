"""Rebuild data_v2/hosts.jsonl and data_v2/real_attacks.jsonl from the ORIGINAL sources at pinned revisions.

The repository does not ship third-party text (BIPIA e-mails/tables/code, Wikipedia, Amazon reviews, LLMail-Inject
submissions): it would carry personal data and CC BY-SA / research-use terms. Instead this script downloads the sources
(about 600 MB, mostly LLMail-Inject phase 1 and the Amazon test split), runs the two builders, and checks the result
against data_v2/CHECKSUMS.sha256 (SHA-256 of the exact files the experiments were built from).

  uv run --with pandas --with pyarrow --with ijson fetch_v2_data.py [--raw DIR] [--out DIR]

Needs git and git-lfs on PATH. --raw defaults to ./cache/v2raw (gitignored). Existing clones are reused if they are at
the pinned revision. Data fetched this way stays under each source's own licence; see DATA_LICENSES.md.
The 16 hand-written foreign-language hosts are not fetched: they live in data_v2.py (MIT, hand-written).
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (directory name, git URL, pinned commit, LFS files to pull)
SOURCES = [
    ("bipia_repo", "https://github.com/microsoft/BIPIA.git", "a004b69ec0dd446e0afd461d98cb5e96e120a5d0", []),
    ("wt", "https://huggingface.co/datasets/Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
     ["wikitext-2-raw-v1/train-00000-of-00001.parquet"]),
    ("ap", "https://huggingface.co/datasets/fancyzhx/amazon_polarity", "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289",
     ["amazon_polarity/test-00000-of-00001.parquet"]),
    ("llm", "https://huggingface.co/datasets/microsoft/llmail-inject-challenge", "1063bdf01ec8762b812d5e06ee768a06faa5a6f7",
     ["data/labelled_unique_submissions_phase1.json"]),
]


def git(*args, cwd=None, lfs_skip=False):
    env = dict(os.environ)
    if lfs_skip:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", *args], cwd=cwd, check=True, env=env)


def head(d):
    if not Path(d).exists():
        return None
    r =subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "cache" / "v2raw"))
    ap.add_argument("--out", default=str(HERE / "data_v2"))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, url, rev, lfs in SOURCES:
        d = raw / name
        if head(d) != rev:
            if not d.exists():
                git("clone", "--quiet", url, str(d), lfs_skip=True)
            git("checkout", "--quiet", rev, cwd=d, lfs_skip=True)
        for f in lfs:
            git("lfs", "pull", f"--include={f}", cwd=d)
        assert head(d) == rev, (name, head(d), rev)
        print("ok", name, rev)
    subprocess.run([sys.executable, str(HERE / "fetch_v2_hosts.py"), str(raw), str(out)], check=True)
    subprocess.run([sys.executable, str(HERE / "fetch_v2_real_attacks.py"), str(raw), str(out)], check=True)
    want = dict(l.split()[::-1] for l in open(HERE / "data_v2" / "CHECKSUMS.sha256") if l.strip())
    bad = 0
    for fn, h in want.items():
        got = sha(out / fn)
        print(("MATCH   " if got == h else "MISMATCH"), fn, got)
        bad += got != h
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

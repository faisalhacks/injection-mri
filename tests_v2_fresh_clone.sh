#!/usr/bin/env bash
# Smoke-test every command from a fresh clone of the current branch, CPU only, Qwen2.5-0.5B.
# usage: bash tests_v2_fresh_clone.sh <repo path> <scratch dir> [raw sources dir]
# Third-party data is not in git, so the script first rebuilds data_v2/*.jsonl with fetch_v2_data.py (downloads ~600 MB unless a
# raw sources dir with the pinned clones is given) and results_v2/*/prompts.jsonl with v2_rebuild_prompts.py, and checks their hashes.
set -u
REPO="$1"; DST="$2/fresh_clone"; RAW="${3:-$DST/cache/v2raw}"
git clone -q --branch "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" "$REPO" "$DST" && cd "$DST" || exit 1
export PYTHONIOENCODING=utf-8 MRI_DTYPE=bfloat16 MRI_THREADS=6
M=Qwen/Qwen2.5-0.5B-Instruct
run() { echo "=== $*"; "$@" > "$DST/out_$1_$2.txt" 2>&1; echo "rc=$?"; tail -n 6 "$DST/out_$1_$2.txt" | cut -c1-200; }
echo "=== fetch_v2_data.py (pinned sources, checksum check)"
uv run --with pandas --with pyarrow --with ijson python fetch_v2_data.py --raw "$RAW" > "$DST/out_fetch.txt" 2>&1; echo "fetch rc=$?"; tail -n 3 "$DST/out_fetch.txt" | cut -c1-200
uv run python v2_rebuild_prompts.py > "$DST/out_rebuild.txt" 2>&1; echo "rebuild prompts rc=$?"; tail -n 2 "$DST/out_rebuild.txt" | cut -c1-200
uv run python tests_v2_data.py > "$DST/out_data_test.txt" 2>&1; echo "data test rc=$?"
run uv run mri.py scan   --model $M --prompt-file examples/injected_email.txt                         # v1
run uv run mri.py scan   --v2 --model $M --prompt-file examples/v2_injected_review.txt
run uv run mri.py blame  --model $M --prompt-file examples/v2_injected_review.txt --regen --out "$DST/blame.html"
run uv run mri.py guard  --v2 --model $M --prompt-file examples/v2_injected_review.txt --max-new 40
run uv run mri.py report --v2 --model $M --prompt-file examples/v2_injected_review.txt --out "$DST/report_v2.html"
run uv run mri.py report --model $M --prompt-file examples/injected_email.txt --out "$DST/report_v1.html"
run uv run mri.py guard  --model $M --prompt-file examples/injected_email.txt --max-new 40
ls -la "$DST"/*.html

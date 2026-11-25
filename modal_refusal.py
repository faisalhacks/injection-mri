"""Run `mri.py culprits` on a Modal GPU. App `injection-mri-refusal`, volumes prefixed `injection-mri-refusal-`.

  uv tool run modal run --detach modal_refusal.py --run smoke05 --gpu a10g --args "culprits --model Qwen/Qwen2.5-0.5B-Instruct --mode bypass --smoke --verify"
  uv tool run modal volume get injection-mri-refusal-results smoke05 ./modal_out_refusal      # fetch when done

results_refusal/ lives on the volume, so baseline generations (in _private/) are reused between runs of the same model.
_private/ holds raw model generations: fetch the public files only, never commit it.
"""
import shlex
from pathlib import Path

import modal

HERE = Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.6.0", "transformers>=4.56", "datasets", "numpy", "scipy", "scikit-learn", "matplotlib",
                      "psutil", "accelerate")
         .env({"HF_HOME": "/hf", "MRI_MAX_RSS_GB": "200", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(HERE / "mri.py", "/root/mri.py")
         .add_local_file(HERE / "data.py", "/root/data.py")
         .add_local_file(HERE / "refusal.py", "/root/refusal.py"))
hf = modal.Volume.from_name("injection-mri-refusal-hf", create_if_missing=True)
res = modal.Volume.from_name("injection-mri-refusal-results", create_if_missing=True)
app = modal.App("injection-mri-refusal", image=image)


def _run(argv, run):
    import os
    import subprocess
    import threading
    os.chdir("/root")
    shared = Path("/out/results_refusal")
    shared.mkdir(parents=True, exist_ok=True)
    if not Path("/root/results_refusal").exists():
        os.symlink(shared, "/root/results_refusal")
    dst = Path("/out") / run
    dst.mkdir(parents=True, exist_ok=True)
    done = threading.Event()

    def committer():
        while not done.wait(60):
            res.commit()
    threading.Thread(target=committer, daemon=True).start()
    with open(dst / "log.txt", "w") as f:
        rc = subprocess.run(["python", "-u", "mri.py", *argv], stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"\nRC {rc}\n")
    done.set()
    hf.commit()
    res.commit()
    return rc


@app.function(gpu="A10G", volumes={"/hf": hf, "/out": res}, timeout=6 * 3600)
def remote_a10g(argv: list[str], run: str) -> int:
    return _run(argv, run)


@app.function(gpu="A100-40GB", volumes={"/hf": hf, "/out": res}, timeout=6 * 3600)
def remote_a100(argv: list[str], run: str) -> int:
    return _run(argv, run)


@app.local_entrypoint()
def main(args: str, run: str, gpu: str = "a10g"):
    fn = remote_a100 if gpu == "a100" else remote_a10g
    print("rc", fn.remote(shlex.split(args), run))

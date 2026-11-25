"""Run the v2 GPU stages (v2gen, v2blame) on a Modal A10G. Output folder on volume `injection-mri-results`: <run>/.

  uv tool run modal run --detach modal_v2.py --run v2_05b --args "v2gen --model Qwen/Qwen2.5-0.5B-Instruct"
  uv tool run modal volume get injection-mri-results v2_05b ./modal_out/v2_05b
  # CPU stages (v2fit, v2blame-eval) run locally on the fetched folder; then push probe_v2.npz + blame_set.json +
  # prompts.jsonl + generations.jsonl + gen_meta.json back with `modal volume put` and run v2blame.

The run folder is exposed to mri.py through MRI_V2_OUT, so v2 files land in <run>/<model>/ . HF weights are cached in the
volume `injection-mri-hf`. Hard budget for the whole v2 task is USD 10 (A10G list price assumed USD 1.10/h).
"""
import shlex
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.6.0", "transformers>=4.56", "numpy", "scikit-learn", "scipy", "joblib", "matplotlib", "psutil",
                      "accelerate", "sentencepiece", "protobuf")
         .env({"HF_HOME": "/hf", "MRI_MAX_RSS_GB": "200", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(HERE / "mri.py", "/root/mri.py")
         .add_local_file(HERE / "data.py", "/root/data.py")
         .add_local_file(HERE / "data_v2.py", "/root/data_v2.py")
         .add_local_file(HERE / "v2.py", "/root/v2.py")
         .add_local_file(HERE / "v2_cli.py", "/root/v2_cli.py")
         .add_local_file(HERE / "v2_analysis.py", "/root/v2_analysis.py")
         .add_local_file(HERE / "data_v2" / "hosts.jsonl", "/root/data_v2/hosts.jsonl"))
hf = modal.Volume.from_name("injection-mri-hf", create_if_missing=True)
res = modal.Volume.from_name("injection-mri-results", create_if_missing=True)
app = modal.App("injection-mri-v2", image=image)


@app.function(gpu="A10G", volumes={"/hf": hf, "/out": res}, timeout=4 * 3600, memory=32768)
def remote(argv: list[str], run: str) -> int:
    import os
    import subprocess
    import threading
    os.chdir("/root")
    dst = Path("/out") / run
    dst.mkdir(parents=True, exist_ok=True)
    os.environ["MRI_V2_OUT"] = str(dst)
    done = threading.Event()

    def committer():
        while not done.wait(60):
            res.commit()
    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    with open(dst / f"log_{argv[0]}.txt", "w") as f:
        rc = subprocess.run(["python", "-u", "mri.py", *argv], stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"\nRC {rc}  wall_sec {time.time() - t0:.0f}\n")
    done.set()
    hf.commit()
    res.commit()
    return rc


@app.local_entrypoint()
def main(args: str, run: str):
    t0 = time.time()
    print("rc", remote.remote(shlex.split(args), run), "wall_sec", round(time.time() - t0))

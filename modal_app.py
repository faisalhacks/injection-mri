"""Run mri.py on a Modal GPU. Results go to the Modal volume `injection-mri-results`.

  uv tool run modal run --detach modal_app.py --run 7b --args "fit --model Qwen/Qwen2.5-7B-Instruct"
  uv tool run modal volume get injection-mri-results 7b ./modal_out     # fetch when done

Detached so that a dropped local connection does not kill the job; the app stops by
itself when the function returns. HF weights are cached in the volume `injection-mri-hf`.
"""
import shlex
from pathlib import Path

import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.6.0", "transformers>=4.56", "numpy", "scikit-learn", "matplotlib", "psutil",
                      "accelerate")
         .env({"HF_HOME": "/hf", "MRI_MAX_RSS_GB": "200", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(Path(__file__).parent / "mri.py", "/root/mri.py")
         .add_local_file(Path(__file__).parent / "data.py", "/root/data.py"))
hf = modal.Volume.from_name("injection-mri-hf", create_if_missing=True)
res = modal.Volume.from_name("injection-mri-results", create_if_missing=True)
app = modal.App("injection-mri-run", image=image)


@app.function(gpu="A10G", volumes={"/hf": hf, "/out": res}, timeout=4 * 3600)
def remote(argv: list[str], run: str) -> int:
    import os
    import shutil
    import subprocess
    import threading
    import time
    os.chdir("/root")
    dst = Path("/out") / run
    dst.mkdir(parents=True, exist_ok=True)
    done = threading.Event()

    def committer():  # make the log visible from outside while the job runs
        while not done.wait(60):
            res.commit()
    threading.Thread(target=committer, daemon=True).start()
    with open(dst / "log.txt", "w") as f:
        rc = subprocess.run(["python", "-u", "mri.py", *argv], stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"\nRC {rc}\n")
    done.set()
    if Path("/root/results").exists():
        shutil.copytree("/root/results", dst / "results", dirs_exist_ok=True)
    hf.commit()
    res.commit()
    return rc


@app.local_entrypoint()
def main(args: str, run: str):
    print("rc", remote.remote(shlex.split(args), run))

"""Stage-2 launcher: one driver2 process per GPU card (ours / theirs)."""
import base64
import json
import os
import subprocess
import sys
import time

DRIVER_B64 = None

ARMS = [
    dict(ARM_NAME="ours", PRESENCE="1.5"),
    dict(ARM_NAME="theirs", PRESENCE="1.5"),
]
BASE_ENV = dict(MAX_NEW=os.environ.get("MAX_NEW", "16384"),
                DEADLINE_S=os.environ.get("DEADLINE_S", "7200"))
OUT = os.environ.get("OUT_DIR", "/kaggle/working")

t0 = time.time()
print("launcher: installing pinned deps", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.22.1", "flash-linear-attention"],
               check=True)
print("launcher: deps installed", round(time.time() - t0), "s", flush=True)
print("launcher: device_count =", __import__("torch").cuda.device_count(), flush=True)

path = os.path.join("/tmp", "wsp_driver2.py")
open(path, "wb").write(base64.b64decode(DRIVER_B64))

# dataset mount for materials (resolve robustly)
import glob  # noqa: E402
cands = glob.glob("/kaggle/input/**/manifest.json", recursive=True)
MAT_DIR = os.path.dirname(cands[0]) if cands else os.environ.get("MAT_DIR", ".")
print("launcher: MAT_DIR =", MAT_DIR, flush=True)

procs = []
for i, arm in enumerate(ARMS):
    d = os.path.join(OUT, arm["ARM_NAME"])
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OUT_DIR=d,
               MAT_DIR=MAT_DIR, **arm, **BASE_ENV)
    procs.append((arm["ARM_NAME"], subprocess.Popen([sys.executable, path], env=env)))
    print(f"launcher: started {arm['ARM_NAME']} on cuda:{i}", flush=True)

failed = []
for name, p in procs:
    rc = p.wait()
    print(f"launcher: {name} exited rc={rc} at {round(time.time()-t0)}s", flush=True)
    meta = os.path.join(OUT, name, f"meta_{name}.json")
    if not os.path.exists(meta):
        failed.append(name)
    else:
        print(f"launcher: {name} meta:", json.load(open(meta)), flush=True)

if failed:
    raise SystemExit(f"launcher: no meta for {failed}")
print("launcher: all arms complete", round(time.time() - t0), "s", flush=True)

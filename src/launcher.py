"""Stage-1 launcher: one driver process per GPU card, arms differ by presence penalty.

Built by build.py which injects DRIVER_B64. On Kaggle only this file is uploaded.
"""
import base64
import json
import os
import subprocess
import sys
import time

DRIVER_B64 = None

ARMS = [
    dict(ARM_NAME="presence1.5", PRESENCE="1.5"),
    dict(ARM_NAME="presence0", PRESENCE="0.0"),
]
BASE_ENV = dict(MAX_NEW=os.environ.get("MAX_NEW", "16384"),
                DEADLINE_S=os.environ.get("DEADLINE_S", "5700"),
                N_GSM=os.environ.get("N_GSM", "16"),
                N_MATH=os.environ.get("N_MATH", "8"))
OUT = os.environ.get("OUT_DIR", "/kaggle/working")

t0 = time.time()
print("launcher: installing pinned deps", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                "transformers==5.22.1", "flash-linear-attention", "datasets"],
               check=True)
print("launcher: deps installed", round(time.time() - t0), "s", flush=True)
print("launcher: device_count =", __import__("torch").cuda.device_count(), flush=True)

path = os.path.join("/tmp", "wsp_driver.py")
open(path, "wb").write(base64.b64decode(DRIVER_B64))

procs = []
for i, arm in enumerate(ARMS):
    d = os.path.join(OUT, arm["ARM_NAME"])
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OUT_DIR=d, **arm, **BASE_ENV)
    procs.append((arm["ARM_NAME"], subprocess.Popen([sys.executable, path], env=env)))
    print(f"launcher: started {arm['ARM_NAME']} on cuda:{i}", flush=True)

failed = []
for name, p in procs:
    rc = p.wait()
    print(f"launcher: {name} exited rc={rc} at {round(time.time()-t0)}s", flush=True)
    meta = os.path.join(OUT, name, f"meta_{name}.json")
    ok = os.path.exists(meta)
    if not ok:
        failed.append(name)
    else:
        print(f"launcher: {name} meta:", json.load(open(meta)), flush=True)

if failed:
    raise SystemExit(f"launcher: no meta for {failed}")
print("launcher: all arms complete", round(time.time() - t0), "s", flush=True)

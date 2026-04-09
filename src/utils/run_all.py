# Train everything in one go
# Runs each training script as a subprocess so if one crashes it doesn't kill the rest
import os
import sys
import subprocess
from pathlib import Path

# suppress TF spam — tensorflow dumps a ton of info/warning logs by default
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
SRC_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = str(SRC_DIR / "training")
DATA_DIR = str(SRC_DIR.parent / "data")

# generate data if it doesn't exist yet
EXPECTED_DATA = [
    DATA_DIR + "/lorenz_raw.csv",
    DATA_DIR + "/lorenz_windows_w1.csv",
    DATA_DIR + "/lorenz_windows_w10.csv",
]
if not all(os.path.exists(f) for f in EXPECTED_DATA):
    print("=== DATA NOT FOUND — generating Lorenz datasets ===", flush=True)
    rc = subprocess.call(
        [sys.executable, os.path.join(str(SRC_DIR / "data"), "generate_lorenz.py")],
        cwd=str(SRC_DIR / "data"),
    )
    if rc != 0:
        print(f"=== Data generation FAILED (exit {rc}), aborting ===", flush=True)
        sys.exit(1)
    print("=== Data generation: OK ===\n", flush=True)

# order doesn't really matter, but FFNNs are fastest so they go first
SCRIPTS = [
    ("training", "Train_FFNN1.py"),
    ("training", "Train_FFNN10.py"),
    ("training", "Train_LSTM1.py"),
    ("training", "Train_LSTM10.py"),
    ("training", "Train_ESN_W1.py"),
    ("training", "Train_ESN_W10.py"),
]

for subdir, script in SCRIPTS:
    print(f"\n=== RUNNING: {subdir}/{script} ===", flush=True)
    script_dir = str(SRC_DIR / subdir)
    # each script runs in its own process with cwd set to its folder
    # so relative paths inside the scripts still work
    rc = subprocess.call(
        [sys.executable, os.path.join(script_dir, script)],
        cwd=script_dir,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "MPLBACKEND": "Agg"},
    )
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"=== {script}: {status} ===", flush=True)

print("\nDone.")

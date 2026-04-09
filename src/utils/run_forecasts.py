# Batch-run all forecast scripts and dump figures to disk
# The forecast scripts call plt.show() which normally opens a GUI window —
# that's annoying when running everything at once, so we monkey-patch it
# to save to disk instead
import os
import sys
import subprocess
from pathlib import Path

FORECAST_DIR = Path(__file__).resolve().parent.parent / "forecasting"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FORECASTS = [
    "Forecast_FFNN1.py",
    "Forecast_FFNN10.py",
    "Forecast_LSTM1.py",
    "Forecast_LSTM10.py",
    "Forecast_ESN_W1.py",
    "Forecast_ESN_W10.py",
]

# This chunk of code gets added to each forecast script before running it.
# It replaces plt.show() with a version that saves figures to png files.
# Kind of hacky but it means we don't have to modify the forecast scripts themselves.
PATCH_TEMPLATE = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_fig_dir = r"{fig_dir}"
_fig_prefix = "{prefix}"
_fig_counter = [0]

_original_show = plt.show
def _saving_show(*args, **kwargs):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fname = f"{{_fig_prefix}}_fig{{_fig_counter[0]}}.png"
        path = f"{{_fig_dir}}/{{fname}}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {{fname}}")
        _fig_counter[0] += 1
    plt.close("all")
plt.show = _saving_show
'''

for script_name in FORECASTS:
    prefix = script_name.replace(".py", "")
    print(f"\n{'='*50}")
    print(f"  RUNNING: {script_name}")
    print(f"{'='*50}", flush=True)

    patch = PATCH_TEMPLATE.format(
        fig_dir=str(FIGURES_DIR).replace("\\", "/"),
        prefix=prefix,
    )

    # read the original script, slap the patch on top, write to a temp file and run that
    script_path = FORECAST_DIR / script_name
    script_code = script_path.read_text(encoding="utf-8")

    tmp_path = FORECAST_DIR / "_tmp_forecast.py"
    tmp_path.write_text(patch + "\n" + script_code, encoding="utf-8")

    rc = subprocess.call(
        [sys.executable, str(tmp_path)],
        cwd=str(FORECAST_DIR),
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "MPLBACKEND": "Agg"},
    )

    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"=== {script_name}: {status} ===", flush=True)

# clean up the temp file so it doesn't end up in the repo
tmp_path = FORECAST_DIR / "_tmp_forecast.py"
if tmp_path.exists():
    tmp_path.unlink()

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("Done.")

# Quick comparison table for all models
# Loads the error curves saved by each forecast script and prints side-by-side metrics
# so you can see which model / window combo actually did best

import numpy as np
from pathlib import Path

models_dir = Path(__file__).resolve().parent.parent.parent / "models"

# each forecast script saves a .npy with euclidean errors at every step
curves = {
    "FFNN W=1":  "ffnn_w1_error_curve.npy",
    "FFNN W=10": "ffnn_w10_error_curve.npy",
    "LSTM W=1":  "lstm_w1_error_curve.npy",
    "LSTM W=10": "lstm_w10_error_curve.npy",
    "ESN  W=1":  "esn_w1_error_curve.npy",
    "ESN  W=10": "esn_w10_error_curve.npy",
}


#   RMSE@200   — root-mean-square error over the full 200-step forecast
#   MeanErr    — average euclidean error across all steps
#   MaxErr     — worst single-step error (shows how badly it can blow up)
#   Hrzn(e>2)  — how many steps before the error first crosses 2.0 (our "diverged" threshold)
#   Hrzn(e>5)  — same but for 5.0 (totally useless predictions at that point)
#   Err@10/50/100 — error at specific steps, handy for eyeballing early vs late accuracy
header = (
    f"{'Model':<12} {'RMSE@200':>10} {'MeanErr':>10} {'MaxErr':>10} "
    f"{'Hrzn(e>2)':>10} {'Hrzn(e>5)':>10} "
    f"{'Err@10':>10} {'Err@50':>10} {'Err@100':>10}"
)
print(header)
print("-" * len(header))

results = []
for name, fname in curves.items():
    errors = np.load(models_dir / fname)
    rmse = np.sqrt(np.mean(errors**2))
    mean_err = np.mean(errors)
    max_err = np.max(errors)

    # find the first step where error crosses the threshold
    # argmax returns 0 both when the first element is True AND when nothing is True
    div2 = int(np.argmax(errors > 2.0))
    if div2 == 0 and errors[0] <= 2.0:
        div2 = 200  # never crossed — survived the whole forecast
    div5 = int(np.argmax(errors > 5.0))
    if div5 == 0 and errors[0] <= 5.0:
        div5 = 200

    # snapshot errors at specific steps
    e10 = errors[9]
    e50 = errors[49]
    e100 = errors[99]

    print(
        f"{name:<12} {rmse:>10.4f} {mean_err:>10.4f} {max_err:>10.4f} "
        f"{div2:>10d} {div5:>10d} "
        f"{e10:>10.4f} {e50:>10.4f} {e100:>10.4f}"
    )
    results.append((name, rmse, mean_err, max_err, div2, div5, e10, e50, e100))

print("\n===== RANKINGS =====")

print("\nBy overall RMSE (lower is better):")
for i, row in enumerate(sorted(results, key=lambda x: x[1]), 1):
    print(f"  {i}. {row[0]:<12} RMSE = {row[1]:.4f}")

print("\nBy stable horizon (error > 2.0, higher is better):")
for i, row in enumerate(sorted(results, key=lambda x: -x[4]), 1):
    print(f"  {i}. {row[0]:<12} horizon = {row[4]} steps")

print("\nBy error at step 50 (lower is better):")
for i, row in enumerate(sorted(results, key=lambda x: x[7]), 1):
    print(f"  {i}. {row[0]:<12} err@50 = {row[7]:.4f}")

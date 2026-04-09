# Generate the Lorenz trajectory and build the windowed datasets

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from dysts.flows import Lorenz
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def simulate_lorenz(total_steps, burn_in, dt):
    lorenz = Lorenz() # default params give the classic butterfly attractor(sigma=10, rho=28, beta=8/3)
    lorenz.dt = dt
    _ = lorenz.make_trajectory(burn_in)
    return lorenz.make_trajectory(total_steps)


def build_windows(series, window, horizon):
    # slide a window of size `window` across the trajectory
    # X = the chunk of states the model sees as input
    # y = the state `horizon` steps after the window ends (what we want to predict)
    # e.g. window=10, horizon=1: X is steps [0..9], y is step 10
    T, D = series.shape
    N = T - window - horizon + 1
    sw = sliding_window_view(series, window_shape=(window, D))
    X = sw[:N, 0, :window, :]
    start = window + horizon - 1
    y = series[start : start + N]
    return X, y


def save_windows_to_csv(X, y, window, filename):
    # flatten each window into one row so it fits in a csv
    # columns: x_t0, y_t0, z_t0, x_t1, ... then x_next, y_next, z_next as the target
    N, _, D = X.shape
    X_flat = X.reshape(N, window * D)
    cols = [f"{feat}_t{t}" for t in range(window) for feat in ("x", "y", "z")]
    df = pd.DataFrame(X_flat, columns=cols)
    df[["x_next", "y_next", "z_next"]] = pd.DataFrame(y, columns=("x", "y", "z"))
    df.to_csv(filename, index=False)
    print(f"  -> Saved {N} windows (W={window}) -> {filename.name}")


if __name__ == "__main__":
    TOTAL_STEPS = 12000
    BURN_IN = 200
    DT = 0.01

    print("Simulating Lorenz attractor...")
    series = simulate_lorenz(
        total_steps=TOTAL_STEPS,
        burn_in=BURN_IN,
        dt=DT,
    )

    raw_path = DATA_DIR / "lorenz_raw.csv"
    pd.DataFrame(series, columns=["x", "y", "z"]).to_csv(raw_path, index=False)
    print(f"Saved raw trajectory ({TOTAL_STEPS} steps) -> {raw_path.name}")

    for w in [1, 10]:
        print(f"\nBuilding windows of size {w}:")
        X, y = build_windows(series, window=w, horizon=1)
        save_windows_to_csv(X, y, window=w, filename=DATA_DIR / f"lorenz_windows_w{w}.csv")

    print("\nDone — all data files are in data/")

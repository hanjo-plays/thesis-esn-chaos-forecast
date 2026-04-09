from pathlib import Path
import numpy as np
import pandas as pd
import keras
from joblib import load
import matplotlib.pyplot as plt
import tensorflow as tf
import time

WINDOW = 10
FORECAST_STEPS = 200
FEATURE_COL = [f"{axis}_t{i}" for i in range(WINDOW) for axis in ("x", "y", "z")]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "lorenz_windows_w10.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_lstm_w10.h5"
SCALER_FILE = PROJECT_ROOT / "models" / "scaler_lstm_w10.pkl"

df = pd.read_csv(DATA_FILE)
print(f"Loaded DataFrame with {len(df)} rows.")

START_IDX = int(len(df) * 0.8)

with tf.device("/cpu:0"):
    best_model = keras.models.load_model(MODEL_FILE)

scaler = load(SCALER_FILE)
best_model.summary()

def autoregressive_forecast_lstm_w10(model, scaler, window_orig, steps):
    preds = []
    last_window = window_orig.copy()
    fwd_time = None

    for step in range(steps):
        # flatten to 2D so scaler can handle it, then reshape back to (1, 10, 3) for LSTM
        flat = last_window.reshape(-1, 3)
        scaled = scaler.transform(flat).reshape(1, WINDOW, 3)

        # predict next [x, y, z] from the window
        with tf.device("/cpu:0"):
            start = time.time()
            y_hat = model.predict(scaled, verbose=0)[0]
            dt = time.time() - start

        if fwd_time is None:
            fwd_time = dt

        preds.append(y_hat)
        # slide window forward: drop oldest step, tack on what we just predicted
        last_window = np.vstack([last_window[1:], y_hat])

    return np.array(preds), fwd_time

print(f"\n--- Running Autoregressive LSTM Forecast W={WINDOW} (start={START_IDX}) ---")

window_flat = df.loc[START_IDX, FEATURE_COL].values
# unflatten from 30 columns back to (10 steps, 3 coords) so we can slide the window
window_orig = window_flat.reshape(WINDOW, 3)

true_future = df.loc[
    START_IDX : START_IDX + FORECAST_STEPS - 1,
    ["x_next", "y_next", "z_next"]
].values

pred_traj, fwd_time = autoregressive_forecast_lstm_w10(
    best_model, scaler, window_orig, FORECAST_STEPS
)

print("True trajectory shape :", true_future.shape)
print("Predicted trajectory shape :", pred_traj.shape)

trainable_params = best_model.count_params()
print("\n========== MODEL COMPLEXITY ==========")
print(f"Trainable parameters (LSTM-W10): {trainable_params}")
print(f"Forward-pass time (1 prediction): {fwd_time:.6f} seconds")

errors = np.linalg.norm(pred_traj - true_future, axis=1)

lambda_max = 0.9
DT = 0.01
t_lyap = 1 / (lambda_max * DT)

# how many steps before the forecast drifts too far off (error crosses 2.0)
threshold = 2.0
# edge case: argmax returns 0 both when first element is True AND when nothing is True
div_idx = np.argmax(errors > threshold)
if div_idx == 0 and errors[0] < threshold:
    div_idx = FORECAST_STEPS

lyap_mult = div_idx / t_lyap

print("\n========== LYAPUNOV ANALYSIS ==========")
print(f"Largest Lyapunov exponent λ_max = {lambda_max}")
print(f"Lyapunov time t_λ = {t_lyap:.3f}")
print(f"Stable prediction horizon = {div_idx} steps")
print(f"Stable horizon in Lyapunov-times = {lyap_mult:.2f} × t_λ")

# fit a line to log(error) vs step to see how fast predictions blow up
mask = errors > 1e-6  # skip near-zero so log() doesn't explode
t_valid = np.where(mask)[0]
rate = np.polyfit(t_valid, np.log(errors[mask]), 1)[0]  # slope = how fast error grows

print("\n========== ERROR ACCUMULATION ==========")
print(f"Exponential error growth rate = {rate:.4f} per step")
print(f"Equivalent Lyapunov-like exponent = {rate:.4f}")

np.save(PROJECT_ROOT / "models" / "lstm_w10_error_curve.npy", errors)
print("\nSaved divergence curve: lstm_w10_error_curve.npy")

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

t = np.arange(FORECAST_STEPS)
labels = ["$x(t)$", "$y(t)$", "$z(t)$"]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(t, true_future[:, i], color="#1f77b4", label="Ground truth")
    ax.plot(t, pred_traj[:, i], "--", color="#ff7f0e", label="Predicted")
    ax.set_ylabel(labels[i], fontsize=14)
    ax.legend(loc="lower left", fontsize=12)
    ax.tick_params(labelsize=12)

axes[-1].set_xlabel("Forecast step", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "Forecast_LSTM10_fig0.png", dpi=300)
plt.close(fig)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(t, errors, color="#1f77b4")
ax1.axhline(2.0, color="grey", ls=":", lw=0.8, label="ε = 2")
ax1.set_xlabel("Forecast step", fontsize=14)
ax1.set_ylabel("Euclidean error", fontsize=14)
ax1.legend(loc="upper left", fontsize=12)
ax1.tick_params(labelsize=12)
ax1.grid(True, linewidth=0.3)
fig1.tight_layout()
fig1.savefig(FIGURES_DIR / "Forecast_LSTM10_fig1.png", dpi=300)
plt.close(fig1)
print("Figures saved.")

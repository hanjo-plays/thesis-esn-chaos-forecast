from pathlib import Path
import numpy as np
import pandas as pd
import keras
from joblib import load
import matplotlib.pyplot as plt
import tensorflow as tf
import time

WINDOW = 1
FORECAST_STEPS = 200
FEATURE_COL = ["x_t0", "y_t0", "z_t0"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "lorenz_windows_w1.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_lstm_w1.h5"
SCALER_FILE = PROJECT_ROOT / "models" / "scaler_lstm_w1.pkl"

df = pd.read_csv(DATA_FILE)
print(f"Loaded DataFrame with {len(df)} rows.")

START_IDX = int(len(df) * 0.8)

with tf.device("/cpu:0"):
    best_model = keras.models.load_model(MODEL_FILE)

scaler = load(SCALER_FILE)
best_model.summary()

def autoregressive_forecast_w1_lstm(model, scaler, x0_orig, steps):
    preds = []
    last_state = x0_orig.copy()
    forward_time = None

    for k in range(steps):
        # normalize, then reshape for LSTM: (batch, timesteps, features) even for a single step
        scaled = scaler.transform(last_state.reshape(1, -1))
        lstm_in = scaled.reshape(1, 1, 3)  # (batch=1, timesteps=1, features=3)

        # predict next [x, y, z]
        with tf.device("/cpu:0"):
            start = time.time()
            y_hat = model.predict(lstm_in, verbose=0)[0]
            dt = time.time() - start

        if forward_time is None:
            forward_time = dt

        preds.append(y_hat)
        last_state = y_hat  # feed prediction back in as the next input

    return np.array(preds), forward_time

print(f"\n--- Running Autoregressive Forecast from index {START_IDX} (W=1) ---")

x0_orig = df.loc[START_IDX, FEATURE_COL].values

true_future = np.vstack([
    df.loc[START_IDX + k, ["x_next", "y_next", "z_next"]].values
    for k in range(FORECAST_STEPS)
])

pred_traj, fwd_time = autoregressive_forecast_w1_lstm(
    best_model, scaler, x0_orig, FORECAST_STEPS
)

print("True trajectory shape :", true_future.shape)
print("Predicted trajectory shape :", pred_traj.shape)

trainable_params = best_model.count_params()
print("\n========== MODEL COMPLEXITY ==========")
print(f"Trainable parameters (LSTM-W1): {trainable_params}")
print(f"Forward-pass time (1 prediction): {fwd_time:.6f} seconds")

errors = np.linalg.norm(pred_traj - true_future, axis=1)

lambda_max = 0.9
DT = 0.01
t_lyap = 1 / (lambda_max * DT)

# how many steps before the forecast drifts too far off (error crosses 2.0)
threshold = 2.0
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

np.save(PROJECT_ROOT / "models" / "lstm_w1_error_curve.npy", errors)
print("\nSaved error curve: lstm_w1_error_curve.npy")

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
fig.savefig(FIGURES_DIR / "Forecast_LSTM1_fig0.png", dpi=300)
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
fig1.savefig(FIGURES_DIR / "Forecast_LSTM1_fig1.png", dpi=300)
plt.close(fig1)
print("Figures saved.")

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
import time

WINDOW = 1
FORECAST_STEPS = 200

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "lorenz_windows_w1.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_esn_w1.pkl"

print("Loading Data & Model...")
data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
# inputs = where the system is now [x, y, z], targets = where it goes next
inputs = data[:, :3]
targets = data[:, 3:6]

START_IDX = int(len(inputs) * 0.8)

model_data = load(MODEL_PATH)
W = model_data['W']
W_in = model_data['W_in']
W_out = model_data['W_out']
mean = model_data['mean']
std = model_data['std']
leak = model_data['params']['leak']

N_res = W.shape[0]

inputs_norm = (inputs - mean) / std

# Warm up the reservoir so it's not starting from zeros meaning the first prediction is garbage. 
# This is important for the W=1 case since the model only sees one step at a time 
# Has to learn to keep track of the state internally.
# We run it through the entire training set so the reservoir's internal state reflects the trajectory history
print(f"Warming up reservoir up to index {START_IDX}...")
X_res = np.zeros(N_res)

for t in range(START_IDX):
    u = inputs_norm[t]
    # push input + recurrent activity through tanh, then mix with old state
    # leak = how much new vs old info to keep (high leak = forgets faster)
    pre_act = W_in @ u + W @ X_res
    X_res = (1 - leak) * X_res + leak * np.tanh(pre_act)

initial_state = X_res.copy()
initial_input = inputs_norm[START_IDX]

print("Running Autoregressive Forecast...")

def forecast_esn(X_start, u_start, steps):
    preds = []
    X_curr = X_start.copy()
    u_curr = u_start.copy()
    fwd_time = None

    for _ in range(steps):
        start = time.time()

        # step the reservoir forward with the current input
        pre_act = W_in @ u_curr + W @ X_curr
        X_curr = (1 - leak) * X_curr + leak * np.tanh(pre_act)

        # readout: use both reservoir state and input to predict next [x, y, z]
        features = np.concatenate([X_curr, u_curr])
        y_norm = features @ W_out
        y_un = y_norm * std + mean  # undo normalization to get real coordinates

        if fwd_time is None: fwd_time = time.time() - start

        preds.append(y_un)
        # Feed back the normalized prediction, not the denormalized one
    
        u_curr = y_norm

    return np.array(preds), fwd_time

pred_traj, fwd_time = forecast_esn(initial_state, initial_input, FORECAST_STEPS)
true_traj = targets[START_IDX : START_IDX + FORECAST_STEPS]

print(f"Inference Time: {fwd_time:.6f}s")
print(f"Best Params used: {model_data['params']}")

errors = np.linalg.norm(pred_traj - true_traj, axis=1)

rmse = np.sqrt(np.mean(errors**2))
print(f"Multi-step RMSE: {rmse:.4f}")

lambda_max = 0.9
DT = 0.01
t_lyap = 1 / (lambda_max * DT)

# how many steps before the forecast drifts too far off (error crosses 2.0)
div_idx = np.argmax(errors > 2.0)
if div_idx == 0 and errors[0] < 2.0: div_idx = FORECAST_STEPS

print("\n========== LYAPUNOV ANALYSIS ==========")
print(f"Stable prediction horizon: {div_idx} steps")
print(f"In Lyapunov times: {div_idx / t_lyap:.2f} * t_lambda")

# fit a line to log(error) vs step to see how fast predictions blow up
mask = errors > 1e-6  # skip near-zero so log() doesn't explode
t_valid = np.where(mask)[0]
rate = np.polyfit(t_valid, np.log(errors[mask]), 1)[0]  # slope = how fast error grows

print(f"\n========== ERROR ACCUMULATION ==========")
print(f"Exponential error growth rate = {rate:.4f} per step")
print(f"Equivalent Lyapunov-like exponent = {rate:.4f}")

np.save(PROJECT_ROOT / "models" / "esn_w1_error_curve.npy", errors)

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

t_steps = np.arange(FORECAST_STEPS)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ['$x(t)$', '$y(t)$', '$z(t)$']
for i, ax in enumerate(axes):
    ax.plot(t_steps, true_traj[:, i], color="#1f77b4", label="Ground truth")
    ax.plot(t_steps, pred_traj[:, i], "--", color="#ff7f0e", label="Predicted")
    ax.set_ylabel(labels[i], fontsize=14)
    ax.legend(loc="lower left", fontsize=12)
    ax.tick_params(labelsize=12)
axes[-1].set_xlabel("Forecast step", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "Forecast_ESN_W1_fig0.png", dpi=300)
plt.close(fig)

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(t_steps, errors, color="#1f77b4")
ax1.axhline(2.0, color="grey", ls=":", lw=0.8, label="ε = 2")
ax1.set_xlabel("Forecast step", fontsize=14)
ax1.set_ylabel("Euclidean error", fontsize=14)
ax1.legend(loc="upper left", fontsize=12)
ax1.tick_params(labelsize=12)
ax1.grid(True, linewidth=0.3)
fig1.tight_layout()
fig1.savefig(FIGURES_DIR / "Forecast_ESN_W1_fig1.png", dpi=300)
plt.close(fig1)
print("Figures saved.")

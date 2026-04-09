# ESN implementation follows Lukoševičius (2012),
# "A Practical Guide to Applying Echo State Networks"

import numpy as np
from sklearn.metrics import root_mean_squared_error
from joblib import dump
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "lorenz_windows_w10.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

WINDOW = 10
FEATURES = WINDOW * 3

print("Loading data...")
data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)

# inputs = last 10 Lorenz states flattened into one row (10*3 = 30 cols)
# targets = the next [x, y, z] after that window
inputs = data[:, :FEATURES]
targets = data[:, FEATURES:FEATURES+3]

train_size = int(len(inputs) * 0.8)


inputs_train_flat = inputs[:train_size].reshape(-1, 3)
mean = inputs_train_flat.mean(axis=0)
std = inputs_train_flat.std(axis=0)

# Reshape to (samples, window, 3), normalize, flatten back
inputs_norm = ((inputs.reshape(len(inputs), WINDOW, 3) - mean) / std).reshape(len(inputs), FEATURES)
targets_norm = (targets - mean) / std

inputs_train, inputs_val = inputs_norm[:train_size], inputs_norm[train_size:]
targets_train, targets_val = targets_norm[:train_size], targets_norm[train_size:]

print(f"Train size: {len(inputs_train)}, Val size: {len(inputs_val)}")


def train_esn(u_train, y_train, params, rng_seed=42):
    rng = np.random.default_rng(rng_seed)

    N_res = 1000
    density = 0.03 # from Jaeger
    leak = params['leak']
    spec_rad = params['spec_rad']
    alpha = params['alpha']
    # Had to lower this from 1.0, with 30D input the tanh was saturating
    input_scale = 0.1

    W = rng.standard_normal((N_res, N_res))
    mask = rng.random((N_res, N_res)) < density
    W *= mask

    eigvals = np.linalg.eigvals(W)
    W *= spec_rad / np.max(np.abs(eigvals))

    W_in = input_scale * rng.standard_normal((N_res, u_train.shape[1]))

    X_res = np.zeros(N_res)
    states = []

    for t in range(len(u_train)):
        u = u_train[t]
        # push input + recurrent activity through tanh, then mix with old state
        # leak = how much new vs old info to keep (high leak = forgets faster)
        pre_act = W_in @ u + W @ X_res
        X_res = (1 - leak) * X_res + leak * np.tanh(pre_act)
        # tack the raw input onto the reservoir state — gives the readout direct access
        # to the input, not just what the reservoir made of it
        states.append(np.concatenate([X_res, u]))

    states = np.array(states)

    # Tried adding squared states (Lukoševičius 2012)
    # Didn't help, just slower
    # states = np.hstack([states, states**2])

    # ridge regression in closed form — solves for the readout weights W_out
    # alpha * I is the regularization term to stop it from overfitting
    n_features = states.shape[1]
    reg_term = alpha * np.eye(n_features)
    W_out = np.linalg.solve(states.T @ states + reg_term, states.T @ y_train)

    return W, W_in, W_out


def evaluate_esn(u_val, y_val, W, W_in, W_out, leak):
    X_res = np.zeros(W.shape[0])
    preds = []

    for t in range(len(u_val)):
        u = u_val[t]
        pre_act = W_in @ u + W @ X_res
        X_res = (1 - leak) * X_res + leak * np.tanh(pre_act)

        features = np.concatenate([X_res, u])
        y_hat = features @ W_out
        preds.append(y_hat)

    return root_mean_squared_error(y_val, preds)


param_grid = {
    'leak': [0.1, 0.3, 0.5, 0.7, 0.9],
    'spec_rad': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    'alpha': [1e-2, 1e-4, 1e-6, 1e-8]
}

best_score = float('inf')
best_model = None
best_params = None
n_trials = 20

print(f"\nStarting Random Search ({n_trials} trials)...")
start_time = time.process_time()

rng_search = np.random.default_rng(123)

for i in range(n_trials):
    params = {
        'leak': rng_search.choice(param_grid['leak']),
        'spec_rad': rng_search.choice(param_grid['spec_rad']),
        'alpha': rng_search.choice(param_grid['alpha'])
    }

    W, W_in, W_out = train_esn(inputs_train, targets_train, params)
    rmse = evaluate_esn(inputs_val, targets_val, W, W_in, W_out, params['leak'])

    print(f"Trial {i+1}/{n_trials} | Params: {params} | Val RMSE: {rmse:.4f}")

    if rmse < best_score:
        best_score = rmse
        best_params = params
        best_model = {
            'W': W,
            'W_in': W_in,
            'W_out': W_out,
            'params': params,
            'mean': mean,
            'std': std
        }

elapsed = time.process_time() - start_time
mins, secs = divmod(elapsed, 60)
print(f"\nSearch complete (CPU time: {int(mins)}m {secs:.0f}s)")
print(f"Best RMSE: {best_score:.4f}")
print(f"Best Params: {best_params}")

save_path = MODELS_DIR / "best_esn_w10.pkl"
dump(best_model, save_path)
print(f"Saved best ESN model to: {save_path}")

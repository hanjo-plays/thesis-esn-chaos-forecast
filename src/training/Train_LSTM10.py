import random
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os
# force CPU — kept getting OOM errors when keras grabbed the GPU alongside other runs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump
import keras
from keras import layers
import keras_tuner as kt

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "lorenz_windows_w10.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

WINDOW = 10

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

feature_col = [f"{axis}_t{i}" for i in range(WINDOW) for axis in ("x", "y", "z")]
target_col = ["x_next", "y_next", "z_next"]

X = df[feature_col].values
y = df[target_col].values

# LSTM needs 3D: (samples, timesteps, features) — with W=10 it actually gets a sequence to work with
X_seq = X.reshape(len(X), WINDOW, 3)

X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y, test_size=0.2, shuffle=False
)

# StandardScaler needs 2D, so flatten to (N*10, 3), scale, then reshape back to 3D for the LSTM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 3)).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, 3)).reshape(X_val.shape)

dump(scaler, MODELS_DIR / "scaler_lstm_w10.pkl")
print("Scaler saved.")

# hp lets keras-tuner swap in different values each trial to find the best combo
def build_lstm(hp):
    # LSTM layer -> dropout -> dense hidden -> dropout -> 3 outputs (x, y, z)
    model = keras.Sequential([
        layers.Input(shape=(WINDOW, 3), name="input_layer"),
        layers.LSTM(
            hp.Int("lstm_units", min_value=32, max_value=256, step=32),
            return_sequences=False,
            name="lstm_layer"
        ),
        layers.Dropout(hp.Choice("dropout_1", values=[0.0, 0.1, 0.2, 0.3])),
        layers.Dense(
            hp.Int("dense_units", min_value=32, max_value=512, step=32),
            activation="relu", name="hidden_dense"
        ),
        layers.Dropout(hp.Choice("dropout_2", values=[0.0, 0.1, 0.2, 0.3])),
        layers.Dense(3, name="output_layer")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("lr", 1e-4, 1e-2, sampling="log")
        ),
        loss="mse",
        metrics=["mae"]
    )
    return model

tuner = kt.BayesianOptimization(
    build_lstm,
    objective="val_loss",
    max_trials=20,
    overwrite=True,
    directory=tempfile.mkdtemp(),
    project_name="LSTM_w10",
)

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

print("Starting Hyperparameter Search (LSTM W=10)...")
start_cpu = time.process_time()
tuner.search(
    X_train_scaled, y_train,
    epochs=150,
    validation_data=(X_val_scaled, y_val),
    batch_size=32,
    callbacks=[es],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]

y_pred = best_model.predict(X_val_scaled, verbose=0)
rmse = root_mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

elapsed = time.process_time() - start_cpu
mins, secs = divmod(elapsed, 60)
print(f"\nTraining Complete (LSTM W=10). (CPU time: {int(mins)}m {secs:.0f}s)")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²:   {r2:.4f}")

save_path = MODELS_DIR / "best_lstm_w10.h5"
best_model.save(save_path)
print(f"Best model saved to: {save_path}")

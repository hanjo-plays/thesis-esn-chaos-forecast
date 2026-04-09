import random
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os

# Kept getting OOM errors when keras tried to grab the GPU alongside other runs
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
DATA_FILE = PROJECT_ROOT / "data" / "lorenz_windows_w1.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_FILE)

X = df[["x_t0", "y_t0", "z_t0"]].values
y = df[["x_next", "y_next", "z_next"]].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

dump(scaler, MODELS_DIR / "scaler_ffnn_w1.pkl")
print("Scaler saved.")

# hp lets keras-tuner swap in different values each trial to find the best combo
def build_ffnn(hp):
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(
            hp.Int("units_1", 32, 512, step=32),
            activation="relu"
        ),
        layers.Dropout(hp.Choice("drop_1", [0.0, 0.1, 0.2, 0.3])),
        layers.Dense(
            hp.Int("units_2", 32, 512, step=32),
            activation="relu"
        ),
        layers.Dropout(hp.Choice("drop_2", [0.0, 0.1, 0.2, 0.3])),
        layers.Dense(3)
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
    build_ffnn,
    objective="val_loss",
    max_trials=20,
    overwrite=True,
    directory=tempfile.mkdtemp(),
    project_name="FFNN_w1",
)

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

print("Starting Hyperparameter Search...")
start_cpu = time.process_time()
tuner.search(
    X_train, y_train,
    epochs=150,
    validation_data=(X_val, y_val),
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]

y_hat = best_model.predict(X_val, verbose=0)
rmse = root_mean_squared_error(y_val, y_hat)
r2 = r2_score(y_val, y_hat)

elapsed = time.process_time() - start_cpu
mins, secs = divmod(elapsed, 60)
print(f"\nTraining Complete. (CPU time: {int(mins)}m {secs:.0f}s)")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²:   {r2:.4f}")

save_path = MODELS_DIR / "best_ffnn_w1.h5"
best_model.save(save_path)
print(f"Best model saved to: {save_path}")

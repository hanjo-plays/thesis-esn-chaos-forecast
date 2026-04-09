# Chaotic Time Series Prediction Using Reservoir Computing

This is the code for my thesis. It compares three neural network architectures
(FFNN, LSTM, ESN) on forecasting the Lorenz attractor, which is a classic
chaotic system where small perturbations blow up exponentially over time.

The main question is whether reservoir computing (ESN) can match or beat
conventional deep learning on this kind of task, and how the input window size
affects prediction stability.

## Structure

`src/data/generate_lorenz.py` produces the raw trajectory and the windowed CSVs
(W=1 and W=10) that all models train on. The training and forecasting scripts
are in `src/training/` and `src/forecasting/` - one file per model/window combo,
named like `Train_ESN_W10.py` or `Forecast_LSTM1.py`. Trained models go in
`models/` (.h5 for keras, .pkl for ESN and scalers), and the forecast scripts
also save error curves there as .npy files.

`src/utils/` has the batch runners (`run_all.py`, `run_forecasts.py`) and a
couple of analysis scripts. `src/visualization/` is just the diagram generators
for the thesis figures. Everything in `docs/figures/` is auto-generated py the scripts.

## How to run

Install dependencies:

```bash
conda env create -f environment.yml
conda activate thesis
```

Generate data and train everything:

```bash
python src/utils/run_all.py
```

Run all forecasts (saves figures to docs/figures/):

```bash
python src/utils/run_forecasts.py
```

Compare models:

```bash
python src/utils/compare.py
```

You can also run individual scripts, e.g.
`python src/training/Train_ESN_W10.py`.

## Models

**FFNN** - Two hidden layers with dropout. Flattened window as input. Nothing
fancy, just a baseline.

**LSTM** - Single LSTM layer + dense head. For W=1 it only sees one timestep so
it's basically just learning a nonlinear map; for W=10 it actually gets to use
the sequence.

**ESN** - 1000-neuron reservoir with sparse random connections. Only the readout
weights are trained (ridge regression), everything else is fixed. Had to lower
`input_scale` to 0.1 for the W=10 case because the tanh was saturating with 30
input dimensions.

All keras models use Bayesian hyperparameter search via keras-tuner (20 trials,
early stopping with patience=20).

## Requirements

Python 3.10, TensorFlow 2.10, scikit-learn, keras-tuner, dysts, matplotlib,
joblib. See `environment.yml`.

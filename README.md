# Urban Traffic Spatio-Temporal Forecasting

A research project for short-term road-traffic speed forecasting with graph neural networks and temporal models. It includes two Kaggle-oriented Jupyter notebooks and a lightweight, static results dashboard.

The notebooks cover two related but distinct experiments:

1. A comparison of four models on PEMS-BAY, followed by cross-network generalization experiments on METR-LA.
2. A GNN–Transformer model evaluated on PEMS-BAY and transferred to METR-LA using three adaptation strategies.


## Contents

- [Project contents](#project-contents)
- [Datasets](#datasets)
- [Methods](#methods)
- [Reported results](#reported-results)
- [Requirements](#requirements)
- [Running the notebooks](#running-the-notebooks)
- [Opening the dashboard](#opening-the-dashboard)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Data, licensing, and citation](#data-licensing-and-citation)

## Project contents

| File | Description |
|---|---|
| `CODE1_PEMS-BAY — Multi-modèles + Généralisation METR.ipynb` | Trains and evaluates LSTM, GNN, ST-GNN, and Hybrid LSTM–GNN models on PEMS-BAY, then explores transfer/generalization to METR-LA. |
| `CODE2_GNN__Transformer+Généralisation.ipynb` | Defines and evaluates the GNN–Transformer model, including PEMS-BAY testing and three METR-LA adaptation strategies. |
| `index.html` | Self-contained dashboard for browsing the GNN–Transformer metrics and comparing the three METR-LA strategies. It uses embedded data. |

## Datasets

The notebooks use the **PEMS-BAY** and **METR-LA** traffic speed datasets. Each contains time series from road sensors and an associated road-network adjacency matrix.

- **PEMS-BAY:** 325 sensors in the San Francisco Bay Area.
- **METR-LA:** 207 sensors in the Los Angeles area.
- The time series are sampled at five-minute intervals.

The notebooks expect the datasets to be attached as Kaggle inputs. In particular, the GNN–Transformer notebook uses these paths:

```text
/kaggle/input/datasets/scchuy/pemsbay/pems-bay.h5
/kaggle/input/datasets/scchuy/pemsbay/adj_mx_bay.pkl
/kaggle/input/datasets/annnnguyen/metr-la-dataset/METR-LA.h5
/kaggle/input/datasets/annnnguyen/metr-la-dataset/adj_METR-LA.pkl
```

The multi-model notebook also uses Kaggle-specific input paths, including:

```text
/kaggle/input/PEMS-BAY/pems-bay.h5
/kaggle/input/PEMS-BAY/adj_mx_bay.pkl
```

If your Kaggle dataset attachment creates different directory names, update the corresponding path constants near the start of the notebook. The dataset files are not included in this repository.

## Methods

### Notebook 1: multi-model PEMS-BAY experiment

This notebook compares four architectures:

- **LSTM** — recurrent sequence model with attention.
- **GNN** — graph convolutional network.
- **ST-GNN** — spatio-temporal graph neural network.
- **Hybrid LSTM–GNN** — combines graph and recurrent components with attention.

It uses a 12-step input window (one hour) and a 3-step forecast horizon (15 minutes). Hyperparameters are tuned with Optuna before final model training and test-set evaluation.

### Notebook 2: GNN–Transformer experiment

This notebook combines:

- graph neural network layers for spatial relationships between sensors;
- Transformer blocks for temporal dependencies; and
- an adaptive attention component.

It uses a 12-step input window and a 12-step forecast horizon (one hour). The METR-LA experiment compares:

| Strategy | Description |
|---|---|
| **A — Zero-shot + BatchNorm recalibration** | Transfers the trained model and recalibrates BatchNorm statistics on target-domain data without gradient-based weight updates. |
| **B — Decoder fine-tuning** | Fine-tunes selected decoder-related layers on METR-LA while keeping the other model layers frozen. |
| **C — Full retraining** | Trains the transferred model on METR-LA. This is a target-domain reference/upper-bound experiment, not zero-shot transfer. |

The dashboard displays the three strategies together in its chart and detailed-results table. The KPI cards use the full-retraining (C) result by default on METR-LA.

## Reported results

The following values are taken from the saved test-set outputs in the notebooks. Re-running training may produce different results.

### GNN–Transformer notebook

| Dataset / strategy | MAE (mph) ↓ | RMSE (mph) ↓ | MAPE (%) ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| PEMS-BAY test set | 4.260968 | 7.494622 | 10.03 | 0.3983 |
| METR-LA — A: zero-shot + BatchNorm recalibration | 15.303073 | 21.782532 | 32.52 | 0.0867 |
| METR-LA — B: decoder fine-tuning | 12.203102 | 18.461308 | 29.46 | 0.3440 |
| METR-LA — C: full retraining | 10.356481 | 17.384211 | 24.99 | 0.4183 |

### Multi-model notebook: PEMS-BAY test set

| Model | MAE (mph) ↓ | RMSE (mph) ↓ | MAPE (%) ↓ |
|---|---:|---:|---:|
| LSTM | 3.9189 | 4.9452 | 10.44 |
| GNN | 4.0703 | 5.1306 | 10.94 |
| ST-GNN | 4.0632 | 5.1385 | 10.66 |
| Hybrid LSTM–GNN | 12.1384 | 15.3554 | 40.25 |


## Requirements

Both notebooks use Python 3.12 kernels and are written for a Kaggle notebook environment. Their setup cells install `torch-geometric` and `optuna`; Kaggle supplies or manages the remaining scientific Python packages in the original workflow.

The code imports packages including:

- PyTorch and PyTorch Geometric
- NumPy, pandas, and h5py
- Optuna
- Matplotlib and Seaborn
- tqdm
- NetworkX (used in the GNN–Transformer notebook)

For a local environment, install PyTorch and PyTorch Geometric versions compatible with your operating system, Python version, and CPU/GPU build before installing the remaining packages.

## Running the notebooks

### Kaggle (recommended)

1. Upload or import this repository's notebooks into Kaggle.
2. Attach the PEMS-BAY dataset. Attach METR-LA as well to run the cross-network experiments.
3. Confirm that the attached dataset paths match the constants near the top of the selected notebook; update them if necessary.
4. Select a suitable accelerator in the notebook settings. A GPU is recommended for training.
5. Run the notebook from top to bottom so that data loading, model definitions, training, and evaluation execute in order.
6. Review the saved output files in the result directory configured by that notebook.

The notebooks use a fast Optuna/training configuration by default, with limited trials, epochs, or batches in some stages. These settings are useful for a demonstration but are not a substitute for a fully tuned experimental run. Increase them deliberately when conducting a more extensive evaluation.

### Generated outputs

Depending on which cells are run, the notebooks may write artifacts such as:

- `best_hyperparameters.json` — selected hyperparameters;
- `best_model.pth` — a trained model checkpoint;
- `results_pems_bay.json` and `final_report.json` — evaluation summaries; and
- `generalization_comparison.png` — a comparison plot.



### Local Jupyter environment

The notebooks can be opened locally, but they are not fully environment-independent. You will need to:

1. create a Python environment with compatible PyTorch/PyTorch Geometric and scientific Python packages;
2. download or otherwise obtain the datasets under their respective terms;
3. change the Kaggle-specific input paths to paths on your machine; and
4. select a compute device and training configuration suitable for your hardware.


## Opening the dashboard

The dashboard is plain HTML, CSS, and JavaScript with no external runtime dependencies.

- Open `index.html` directly in a modern browser; or
- serve the repository directory locally:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000`.

To publish the dashboard with GitHub Pages, configure Pages to deploy from the repository root (or move the site to the publishing directory selected for the project). The page is static: its metrics are embedded in `index.html` and are not fetched from the notebooks at runtime.



# 2025AB05088 — RNN vs Transformer for Time Series

## Student
- BITS ID: 2025AB05088
- Name: P L V S ADITHYA
- Email: 2025ab05088@wilp.bits-pilani.ac.in
- Notebook: `2025AB05088_rnn_assignment.ipynb`

## Overview
This repository contains a lab assignment comparing a stacked LSTM (RNN) and a Transformer encoder on a univariate time series forecasting task (TCS stock close price). The goal is to implement, train, and compare both architectures on the same temporal train/test split and report standard forecasting metrics.

Key points:
- Framework used: PyTorch
- Dataset: `tcs_stock_data.csv` (Close price used)
- Sequence length (lookback): 30
- Prediction horizon: 1
- Temporal train/test split: 90/10 (no shuffling)
- Positional encoding: sinusoidal positional encoding is implemented and applied to the Transformer input

## Files in this folder
- `2025AB05088_rnn_assignment.ipynb` — main Jupyter notebook (complete implementation, training, evaluation, visualizations, and final JSON results for auto-grader)
- `tcs_stock_data.csv` — stock price data used for the assignment

## Models implemented
1. LSTM (stacked)
   - Two stacked LSTM layers
   - Dropout between layers
   - Fully-connected output projection
   - Loss: MSE
   - Optimizer: Adam

2. Transformer Encoder
   - Input projection to d_model
   - Sinusoidal positional encoding added to inputs
   - PyTorch's `nn.TransformerEncoder` with multi-head attention
   - Global average pooling over sequence dimension
   - Loss: MSE
   - Optimizer: Adam

## Representative Results (from notebook run)
These numbers are the values produced when the notebook was executed (they are printed in the notebook and included in the assignment's JSON summary):

- LSTM (RNN)
  - MAE: 115.04
  - RMSE: 136.66
  - MAPE: 2.92%
  - R²: 0.5910
  - Training time: ≈ 8.75 s
  - Parameters: 29,729

- Transformer
  - MAE: 131.73
  - RMSE: 162.76
  - MAPE: 3.33%
  - R²: 0.4199
  - Training time: ≈ 59.18 s
  - Parameters: 100,161

Notes: Results depend on exact environment (CPU/GPU), PyTorch version, and randomness. The notebook sets random seeds for reproducibility but runtime differences may still appear across systems.

## Dependencies
The notebook requires the following Python packages (typical install via pip):

- python 3.8+ (recommended)
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- torch (PyTorch)

A minimal requirements install command:

```bash
# Windows (cmd.exe)
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch
```

If you use a GPU, install the PyTorch build matching your CUDA version as described on https://pytorch.org/.

## Reproduce / Run the Notebook
1. Make sure `tcs_stock_data.csv` is placed in the same folder as the notebook.
2. Open `2025AB05088_rnn_assignment.ipynb` in Jupyter Notebook / JupyterLab / VS Code and select the Python environment with the required dependencies.
3. Restart kernel and Run → Restart & Run All. The notebook executes end-to-end and prints an assignment results JSON at the end (used by the auto-grader).

Optional: run the notebook headlessly and convert to HTML (requires nbconvert):

```bash
# Example: convert to HTML after executing (requires nbconvert and jupyter)
jupyter nbconvert --to html --execute "2025AB05088_rnn_assignment.ipynb" --ExecutePreprocessor.timeout=600
```

## What the notebook produces
- Training loss curves for both models
- Actual vs predicted plots and residual plots
- Four evaluation metrics for each model: MAE, RMSE, MAPE, R²
- Training time and parameter counts
- A final JSON dictionary named `assignment_results` containing dataset info, model configs, metrics, and textual analysis — printed at the end of the notebook for automatic grading

## Notes for graders / reviewers
- The notebook follows the assignment constraints: temporal split (no shuffling), positional encoding added to Transformer, stacked recurrent layers for the LSTM, and metrics calculation for both models.
- Please run Kernel → Restart & Run All to ensure all outputs are visible.

## Contact
If you need clarifications, contact:
- P L V S ADITHYA — 2025ab05088@wilp.bits-pilani.ac.in

## License
This material is provided for coursework submission. You may copy or adapt it for educational purposes. No warranty is provided.

---
Generated from the corresponding Jupyter notebook `2025AB05088_rnn_assignment.ipynb` on the local workspace.

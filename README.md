# RNN vs Transformer for Time Series Forecasting
<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

<br>

## Student Details
| Detail | Information |
|--------|-------------|
| **BITS ID** | `2025AB05088` |
| **Name** | P L V S ADITHYA |
| **Email** | [2025ab05088@wilp.bits-pilani.ac.in](mailto:2025ab05088@wilp.bits-pilani.ac.in) |
| **Notebook** | [`2025AB05088_rnn_assignment.ipynb`](./2025AB05088_rnn_assignment.ipynb) |


## Overview
This repository contains a lab assignment comparing a **Stacked LSTM (RNN)** and a **Transformer Encoder** on a univariate time series forecasting task. The goal is to implement, train, and compare both architectures on the same temporal train/test split and report standard forecasting metrics.

### Assignment Configuration
| Parameter | Specification |
|-----------|---------------|
| **Framework** | PyTorch |
| **Dataset** | `tcs_stock_data.csv` (Close price) |
| **Sequence Length** | `30` (lookback) |
| **Prediction Horizon**| `1` |
| **Train/Test Split** | `90/10` (Temporal, no shuffling) |
| **Positional Encoding**| Sinusoidal (Applied to Transformer) |


## Repository Structure
```text
.
├── 2025AB05088_rnn_assignment.ipynb   # Main Jupyter notebook
└── tcs_stock_data.csv                 # Stock price dataset
```

## Models Implemented

### 1. LSTM (Stacked)
- Two stacked LSTM layers with dropout between them.
- Fully-connected output projection.
- **Loss Function:** MSE
- **Optimizer:** Adam

### 2. Transformer Encoder
- Input projection to `d_model`.
- Sinusoidal positional encoding added to inputs.
- PyTorch's `nn.TransformerEncoder` with multi-head attention.
- Global average pooling over sequence dimension.
- **Loss Function:** MSE
- **Optimizer:** Adam


## Representative Results
> **Note:** These metrics reflect the values produced during the most recent execution. Results may vary slightly depending on hardware environment (CPU/GPU), library versions, and inherent randomness.

| Metric | LSTM (RNN) | Transformer |
|--------|------------|-------------|
| **MAE** | `115.04` | `131.73` |
| **RMSE** | `136.66` | `162.76` |
| **MAPE** | `2.92%` | `3.33%` |
| **R² Score** | `0.5910` | `0.4199` |
| **Training Time** | `≈ 8.75 s` | `≈ 59.18 s` |
| **Parameters** | `29,729` | `100,161` |


## Dependencies
The implementation requires `python 3.8+` and the following core packages:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`

**Installation Command:**
```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install required dependencies
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch
```
> For GPU acceleration, please install the PyTorch build tailored to your specific CUDA version as detailed on the [Official PyTorch Website](https://pytorch.org/).


## Execution Guide
1. Ensure the dataset `tcs_stock_data.csv` is located in the same directory as the notebook.
2. Open `2025AB05088_rnn_assignment.ipynb` within your preferred environment (Jupyter / VS Code) utilizing the configured Python kernel.
3. Select **Restart Kernel and Run All**. The notebook is designed to execute end-to-end and will output a structured JSON dictionary for the auto-grader upon completion.

**Optional: Headless Execution & HTML Conversion**
```bash
# Export executed notebook to HTML (requires nbconvert)
jupyter nbconvert --to html --execute "2025AB05088_rnn_assignment.ipynb" --ExecutePreprocessor.timeout=600
```


## System Outputs
- **Visualizations:** Training loss curves, Actual vs. Predicted values, and Residual plots.
- **Performance Metrics:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Mean Absolute Percentage Error (MAPE), and R² Score.
- **Diagnostics:** Iteration training time and total trainable parameters.
- **Final Output:** A JSON dictionary (`assignment_results`) aggregating dataset parameters, model configurations, evaluation metrics, and analytical summaries.


## Reviewer Notes
> The implementation strictly adheres to the mandated assignment constraints, specifically the temporal split methodology (without shuffling), application of sinusoidal positional encoding for the Transformer variant, and utilization of stacked recurrent layers for the LSTM model. Please ensure **Kernel → Restart & Run All** is executed to properly render all intended outputs.


## Support
For technical clarifications regarding this implementation, please contact:
- **P L V S ADITHYA** - [2025ab05088@wilp.bits-pilani.ac.in](mailto:2025ab05088@wilp.bits-pilani.ac.in)


## License
This repository and its contents are provided strictly for coursework submission and academic evaluation. Unauthorized commercial use is prohibited.

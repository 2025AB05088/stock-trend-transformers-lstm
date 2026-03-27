# Architectural Evaluation: Recurrent vs. Attention Mechanisms for Financial Time Series Forecasting

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Azure Machine Learning](https://img.shields.io/badge/Azure_ML-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)

</div>

<br>

## Executive Summary
This repository contains an architectural evaluation and experimentation pipeline comparing a **Deep Stacked LSTM (RNN)** against a **Transformer Encoder** topology. The primary objective is to evaluate latency, parameter efficiency, and predictive accuracy on a univariate financial time series forecasting workload (TCS equity close prices) prior to production deployment and integration into our Azure Machine Learning (AML) model registry.

### Experiment Configuration
| Parameter | Specification |
|-----------|---------------|
| **Core Framework** | PyTorch (Optimized for Azure NV-series VMs) |
| **Ingestion Source** | `tcs_stock_data.csv` (Standardized Close price features) |
| **Lookback Window** | `30` (Temporal sequence length) |
| **Forecast Horizon**| `1` (Next-step projection) |
| **Validation Strategy**| `90/10` (Sequential hold-out; zero data leakage) |
| **Positional Encoding**| Sinusoidal mapping (Transformer topology only) |


## Repository Artifacts
```text
.
├── 2025AB05088_rnn_assignment.ipynb   # Main Experimentation & Training Pipeline
└── tcs_stock_data.csv                 # Raw historical telemetry data
```

## Evaluated Topologies

### 1. Prototype Alpha: Deep Stacked LSTM
- Multi-layer Long Short-Term Memory (LSTM) network equipped with inter-layer dropout for regularization.
- Fully-connected dense layer for final output projection.
- **Loss Optimization:** Mean Squared Error (MSE)
- **Gradient Optimization:** Adam (Adaptive Moment Estimation)

### 2. Prototype Beta: Transformer Encoder
- Linear feature projection to high-dimensional embedding space (`d_model`).
- Sinusoidal temporal embedding to inject sequence awareness.
- Native PyTorch `nn.TransformerEncoder` backbone via Multi-Head Attention mechanisms.
- Global average pooling across the temporal dimension for fixed-length vector projection.
- **Loss Optimization:** Mean Squared Error (MSE)
- **Gradient Optimization:** Adam


## Performance Benchmarks & Telemetry
> **Note:** The following metrics reflect the latest benchmarking execution on local compute prior to scaling out on Azure ML clusters. Metrics are subject to marginal variance based on CUDA/cuDNN versioning and hardware stochasticity.

| Key Performance Indicator | Stacked LSTM | Transformer Encoder |
|---------------------------|--------------|---------------------|
| **Mean Absolute Error (MAE)** | `115.04` | `131.73` |
| **Root Mean Square Error (RMSE)**| `136.66` | `162.76` |
| **Mean Absolute Pct Error (MAPE)**| `2.92%` | `3.33%` |
| **R² Explanatory Power** | `0.5910` | `0.4199` |
| **Iteration Latency (Training)** | `≈ 8.75 s` | `≈ 59.18 s` |
| **Model Footprint (Parameters)** | `29,729` | `100,161` |

**Architectural Assessment:** The Stacked LSTM currently yields superior sample efficiency and lower latency, achieving a highly optimal fit (MAPE: ~2.9%) within a fraction of the compute footprint compared to the Transformer model.


## Environment Setup & Initialization
This pipeline requires a `python 3.8+` environment configured with the fundamental data science and deep learning stacks:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`

**Local Environment / Azure Compute Instance Bootstrap:**
```bash
# Ensure package manager is updated
python -m pip install --upgrade pip

# Provision foundational dependencies
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch
```
> For execution on Azure NC/ND-Series compute nodes, verify that the PyTorch build aligns with the host's CUDA drivers strictly as detailed in the [PyTorch Documentation](https://pytorch.org/).


## Execution & Deployment Pipeline
1. Verify the `tcs_stock_data.csv` artifact is staged accurately in the project root.
2. Initialize the `2025AB05088_rnn_assignment.ipynb` experimentation pipeline via JupyterLab or VS Code using the provisioned virtual environment.
3. Execute the full pipeline synchronously (`Restart Kernel and Run All`). Upon normal completion, the pipeline automatically dumps a structured JSON payload representing the model metadata, topology hyperparameters, and final metrics (primed for Azure ML flow tracking).

**Batch Execution / Headless Rendering**
```bash
# Pre-compile the pipeline to an HTML operational report
jupyter nbconvert --to html --execute "2025AB05088_rnn_assignment.ipynb" --ExecutePreprocessor.timeout=600
```


## Output Artifacts & Observability
- **Diagnostic Visualizations:** Epoch-wise training decay curves, empirical vs. forecasted trajectories, and discrete residual density mapping.
- **Validation Metrics:** Standardized quantitative metrics (MAE, RMSE, MAPE, R²).
- **System Telemetry:** Captured iteration execution latency and granular parameter counts per topology.
- **Payload Generation:** An aggregated JSON manifest (`assignment_results`) encapsulating the run configuration, suitable for automated CI/CD pipeline ingestion or Model Registry tagging.


## Engineering Notes
> This repository strictly mandates chronological validation splits to simulate real-world streaming data (preventing target leakage). Both models adhere identically to the preprocessing pipeline, ensuring a level playing field for the benchmark.

## Maintainers
- **P L V S ADITHYA** - Lead Architect / MLOps Engineer
- [2025ab05088@wilp.bits-pilani.ac.in](mailto:2025ab05088@wilp.bits-pilani.ac.in)

## License
Proprietary evaluation codebase. Internal distribution only.

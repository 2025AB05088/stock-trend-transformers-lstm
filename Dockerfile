# ==============================================================================
# Enterprise MLOps Pipeline Container Specification
# Target Environments: Azure Machine Learning Compute, AKS (Azure Kubernetes Service)
# Optimized for: NVIDIA NV-series / NC-series VMs via CUDA 11.8
# Author: Lead Architect / MLOps Engineer
# ==============================================================================

# Stage 1: Base Runtime Layer
# Sourcing official Microsoft Azure ML container registry for optimal native compute integration
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS runtime

# Inject architectural identity and build provenance
LABEL maintainer="MLOps Platform Engineering <2025ab05088@wilp.bits-pilani.ac.in>" \
      com.azure.ml.ops.tier="production-grade" \
      com.azure.ml.ops.pipeline="time-series-forecasting"

# Define immutable environment invariants for deterministic execution
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/miniconda/bin:$PATH"

# ==============================================================================
# OS Foundation & Security Hardening
# ==============================================================================
USER root

# Mitigate vulnerabilities, purge bloat, and establish foundational build tools
RUN apt-get update -y && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Model Environment Initialization
# ==============================================================================
# Establish distinct staging volume for computational artifacts
WORKDIR /workspace/mlops_experiment

# Initialize dependency cache layer (optimized for Docker layer caching)
# Note: Upgrading core pip binaries securely prior to library ingestion
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core forecasting topologies and headless operational execution dependencies
# (In an actual production pipeline, these would be securely pinned via a requirements.txt lockfile)
RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    torch \
    jupyter \
    nbconvert \
    papermill

# ==============================================================================
# Artifact Staging
# ==============================================================================
# Transfer the evaluation telemetry dataset and topology simulation scripts
COPY tcs_stock_data.csv ./
COPY 2025AB05088_rnn_assignment.ipynb ./

# ==============================================================================
# Privilege Dropping & Container Telemetry
# ==============================================================================
# Instantiate standard non-privileged user profile for hardened deployment architectures
RUN useradd -m -s /bin/bash mlops_sys \
    && chown -R mlops_sys:mlops_sys /workspace/mlops_experiment

USER mlops_sys

# Surface telemetry standard output port (optional, utilized by local Jupyter spin-ups)
EXPOSE 8888

# ==============================================================================
# Pipeline Entrypoint Activation
# ==============================================================================
# Default Execution: Headless pipeline execution generating a structured HTML trace.
# Note: In an integrated Azure ML setup, this is typically overridden by the `command:` specified in the sweeping job YAML.
ENTRYPOINT ["jupyter", "nbconvert", "--to", "html", "--execute", "2025AB05088_rnn_assignment.ipynb", "--ExecutePreprocessor.timeout=3600"]

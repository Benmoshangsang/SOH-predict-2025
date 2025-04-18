# SOH Prediction and Component Detection for Lithium-ion Batteries

This repository contains the source code for lithium-ion battery State-of-Health (SOH) prediction based on a customized deep learning pipeline. The implementation uses Python 3.11.5 and TensorFlow.

## Project Structure

The codebase is organized by functionality across multiple folders. Each folder contains a specific set of related modules:

- `main/`: Contains the core functions and scripts for model training, evaluation, and prediction.
- `utils/`: Utility functions such as preprocessing, metrics, and visualizations.
- `losses/`: Custom loss functions.
- `augmentation/`: Data augmentation strategies.

> Note: Some support scripts (e.g., from `utils/`) are required to run code in `main/`. If you encounter module import errors when running outside the root directory, consider copying the necessary support files into the same working folder.

## Python Environment

- Python version: `3.11.5`
- Deep learning framework: `TensorFlow`
- Other dependencies can be installed using:

```bash
pip install -r requirements.txt

## Dataset and Results

All datasets used in this study, including the original and augmented versions, along with experimental results (e.g., baseline model, hyperparameter optimization, ablation results), are available via Zenodo:

DOI: 10.5281/zenodo.15239011

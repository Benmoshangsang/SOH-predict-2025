SOH Prediction and Component Detection for Lithium-ion Batteries
This repository contains the source code for predicting the State-of-Health (SOH) of lithium-ion batteries using a modified deep learning pipeline based on CNN-LSTM with attention mechanisms. The project also supports various experiments including hyperparameter optimization, loss function evaluation, ablation studies, and data augmentation techniques.

📁 Project Structure
The repository consists of two main folders:

Main/:
Contains separate subfolders for each experimental module:
Main function comparison experiment: Models with different main network structures
Loss function comparison experiment: Experiments comparing multiple loss functions
Hyperparameter comparison experiment: Runs using different hyperparameter configurations
Data Augmentation: Experiments using original + augmented datasets (slicing, flipping, and linear mix)
Ablation experiment: Component-wise evaluation and removal of modules to assess contribution

utils/:
Contains support scripts and shared modules such as:
param_*.py: Model parameter configurations
utils.py, scale.py: Utility functions for normalization, metrics, and preprocessing
param_V_CNN_C_LSTM.py: Main model architecture script (CNN + LSTM + Attention)

🔁 If any script from Main/ cannot run due to missing modules, copy the required file(s) from utils/ into the same working directory.
🐍 Environment Setup
Python version: 3.11.5
Framework: TensorFlow
Recommended environment manager: Anaconda
Install dependencies via:

pip install -r requirements.txt
📊 Dataset & Results
All datasets and experiment results have been deposited to Zenodo. This includes:
Raw NASA battery datasets
Augmented datasets (generated using slicing, flipping, and mix techniques)
SOH prediction results for all experiments
Baseline vs enhanced model comparisons
Hyperparameter tuning logs
Ablation study output
📥 DOI: 10.5281/zenodo.15239011

🚨 All new datasets were created using scripts in the repository.

🔧 How to Run
Each experiment in the Main/ folder can be run independently by executing the corresponding .py file. Example:
python "Main/Main function comparison experiment/run_main_model.py"
Make sure required utility files from utils/ are either in the same directory or properly imported.

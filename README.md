# 🔋 SOH Prediction and Component Detection for Lithium-ion Batteries

This repository provides source code for predicting the **State-of-Health (SOH)** of lithium-ion batteries using a **customized CNN-LSTM-Attention** deep learning architecture. The project supports multiple experiments, including:

- ✅ Baseline & enhanced model evaluation  
- ✅ Loss function comparison  
- ✅ Hyperparameter optimization  
- ✅ Data augmentation  
- ✅ Ablation studies  

---

## 📁 Project Structure

```
📦 SOH-predict-2025  
├── Main/                           ← Experimental modules  
│   ├── Main function comparison experiment/  
│   ├── Loss function comparison experiment/  
│   ├── Hyperparameter comparison experiment/  
│   ├── Data Augmentation/  
│   └── Ablation experiment/  
│  
├── utils/                          ← Utility scripts  
│   ├── param_V_CNN_C_LSTM.py      ← Main model  
│   ├── utils.py, scale.py         ← Helper functions  
│   └── param_separated.py         ← Config & params  
│  
├── requirements.txt               ← Python dependencies  
└── README.md                      ← Project overview
```

> 💡 **Note**: Some scripts from `Main/` depend on utility files from `utils/`. If running standalone, copy necessary scripts to the same directory to resolve import issues.

---

## 💻 Environment Setup

- **Python version**: `3.11.5`  
- **Framework**: `TensorFlow`  
- **Recommended**: Use Anaconda

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset & Results

All datasets and result files are publicly available via **Zenodo**:

- 🗃 **Raw dataset**: NASA battery datasets  
- 🧪 **Augmented datasets**: Created using slicing, flipping, and linear mix techniques  
- 📈 **Experimental results**: Baseline vs enhanced model, loss function tests, ablation outputs  

📌 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15239011.svg)](https://doi.org/10.5281/zenodo.15239011)

---

## ▶️ Running Experiments

Each subfolder in `Main/` contains an independent experiment.

Example:

```bash
python "Main/Loss function comparison experiment/cnn-lstm-attention huber_loss.py"
```

If you encounter errors like `ModuleNotFound`, copy files from `utils/` into the current folder or adjust Python import paths.

---

## 📬 Contact

For questions, please raise an [Issue](https://github.com/Benmoshangsang/SOH-predict/issues) or email the corresponding author listed in the related publication.

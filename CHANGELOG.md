# 📋 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025-03-26

### 🎉 Initial Release

#### Added
- Full MNIST data pipeline using `tf.keras.datasets.mnist.load_data()`
- Image preprocessing: reshape `(28×28)` → `(784,)` and normalize `0–255` → `0.0–1.0`
- `MLPClassifier` with 2 hidden layers of 64 neurons each, ReLU activation, Adam optimizer
- Training with 30 epochs and verbose loss logging
- Loss curve visualization using Matplotlib
- Confusion matrix evaluation on 10,000 test samples
- Accuracy score: **97.35%**
- Custom image prediction pipeline with pixel inversion and normalization
- Google Colab support with `Open in Colab` badge
- Full project documentation in `README.md`
- MIT License

---

## [Unreleased]

### Planned
- CNN implementation for higher accuracy
- Gradio/Streamlit web interface for live digit drawing
- Model serialization with `joblib`
- Cross-validation support
- Hyperparameter tuning experiments

<div align="center">

# 🧠 Handwritten Digit Recognizer

### Recognizing handwritten digits (0–9) using a Multi-Layer Perceptron trained on the MNIST dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedadarohan-dot/Handwrittenrecognizer/blob/main/Handwrittendigitsrecognizer.ipynb)

<br/>

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" width="480" alt="MNIST Sample Digits"/>

*Sample digits from the MNIST dataset*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Custom Image Prediction](#-custom-image-prediction)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

This project builds a **handwritten digit classifier** capable of recognizing digits from **0 to 9** using a fully connected neural network (MLP) trained on the classic **MNIST dataset**.

The project demonstrates an end-to-end ML pipeline:

- Loading and preprocessing image data
- Training a neural network with `scikit-learn`'s `MLPClassifier`
- Evaluating with a confusion matrix and accuracy score
- Running inference on a **custom real-world handwritten image**

> **Accuracy achieved: ~97.35% on the MNIST test set**

---

## 🎬 Demo

```
Input: Custom handwritten image of digit "5"
Model Prediction: [5] ✅
```

The model processes a 28×28 grayscale image, normalizes the pixel values,
and outputs the predicted digit in real time.

---

## 🛠 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| NumPy | 2.0.2 | Array operations |
| Pillow (PIL) | 11.3.0 | Image loading & processing |
| Scikit-learn | 1.6.1 | MLPClassifier, metrics |
| TensorFlow / Keras | 2.19.0 | MNIST dataset loading |
| Matplotlib | Latest | Loss curve visualization |

---

## 📂 Dataset

The **MNIST (Modified National Institute of Standards and Technology)** database is one of the most well-known benchmarks in machine learning.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

| Split | Samples | Image Size | Classes |
|---|---|---|---|
| Training | 60,000 | 28 × 28 px | 10 (digits 0–9) |
| Testing | 10,000 | 28 × 28 px | 10 (digits 0–9) |

---

## ⚙️ Data Preprocessing

Scikit-learn models require a 2D feature matrix, so each image is **flattened** and **normalized**:

```python
# Flatten: (60000, 28, 28) → (60000, 784)
x_train = x_train_raw.reshape((x_train_raw.shape[0], -1))

# Normalize: pixel values 0–255 → 0.0–1.0
x_train = x_train / 255.0
```

| Step | Before | After |
|---|---|---|
| Shape | `(60000, 28, 28)` | `(60000, 784)` |
| Pixel Range | `0 – 255` | `0.0 – 1.0` |

---

## 🧠 Model Architecture

```python
MLPClassifier(
    hidden_layer_sizes = (64, 64),
    activation         = 'relu',
    solver             = 'adam',
    learning_rate_init = 0.001,
    max_iter           = 30,
    random_state       = 1
)
```

```
Input Layer      →  784 neurons   (flattened 28×28 image)
Hidden Layer 1   →   64 neurons   (ReLU activation)
Hidden Layer 2   →   64 neurons   (ReLU activation)
Output Layer     →   10 neurons   (digits 0–9)

Optimizer:  Adam
Loss:       Cross-Entropy
```

---

## 📊 Results

### Accuracy

```
Test Accuracy: 97.35%
```

### Confusion Matrix

```
           Predicted →
Actual ↓   0     1     2     3     4     5     6     7     8     9
  0      [ 971    0     1     1     0     0     2     1     4     0]
  1      [   0  1122    1     4     0     0     2     1     5     0]
  2      [   5    0   998    10     5     0     2     4     7     1]
  3      [   1    0     4   990     0     6     0     4     4     1]
  4      [   0    1     3     1   967     0     3     1     1     5]
  5      [   4    1     0    16     3   844    11     1     9     3]
  6      [   5    1     0     1     9     2   938     0     2     0]
  7      [   2    3     8     6     1     0     0   999     2     7]
  8      [   5    0     2    15     3     1     4     3   938     3]
  9      [   3    2     0     9    16     1     1     4     5   968]
```

### Loss Curve

The model's training loss decreases steadily across 30 epochs, confirming healthy convergence:

```
Epoch  1 → Loss: 0.453
Epoch 10 → Loss: 0.048
Epoch 20 → Loss: 0.017
Epoch 30 → Loss: 0.009
```

---

## 🖼 Custom Image Prediction

The trained model was tested on a **custom handwritten digit image** (`Number.png`).

### Preprocessing Pipeline for Custom Images

```python
from PIL import Image
import numpy as np

# Step 1: Load the image
img = Image.open('Number.png')

# Step 2: Convert to grayscale
img = img.convert('L')

# Step 3: Resize to 28×28
img = img.resize((28, 28), Image.Resampling.LANCZOS)

# Step 4: Extract pixel data and invert
#         (MNIST uses white-on-black; typical images are black-on-white)
data = [255 - px for px in list(img.getdata())]

# Step 5: Normalize to 0–1
data = np.array(data) / 256.0

# Step 6: Predict
prediction = mlp.predict([data])
print(f"Predicted Digit: {prediction[0]}")
```

> ⚠️ **Important:** Pixel values must be **inverted** before prediction.
> MNIST digits are white on black background, while typical images are black on white.

---

## 📁 Project Structure

```
Handwrittenrecognizer/
│
├── 📓 Handwrittendigitsrecognizer.ipynb   # Main notebook
├── 🖼  Number.png                          # Sample custom image for testing
├── 📄 README.md                           # Project documentation
├── 📄 LICENSE                             # MIT License
├── 📄 CONTRIBUTING.md                     # Contribution guidelines
├── 📄 CHANGELOG.md                        # Version history
└── 📄 .gitignore                          # Git ignore rules
```

---

## 🏁 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pedadarohan-dot/Handwrittenrecognizer.git
cd Handwrittenrecognizer

# 2. Install dependencies
pip install pillow numpy scikit-learn tensorflow matplotlib

# 3. Launch Jupyter Notebook
jupyter notebook Handwrittendigitsrecognizer.ipynb
```

Or run directly in Google Colab using the badge at the top ☝️

### Testing on Your Own Image

1. Place your image file in the project root
2. Update the `file_path` variable in the notebook:
   ```python
   file_path = '/path/to/your/image.png'
   ```
3. Run all cells

---

## 💡 Key Learnings

> **"High accuracy on a test set does not guarantee correct prediction on every real-world input."**

This project reinforced several important ML principles:

- **Accuracy ≠ Perfection** — 97.35% accuracy still means ~265 misclassified images
- **Data distribution matters** — custom images must be preprocessed to match the training data format exactly
- **Inversion is critical** — MNIST trains on white ink on black backgrounds; ignoring this breaks predictions
- **Loss curves tell stories** — a steadily decreasing loss confirms proper convergence; plateaus suggest tuning is needed

---

## 🔭 Future Improvements

- [ ] Increase `max_iter` to allow full convergence
- [ ] Hyperparameter tuning (layer sizes, learning rate, dropout)
- [ ] Implement a CNN (Convolutional Neural Network) for higher accuracy
- [ ] Build a Gradio or Streamlit web app for live drawing predictions
- [ ] Add cross-validation for more robust evaluation
- [ ] Export model with `joblib` for reuse

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

Built with ❤️ as part of a continuous learning journey in Machine Learning & AI.

**⭐ Star this repo if you found it helpful!**

</div>

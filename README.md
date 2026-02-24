---

# 🧠 Handwritten Digit Recognizer (0–9)

A Machine Learning project that recognizes handwritten digits (0–9) using **Scikit-learn’s MLPClassifier** trained on the **MNIST dataset**.

This project demonstrates how neural networks process image data and make predictions based on pixel patterns.

---

## 🚀 Project Overview

The goal of this project is to:

* Train a neural network model on the MNIST dataset
* Preprocess image data for Scikit-learn compatibility
* Evaluate performance using accuracy score and confusion matrix
* Test the trained model on a custom handwritten image

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pillow (PIL)
* Matplotlib
* Scikit-learn
* TensorFlow (for dataset loading)

---

## 📂 Dataset

The project uses the **MNIST dataset**, loaded via TensorFlow:

```python
tf.keras.datasets.mnist.load_data()
```

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* Grayscale images

---

## ⚙️ Data Preprocessing

Since Scikit-learn models expect 2D feature arrays:

1. Images were reshaped from:

   ```
   (28 × 28) → (784,)
   ```

2. Pixel values were normalized:

   ```
   0–255 → 0–1
   ```

This improves model convergence and training stability.

---

## 🧠 Model Architecture

The model used:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=50,
    random_state=1
)
```

### Model Details:

* 2 hidden layers
* 64 neurons in each layer
* ReLU activation
* Adam optimizer
* Loss curve visualization

---

## 📊 Model Evaluation

Evaluation metrics used:

* Confusion Matrix
* Accuracy Score

Example output:

```
Accuracy: ~97% (varies slightly per run)
```

A loss curve was plotted to monitor training performance.

---

## 🖼️ Custom Image Prediction

The model was also tested on a custom handwritten image:

Steps performed:

1. Convert image to grayscale
2. Resize to 28×28 pixels
3. Flatten to 784 features
4. Invert pixel values (to match MNIST format)
5. Normalize
6. Predict using trained model

---

## ⚠️ Key Learning

One important realization from this project:

> Even when the accuracy score is high, the final real-world prediction can still be wrong.

This highlights that:

* Accuracy ≠ Perfection
* Metrics must be interpreted carefully
* Real-world testing is essential

Machine Learning models can perform well statistically but still misclassify individual inputs.

---

## 📈 Future Improvements

* Hyperparameter tuning
* Increasing hidden layers
* Trying CNN (Convolutional Neural Network)
* Cross-validation
* Deployment as a web app

---

## 🏁 Conclusion

This project strengthened my understanding of:

* Image preprocessing
* Neural network fundamentals
* Model evaluation techniques
* Real-world testing challenges

A solid foundational project in Machine Learning and Deep Learning.

---

## 📌 How to Run

1. Install dependencies:

   ```bash
   pip install pillow numpy scikit-learn tensorflow matplotlib
   ```

2. Run the Python script.

3. Provide a custom image file path to test predictions.

---

## 👨‍💻 Author

Built as part of continuous learning in Machine Learning and AI.

---

If you want, I can also generate:

* A more beginner-friendly README
* A more technical/research-style README
* Or a visually enhanced GitHub README with badges and banners 🚀

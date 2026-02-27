# 🤖 Polynomial Regression from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-green.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **A complete implementation of Polynomial Regression using Gradient Descent from scratch in Python**  
> Comprehensive analysis of model complexity, overfitting, and the impact of training data size.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Mathematical Background](#-mathematical-background)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Technical Details](#-technical-details)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements **Polynomial Regression** completely from scratch using Python and NumPy, without relying on scikit-learn or other ML libraries. It explores fundamental machine learning concepts including:

- **Gradient Descent Optimization**
- **Bias-Variance Tradeoff**
- **Overfitting vs Underfitting**
- **Impact of Model Complexity (polynomial degree)**
- **Effect of Training Data Size**

The implementation includes a full-featured `PolynomialRegression` class with customizable hyperparameters and comprehensive visualization of results.

---

## ✨ Features

### Core Implementation
- ✅ **Pure NumPy Implementation**: No ML libraries used for regression
- ✅ **Custom Gradient Descent**: Hand-coded optimization algorithm
- ✅ **Polynomial Feature Engineering**: Automatic generation of polynomial features
- ✅ **Data Normalization**: Built-in feature scaling for numerical stability

### Analysis & Experiments
- 📊 **Model Complexity Analysis**: Testing degrees m = {1, 2, 4, 8, 10, 13, 15, 17, 19}
- 📈 **Training Size Impact**: Experiments with n = {10, 20, 40, 80, 160, 320} samples
- 📉 **Learning Curves**: Train/Test MSE visualization
- 🎯 **Parameter Norm Analysis**: L2 regularization insights

### Visualizations
- 🌈 Multiple high-quality matplotlib plots
- 📐 Scatter plots with fitted polynomials
- 📊 MSE comparison charts
- 🔍 Residual analysis
- 🎨 Professional styling and color schemes

---

## 📐 Mathematical Background

### Polynomial Model

The polynomial regression model of degree `m` is defined as:

```
y' = w₀ + w₁x + w₂x² + ... + wₘxᵐ
```

Where:
- `x` is the input feature (univariate)
- `y'` is the predicted output
- `{w₀, w₁, ..., wₘ}` are the model parameters

### Feature Transformation

```
φ(x) = [1, x, x², ..., xᵐ]
```

This transforms the problem into a linear regression in the feature space:

```
y' = wᵀφ(x)
```

### Loss Function

Mean Squared Error (MSE):

```
L(w) = (1/n) Σᵢ (y'ᵢ - yᵢ)²
```

### Gradient Descent Update Rule

```
wⱼ = wⱼ - α × ∂L/∂wⱼ
```

Where the gradient is:

```
∂L/∂wⱼ = (2/n) Σᵢ (y'ᵢ - yᵢ) × xᵢʲ
```

### L2 Parameter Norm

```
||w||₂² = Σⱼ₌₁ᵐ wⱼ²
```

(Note: excludes bias term w₀)

---

## 📁 Project Structure

```
polynomial-regression/
│
├── AI_Phase2_Complete.ipynb       # Main Jupyter notebook
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── data/                           # Dataset directory
│   ├── trainX.npy                 # Training inputs (320 samples)
│   ├── trainY.npy                 # Training outputs
│   ├── testX.npy                  # Test inputs (180 samples)
│   └── testY.npy                  # Test outputs
│
└── results/                        # Generated plots and analysis
    ├── scatter_plot.png
    ├── polynomial_curves.png
    ├── mse_vs_degree.png
    ├── mse_vs_samples.png
    └── parameter_norm.png
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/polynomial-regression.git
cd polynomial-regression
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install numpy matplotlib pandas jupyter
```

3. **Launch Jupyter Notebook**

```bash
jupyter notebook AI_Phase2_Complete.ipynb
```

### Google Colab

Alternatively, run directly in Google Colab:

1. Upload the notebook to Google Colab
2. Upload data files when prompted
3. Run all cells

---

## 🚀 Usage

### Quick Start

```python
import numpy as np
from polynomial_regression import PolynomialRegression

# Load data
trainX = np.load('data/trainX.npy')
trainY = np.load('data/trainY.npy')

# Create and train model
model = PolynomialRegression(degree=4, learning_rate=0.001, epochs=5000)
model.fit(trainX, trainY)

# Make predictions
predictions = model.predict(testX)

# Evaluate
mse = model.score(testX, testY)
print(f"Test MSE: {mse:.4f}")
```

### Custom Parameters

```python
# For high-degree polynomials with small datasets
model = PolynomialRegression(
    degree=15,
    learning_rate=1e-6,  # Lower LR for stability
    epochs=10000
)
```

### Normalized Data

```python
# Normalize features for numerical stability
X_mean, X_std = trainX.mean(), trainX.std()
X_norm = (trainX - X_mean) / X_std

model.fit(X_norm, trainY)
```

---

## 🔍 Key Findings

### 1. Model Complexity (Polynomial Degree)

| Degree | Behavior | Train MSE | Test MSE |
|--------|----------|-----------|----------|
| m = 1-2 | **Underfitting** | High | High |
| m = 4-8 | **Optimal** | Medium | Low |
| m = 13+ | **Overfitting** | Very Low | High |

**Key Insight**: With only 20 training samples, optimal degree is around **m=4-8**.

### 2. Training Data Size Impact

With fixed m=19:

| Samples | Train MSE | Test MSE | Overfitting |
|---------|-----------|----------|-------------|
| n = 10 | 0.62 | 2.53 | Severe |
| n = 40 | 0.71 | 0.66 | Moderate |
| n = 160 | 0.65 | 0.67 | Minimal |
| n = 320 | 0.87 | 0.90 | None |

**Key Insight**: More data → better generalization. Rule of thumb: **n >> m+1**

### 3. Parameter Growth

- As degree increases, L2 norm of parameters grows exponentially
- High parameter values indicate overfitting
- Regularization recommended for m > 10

### 4. Numerical Stability

- For m ≥ 15, vanilla gradient descent becomes unstable
- Feature normalization is **essential**
- Adaptive learning rates required: lr ∝ 1/(n × max(x)^(2m))

---

## 🛠️ Technical Details

### PolynomialRegression Class

#### Parameters

- `degree` (int): Polynomial degree (m)
- `learning_rate` (float): Step size for gradient descent (α)
- `epochs` (int): Number of training iterations

#### Methods

- `fit(X, y)`: Train the model using gradient descent
- `predict(X)`: Generate predictions for new data
- `score(X, y)`: Calculate MSE on given data
- `get_weights_norm()`: Compute L2 norm of parameters

#### Attributes

- `weights`: Learned parameters [w₀, w₁, ..., wₘ]
- `loss_history`: Training loss at each epoch

### Gradient Descent Implementation

```python
def fit(self, X, y):
    X_poly = self._create_polynomial_features(X)
    n_samples = X.shape[0]
    
    # Initialize weights
    self.weights = np.random.randn(self.degree + 1, 1) * 0.01
    
    # Training loop
    for epoch in range(self.epochs):
        # Forward pass
        y_pred = X_poly @ self.weights
        
        # Compute gradient
        error = y_pred - y
        gradient = (2 / n_samples) * (X_poly.T @ error)
        
        # Update weights
        self.weights -= self.learning_rate * gradient
```

### Handling Numerical Issues

**Problem**: For high-degree polynomials, x^m becomes extremely large → exploding gradients

**Solution**: 
1. Feature normalization: `X_norm = (X - μ) / σ`
2. Target normalization: `y_norm = (y - μ_y) / σ_y`
3. Adaptive learning rates based on data range

```python
# Learning rate schedule for m=19
lr_map = {
    10: 1e-6,
    20: 1e-7,
    40: 1e-9,
    80: 1e-10,
    160: 1e-11,
    320: 1e-12
}
```

---

## 📈 Results

### Best Model Selection

For the given dataset:

- **With n=20 samples**: Optimal degree = **m=4** (Test MSE: 0.15)
- **With n=320 samples**: Can use up to m=15 without overfitting

### Overfitting Detection

**Signs of overfitting observed**:
1. Large gap between train and test MSE
2. High L2 norm of parameters (||w||₂² > 100)
3. Train MSE near zero, but test MSE high
4. Oscillatory behavior in fitted curves

### Practical Guidelines

1. **Data Collection**: Aim for n ≥ 10(m+1) samples
2. **Model Selection**: Start simple (m=2-4), increase gradually
3. **Validation**: Always use separate test set
4. **Regularization**: Add L2 penalty for m > 10
5. **Normalization**: Essential for m ≥ 15

---

## 🎓 Learning Outcomes

This project demonstrates:

- **Bias-Variance Tradeoff**: Simple models have high bias; complex models have high variance
- **Importance of Data**: More data is often better than more complex models
- **Gradient Descent**: Understanding optimization from first principles
- **Numerical Stability**: Real-world challenges in implementing ML algorithms
- **Model Evaluation**: Why train error alone is misleading

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add regularization (Ridge/Lasso)
- [ ] Implement cross-validation
- [ ] Add early stopping
- [ ] Support for multivariate regression
- [ ] Interactive visualizations with Plotly
- [ ] Comparison with sklearn implementation

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


</div>

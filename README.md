# 🧠 ML Course – Algorithms from Scratch

This repository contains hands-on machine learning projects implementing core algorithms from scratch using NumPy and comparing them with their Scikit-learn counterparts. Each subdirectory focuses on a specific algorithm and includes data preprocessing, training, evaluation, and visualizations.

---

## 📂 Projects Overview

### 🔹 [KNN Classifier](knn/README.md)

- Implements K-Nearest Neighbors from scratch.
- Uses the Iris dataset.
- Includes heatmap and result comparison.
- Accuracy: **0.98** (same as Scikit-learn).

### 🔹 [Linear Regression](linear_regression/README.md)

- Implements gradient descent for linear regression.
- Normalizes input data and compares against Scikit-learn's `LinearRegression`.
- Final Loss (Custom): **0.5309**, Sklearn: **0.5243**

### 🔹 [Logistic Regression (Titanic Survival)](logistic_regression/README.md)

- Implements logistic regression with NumPy.
- Predicts Titanic survival using real data.
- Includes correlation, survival analysis, and multiple plots.
- Achieves up to **78.77%** accuracy depending on the split.

### 🔹 [Ridge Regression (Polynomial)](ridge_regression/README.md)

- Implements polynomial regression with L2 regularization.
- Includes data generation, training (GD and closed-form), and visualizations.
- Demonstrates overfitting, regularization, and model complexity.

---

## 🧑‍💻 Author

Created by Marcin Bugala as a hands-on exercise in building machine learning algorithms from scratch.  
This project is intended for learning and exploration
# Machine Learning: Optimizers Comparison on Polynomial & Logistic Regression

This repository contains 5 Kaggle notebooks implementing **Polynomial Linear Regression** (Assignment 1) and **Binary Logistic Regression** (Assignment 2) using Scikit-Learn across **5 different optimization algorithms**.

## 📊 Dataset & Preprocessing
* **Dataset:** [Climate Change: Earth Surface Temperature Data](https://kaggle.com) (`GlobalLandTemperaturesByMajorCity.csv`).
* **Subset:** Monthly average temperatures for **Chicago** (`2000-07-01` to `2001-07-01`), capturing a full seasonal U-shape cycle (Summer $\rightarrow$ Winter $\rightarrow$ Summer).
* **Feature Engineering:**
  * Centered time steps (`Months Centered`).
  * Polynomial feature expansion (`degree=2`) to capture non-linear seasonal patterns.
  * Feature standardization using `StandardScaler`.

---

## 🚀 Implemented Optimizers (5 Notebooks)
Each notebook isolates a specific optimization strategy applied to both regression and classification tasks:

1. **Gradient Descent (GD):** Full-batch parameter updates.
2. **Stochastic Gradient Descent (SGD):** Iterative single-sample updates (`SGDRegressor` / `SGDClassifier`).
3. **Mini-Batch Gradient Descent:** Incremental batch updates using `partial_fit` over multiple epochs.
4. **AdaGrad (Adaptive Gradient):** Adaptive learning rate scheduling (`learning_rate='adaptive'`).
5. **Momentum & Adam:** Accelerated gradient updates using momentum (`momentum=0.9`) and adaptive moment estimation (`solver='adam'`).

---

## 🎯 Assignment 2: Binary Classification (Logistic Regression)
In the second phase of each notebook, the continuous temperature target was converted into a **Binary Classification** problem:
* **Class `1` (Warm):** Temperature above the annual mean.
* **Class `0` (Cold):** Temperature at or below the annual mean.

### 📈 Evaluation Results Comparison

| Optimizer | Log Loss | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Descent (GD)** | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` |
| **Stochastic GD (SGD)** | `0.2393` | `0.9231` | `1.0000` | `0.8571` | `0.9231` |
| **Mini-Batch GD** | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` |
| **AdaGrad** | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` |
| **GD with Momentum** | `0.2514` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **Adam** | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` | `0.xxxx` |

*(Note: Replace `0.xxxx` with your exact output metrics from the remaining Kaggle notebooks.)*

---

## 🛠️ Technologies Used
* **Python 3**
* **Pandas & NumPy** (Data manipulation & numerical operations)
* **Scikit-Learn** (`SGDRegressor`, `SGDClassifier`, `MLPRegressor`, `MLPClassifier`, `PolynomialFeatures`, `StandardScaler`, `metrics`)
* **Matplotlib** (Data & decision boundary visualization)

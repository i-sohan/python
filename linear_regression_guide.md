# Master Guide: Linear Regression for Machine Learning Associates

---

## Table of Contents
1. [Introduction & Fundamental Concepts](#1-introduction--fundamental-concepts)
2. [Mathematical Formulations](#2-mathematical-formulations)
3. [The 5 Core Assumptions of Linear Regression & Diagnostics](#3-the-5-core-assumptions-of-linear-regression--diagnostics)
4. [Model Estimation & Fitting Algorithms](#4-model-estimation--fitting-algorithms)
   - [A. Ordinary Least Squares (OLS) Analytical Solution](#a-ordinary-least-squares-ols-analytical-solution)
   - [B. Gradient Descent (Iterative Solution)](#b-gradient-descent-iterative-solution)
5. [Model Evaluation Metrics](#5-model-evaluation-metrics)
6. [Regularization Techniques ($L_1$, $L_2$, ElasticNet)](#6-regularization-techniques-l_1-l_2-elasticnet)
7. [Feature Engineering & Preprocessing](#7-feature-engineering--preprocessing)
8. [Complete Python Implementations](#8-complete-python-implementations)
   - [Implementation 1: Pure NumPy from Scratch (OLS & Gradient Descent)](#implementation-1-pure-numpy-from-scratch-ols--gradient-descent)
   - [Implementation 2: Production-Grade Scikit-Learn Pipeline & Diagnostics](#implementation-2-production-grade-scikit-learn-pipeline--diagnostics)
9. [ML Associate Certification & Interview Cheat Sheet](#9-ml-associate-certification--interview-cheat-sheet)

---

## 1. Introduction & Fundamental Concepts

**Linear Regression** is a fundamental supervised learning algorithm used to model the linear relationship between one or more independent predictor variables ($X$) and a continuous target variable ($y$).

### Classification of Linear Regression
- **Simple Linear Regression**: Uses a single feature $x$ to predict target $y$.
- **Multiple Linear Regression**: Uses multiple features $x_1, x_2, \dots, x_p$ to predict target $y$.
- **Multivariate Linear Regression**: Predicts multiple continuous target variables $y_1, y_2, \dots, y_k$ simultaneously (less common, distinct from multiple linear regression).

---

## 2. Mathematical Formulations

### Simple Linear Regression
$$y = \beta_0 + \beta_1 x + \epsilon$$

- $y$: Dependent target variable.
- $x$: Independent feature variable.
- $\beta_0$: $y$-intercept (value of $y$ when $x = 0$).
- $\beta_1$: Slope coefficient (change in $y$ per unit change in $x$).
- $\epsilon$: Error term (unexplained noise/residuals), where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

### Multiple Linear Regression (Scalar Form)
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$

### Multiple Linear Regression (Matrix / Vector Form)
For $n$ observations and $p$ features:

$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}$$

Where:
- $\mathbf{y}$ is an $n \times 1$ target vector.
- $\mathbf{X}$ is an $n \times (p+1)$ design matrix (including a column of $1$s for the intercept $\beta_0$):
  $$\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{12} & \dots & x_{1p} \\ 1 & x_{21} & x_{22} & \dots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \dots & x_{np} \end{bmatrix}$$
- $\boldsymbol{\beta}$ is a $(p+1) \times 1$ parameter/weight vector: $\boldsymbol{\beta} = [\beta_0, \beta_1, \dots, \beta_p]^T$.
- $\boldsymbol{\epsilon}$ is an $n \times 1$ residual error vector.

---

## 3. The 5 Core Assumptions of Linear Regression & Diagnostics

For Ordinary Least Squares (OLS) estimates to be **BLUE** (**Best Linear Unbiased Estimator**) according to the **Gauss-Markov Theorem**, the dataset and model residuals must satisfy five fundamental assumptions.

```
+-------------------------------------------------------------------------------+
|                       5 CORE OLS ASSUMPTIONS                                 |
+---------------------+-------------------------------+-------------------------+
| Assumption          | Diagnostic Tool               | Remedial Action         |
+---------------------+-------------------------------+-------------------------+
| 1. Linearity        | Residuals vs. Fitted plot     | Log/Polynomial transforms|
| 2. Independence     | Durbin-Watson Test (~2.0)     | Time-series / AR models |
| 3. Homoscedasticity | Breusch-Pagan / Scale-Location| Box-Cox / Log transform |
| 4. Normality        | Q-Q Plot / Shapiro-Wilk Test  | Non-linear transform    |
| 5. No Multicollinearity| Variance Inflation Factor (VIF)| Drop features / Ridge  |
+---------------------+-------------------------------+-------------------------+
```

### 1. Linearity
- **Concept**: The relationship between independent variables $X$ and dependent variable $y$ is additive and linear.
- **Diagnostic**: Plot **Residuals vs. Fitted Values ($\hat{y}$)**. Residuals should be randomly scattered around $y=0$ with no discernible pattern (curves, U-shapes).
- **Fix**: Apply non-linear feature transformations (e.g., $\log(x)$, $\sqrt{x}$, $x^2$) or interaction terms.

### 2. Independence of Errors (No Autocorrelation)
- **Concept**: Residual errors $\epsilon_i$ and $\epsilon_j$ must be uncorrelated for all $i \neq j$. Crucial in time-series data.
- **Diagnostic**: **Durbin-Watson Test**.
  - $d \approx 2.0$: No autocorrelation.
  - $d < 1.5$: Positive autocorrelation.
  - $d > 2.5$: Negative autocorrelation.
- **Fix**: Add lag variables or use Generalized Least Squares (GLS) / ARIMA models.

### 3. Homoscedasticity (Constant Variance of Residuals)
- **Concept**: The variance of residual errors $\text{Var}(\epsilon_i) = \sigma^2$ must remain constant across all levels of predicted values $\hat{y}$.
- **Heteroscedasticity** (violation): Variance spreads out (e.g., funnel/cone shape in residual plot).
- **Diagnostic**: **Breusch-Pagan test** or **Scale-Location plot** ($\sqrt{|\text{standardized residuals}|}$ vs $\hat{y}$).
- **Fix**: Log transform target variable $\log(y)$, Box-Cox transformation, or Weighted Least Squares (WLS).

### 4. Normality of Residuals
- **Concept**: The residual errors $\epsilon_i$ should be normally distributed: $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
- **Diagnostic**: **Quantile-Quantile (Q-Q) Plot** (points should lie along the 45-degree line) or statistical tests (Shapiro-Wilk test, Jarque-Bera test).
- **Fix**: Target variable transformation ($\log$, power transform) or remove severe outliers.

### 5. Absence of Multicollinearity
- **Concept**: Predictor features $x_i$ and $x_j$ must not be highly linearly correlated with each other. Multicollinearity makes coefficient estimates unstable and inflates standard errors.
- **Diagnostic**: **Variance Inflation Factor (VIF)** for feature $j$:
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  - $\text{VIF} = 1$: No correlation.
  - $1 < \text{VIF} < 5$: Moderate correlation (acceptable).
  - $\text{VIF} \ge 5\text{ to }10$: High multicollinearity (problematic).
- **Fix**: Drop redundant features, combine correlated features (PCA), or apply **Ridge ($L_2$) Regularization**.

---

## 4. Model Estimation & Fitting Algorithms

### A. Ordinary Least Squares (OLS) Analytical Solution

OLS minimizes the Sum of Squared Residuals (SSR):

$$S(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Expanding $S(\boldsymbol{\beta})$:
$$S(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

Taking the derivative with respect to $\boldsymbol{\beta}$ and setting to $0$:
$$\frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0$$

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

Solving for $\boldsymbol{\hat{\beta}}$ yields the **Normal Equation**:

$$\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

- **Pros**: Exact closed-form solution; no hyperparameter tuning (no learning rate).
- **Cons**: Matrix inversion $(\mathbf{X}^T \mathbf{X})^{-1}$ takes $\mathcal{O}(p^3)$ time complexity. Fails or becomes unstable if $\mathbf{X}^T \mathbf{X}$ is non-invertible (singular/multicollinear).

---

### B. Gradient Descent (Iterative Solution)

Used when the number of features $p$ is very large ($p > 10,000$) or for online/streaming data learning.

#### Cost Function (Mean Squared Error)
$$J(\boldsymbol{\beta}) = \frac{1}{2n} \sum_{i=1}^n (\mathbf{x}_i^T \boldsymbol{\beta} - y_i)^2 = \frac{1}{2n} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2_2$$

*(The factor of $\frac{1}{2}$ simplifies the derivative).*

#### Gradient Calculation
$$\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = \frac{1}{n} \mathbf{X}^T (\mathbf{X}\boldsymbol{\beta} - \mathbf{y})$$

#### Weight Update Rule
$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}^{(t)})$$

Where $\alpha$ is the **Learning Rate**.

```
+-------------------------------------------------------------------------------------+
|                              GRADIENT DESCENT VARIANTS                              |
+---------------------+-------------------------------+-------------------------------+
| Variant             | Batch Size per Update         | Pros / Cons                   |
+---------------------+-------------------------------+-------------------------------+
| Batch GD            | Entire Dataset ($n$ samples)  | Smooth convergence; slow for  |
|                     |                               | huge datasets.                |
| Stochastic GD (SGD) | 1 sample                      | Fast; noisy updates; escapes  |
|                     |                               | local minima.                 |
| Mini-Batch GD       | Batch size $b$ (e.g. 32-256)  | Optimal vectorization balance;|
|                     |                               | standard in Deep Learning.    |
+---------------------+-------------------------------+-------------------------------+
```

---

## 5. Model Evaluation Metrics

Assume $y_i$ is actual, $\hat{y}_i$ is predicted, and $\bar{y}$ is the mean of true values.

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Interpretation**: Average absolute deviation. Robust to outliers.

### 2. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
- **Interpretation**: Punishes larger errors quadratically. Differentiable.

### 3. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Interpretation**: Same scale as target variable $y$.

### 4. Coefficient of Determination ($R^2$ Score)
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

- **Interpretation**: Percentage of variance in $y$ explained by the model features $X$.
- Range: $(-\infty, 1.0]$. $R^2 = 1.0$ is perfect fit; $R^2 = 0.0$ performs as poorly as predicting the mean $\bar{y}$.
- **Flaw**: $R^2$ **always increases or stays same** when adding new variables, even if they are irrelevant noise!

### 5. Adjusted $R^2$ Score
$$\bar{R}^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$

- Where $n$ is total sample size, and $p$ is the number of predictor features.
- **Interpretation**: Penalizes model complexity for adding irrelevant features. Used for model comparison and variable selection.

---

## 6. Regularization Techniques ($L_1$, $L_2$, ElasticNet)

Regularization prevents **overfitting** (high variance) by adding a penalty term to the OLS loss function.

```
+--------------------------------------------------------------------------------------+
|                           REGULARIZATION COMPARISON                                  |
+--------------------+-----------------------+--------------------+--------------------+
| Model              | Penalty Type          | Sparsity (Feature  | Best For           |
|                    |                       | Selection)         |                    |
+--------------------+-----------------------+--------------------+--------------------+
| Ridge Regression   | $L_2: \lambda \|\beta\|_2^2$ | No (shrinks coefficients) | Multicollinearity & |
|                    |                       |                    | many features.     |
| Lasso Regression   | $L_1: \lambda \|\beta\|_1$   | Yes (exact zeros)  | High-dimensional   |
|                    |                       |                    | feature selection. |
| ElasticNet         | $\alpha L_1 + (1-\alpha)L_2$ | Yes                | Correlated feature |
|                    |                       |                    | groups.            |
+--------------------+-----------------------+--------------------+--------------------+
```

### A. Ridge Regression ($L_2$ Regularization)
$$J_{\text{Ridge}}(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2_2 + \lambda \|\boldsymbol{\beta}\|^2_2 = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p \beta_j^2$$

#### Closed-Form Solution:
$$\boldsymbol{\hat{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

- Adds $\lambda \mathbf{I}$ to matrix invertibility, ensuring $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})$ is **always invertible**, resolving collinearity issues.
- Shrinks weights towards zero, but **never sets weights to exact zero**.

### B. Lasso Regression ($L_1$ Regularization)
$$J_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2_2 + \lambda \|\boldsymbol{\beta}\|_1 = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p |\beta_j|$$

- Uses diamond-shaped geometry for penalty constraints.
- Drives coefficients of non-important features **to exact zero**, performing automatic **feature selection**.

### C. ElasticNet Regression
$$J_{\text{ElasticNet}}(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{X}\boldsymbol{\beta} - \mathbf{y}\|^2_2 + \lambda \left( r \|\boldsymbol{\beta}\|_1 + \frac{1 - r}{2} \|\boldsymbol{\beta}\|^2_2 \right)$$

- $r \in [0, 1]$ is the $L_1$ ratio.
- Combines $L_1$ and $L_2$ advantages. Ideal when features are highly correlated in groups.

---

## 7. Feature Engineering & Preprocessing

1. **Feature Scaling (Mandatory for GD & Regularization)**:
   - **Standardization (Z-score)**: $x' = \frac{x - \mu}{\sigma}$. Required for Ridge/Lasso penalties to treat features equally.
   - **Min-Max Scaling**: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$.
2. **Handling Categorical Features**:
   - **One-Hot Encoding**: Converts categorical levels into binary columns.
   - **Dummy Variable Trap**: Avoid multi-collinearity by dropping one level ($k-1$ dummy columns created for $k$ categories).
3. **Polynomial Features**:
   - Models non-linear curves using linear parameters: $y = \beta_0 + \beta_1 x + \beta_2 x^2$.

---

## 8. Complete Python Implementations

### Implementation 1: Pure NumPy from Scratch (OLS & Gradient Descent)

```python
import numpy as np

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy.
    Supports both OLS Analytical Solution and Gradient Descent optimization.
    """
    def __init__(self, method="ols", lr=0.01, n_iters=1000):
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.method == "ols":
            # Add bias column (column of 1s) to X design matrix
            X_b = np.c_[np.ones((n_samples, 1)), X]
            # Closed form: beta = (X^T * X)^(-1) * X^T * y
            beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = beta[0]
            self.weights = beta[1:]

        elif self.method == "gd":
            # Initialize weights and bias to zeros
            self.weights = np.zeros(n_features)
            self.bias = 0.0

            for i in range(self.n_iters):
                # Forward pass: y_hat = X * w + b
                y_predicted = np.dot(X, self.weights) + self.bias

                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)

                # Gradient descent step
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                # Record loss (MSE)
                mse = np.mean((y_predicted - y) ** 2)
                self.loss_history.append(mse)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Quick Verification
if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100) * 0.5  # True relationship: y = 4 + 3x

    # OLS Model
    ols_model = LinearRegressionScratch(method="ols")
    ols_model.fit(X, y)
    print(f"OLS Estimate -> Intercept: {ols_model.bias:.4f}, Slope: {ols_model.weights[0]:.4f}")

    # Gradient Descent Model
    gd_model = LinearRegressionScratch(method="gd", lr=0.1, n_iters=1000)
    gd_model.fit(X, y)
    print(f"GD  Estimate -> Intercept: {gd_model.bias:.4f}, Slope: {gd_model.weights[0]:.4f}")
```

---

### Implementation 2: Production-Grade Scikit-Learn Pipeline & Diagnostics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame.sample(n=1000, random_state=42)  # Sample for quick diagnostic run
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocessing (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models
models = {
    "OLS Linear Regression": LinearRegression(),
    "Ridge (L2)": RidgeCV(alphas=np.logspace(-3, 3, 10)),
    "Lasso (L1)": LassoCV(alphas=np.logspace(-3, 3, 10), cv=5, random_state=42),
    "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    results.append({
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Adj R2": round(adj_r2, 4)
    })

# Display Evaluation Summary Table
results_df = pd.DataFrame(results)
print("=== MODEL EVALUATION METRICS ===")
print(results_df.to_string(index=False))

# 5. Diagnostic Residual Plot for OLS Model
ols_model = models["OLS Linear Regression"]
y_test_pred = ols_model.predict(X_test_scaled)
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Residuals vs Fitted
sns.scatterplot(x=y_test_pred, y=residuals, ax=axes[0], alpha=0.7, color="navy")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals vs Fitted (Check Linearity & Homoscedasticity)")
axes[0].set_xlabel("Fitted Values (y_pred)")
axes[0].set_ylabel("Residuals (y_true - y_pred)")

# Plot B: Residual Distribution
sns.histplot(residuals, kde=True, ax=axes[1], color="teal")
axes[1].set_title("Residuals Distribution (Check Normality)")
axes[1].set_xlabel("Residual Error")

plt.tight_layout()
plt.show()
```

---

## 9. ML Associate Certification & Interview Cheat Sheet

| Question / Concept | Core Explanation / Key Takeaway |
| :--- | :--- |
| **Dummy Variable Trap** | High multicollinearity caused by including all one-hot encoded categories. Solution: Drop 1 dummy column ($k-1$ dummies). |
| **$R^2$ vs. Adjusted $R^2$** | $R^2$ never decreases when adding features. Adjusted $R^2$ penalizes irrelevant features using formula $1 - \frac{(1-R^2)(n-1)}{n-p-1}$. |
| **OLS vs Gradient Descent** | OLS uses analytical formula $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ (fast for small $p$). GD is iterative (scalable for $p > 10,000$). |
| **Lasso ($L_1$) vs Ridge ($L_2$)** | Lasso drives coefficients to **exact 0** (feature selection). Ridge shrinks weights towards 0, solving multicollinearity. |
| **Heteroscedasticity** | Non-constant variance of residuals. Identified via funnel shape on residual plot. Fixed via log/power transformations. |
| **VIF (Variance Inflation Factor)** | Measures feature multicollinearity. $\text{VIF} > 5 \text{ to } 10$ indicates severe correlation between predictors. |
| **Feature Scaling** | Required before Gradient Descent or Ridge/Lasso regularization so penalty terms treat all features equally. |

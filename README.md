# Machine Learning Algorithms

### **Supervised Learning**
- **Linear Regression**: Predicts continuous values based on input features. [Read it in Detail](#LinearRegressionHead)
- **Logistic Regression**: Used for binary classification problems.
- **Decision Trees**: Splits data into branches to make predictions.
- **Support Vector Machines (SVM)**: Finds the optimal boundary between classes.
- **Neural Networks**: Mimics the human brain to solve complex problems.

### **Unsupervised Learning**
- **K-Means Clustering**: Groups data into clusters based on similarity.
- **Principal Component Analysis (PCA)**: Reduces dimensionality while preserving variance.
- **Hierarchical Clustering**: Builds a tree-like structure for data grouping.

### **Ensemble Methods**
- **Random Forest**: Combines multiple decision trees for better accuracy.
- **Gradient Boosting**: Sequentially improves predictions by correcting errors.

### **Deep Learning**
- **Convolutional Neural Networks (CNNs)**: Specialized for image data.
- **Recurrent Neural Networks (RNNs)**: Processes sequential data like time series.

### **Reinforcement Learning**
- **Q-Learning**: Learns optimal actions by maximizing rewards.
- **Deep Q-Networks (DQN)**: Combines Q-learning with deep neural networks.


<h1 id="LinearRegressionHead">Linear Regression Explained</h1>

## What is Linear Regression?
Linear Regression is a supervised learning algorithm used for predictive modeling. It models the relationship between a dependent variable (target) and one or more independent variables (predictors) by fitting a linear equation to observed data.

The simplest form is **Simple Linear Regression**, which uses one independent variable. For multiple predictors, it's called **Multiple Linear Regression**.

---

## Simple Linear Regression

### Formula:
The equation of a straight line is:
**y = β₀ + β₁x + ε**

Where:
- **y**: Dependent variable (response).
- **x**: Independent variable (predictor).
- **β₀**: Intercept (value of y when x = 0).
- **β₁**: Slope of the line (rate of change in y with respect to x).
- **ε**: Error term (accounts for variability in y not explained by the model).

### Goal:
To minimize the **Residual Sum of Squares (RSS)**:
**RSS = Σ(yᵢ - (β₀ + β₁xᵢ))²**

---

## Multiple Linear Regression

### Formula:
The equation for multiple predictors is:
**y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε**

Where:
- **x₁, x₂, ..., xₖ**: Independent variables.
- **β₁, β₂, ..., βₖ**: Corresponding coefficients.

### Matrix Representation:
Linear regression can also be expressed as a matrix equation:
**y = Xβ + ε**

Where:
- **y**: Vector of observations.
- **X**: Matrix of predictor variables.
- **β**: Vector of coefficients.
- **ε**: Vector of errors.

---

## Assumptions of Linear Regression

### 1. **Linearity**
The relationship between the independent variables (predictors) and the dependent variable (target) must be **linear** in linear regression.

### Mathematical Representation:

\[
y = β₀ + β₁x₁ + β₂x₂ + \dots + βₖxₖ + ε
\]

Where:
- **y**: Dependent variable (target).
- **x₁, x₂, ..., xₖ**: Independent variables (predictors).
- **β₀, β₁, ..., βₖ**: Coefficients of the independent variables.
- **ε**: Error term that accounts for variability not explained by the model.

---

## Example with Mathematical Proof

Let us assume:
- **y** is the price of a house (in $1000s),
- **x₁** is the size of the house (in square feet),
- **x₂** is the number of bedrooms.

The regression equation is given as:

\[
y = 50 + 0.2x₁ + 5x₂
\]

### Coefficients:
- **Intercept (β₀ = 50):** The base price of a house when size and bedrooms are zero.
- **Coefficient of x₁ (β₁ = 0.2):** For every additional square foot, the price increases by $200.
- **Coefficient of x₂ (β₂ = 5):** For every additional bedroom, the price increases by $5000.

### Example Calculation:
Let:
- **x₁ = 2000** (size = 2000 sq. ft.),
- **x₂ = 3** (number of bedrooms = 3).

Using the equation:
\[
y = β₀ + β₁x₁ + β₂x₂
\]

Substitute values:
\[
y = 50 + 0.2(2000) + 5(3)
\]

Simplify:
\[
y = 50 + 400 + 15 = 465
\]

Thus, the predicted price of the house is **$465,000**.

---

### Checking the Assumption of Linearity

The assumption of linearity can be checked by analyzing the residuals (i.e., <strong>actual (y)  - predicted (y)</strong>).
A random pattern in residual plots supports the linearity assumption.

---

### 2. **Autocorrelation - Independence**
The residuals (errors) should be independent of one another. Violating this assumption results in autocorrelation.

Autocorrelation refers to the correlation between residuals of consecutive observations, common in time-series data. It violates the assumption of independence.
Mathematical representation, Residuals should satisfy:
**Cov(εᵢ, εᵢ₋₁) = 0**

Test for autocorrelation:
- **Durbin-Watson Statistic**:
  **DW = Σ((εᵢ - εᵢ₋₁)²) / Σ(εᵢ²)**

Where:
- **DW ≈ 2**: Indicates no autocorrelation.
- **DW < 2**: Indicates positive autocorrelation.
- **DW > 2**: Indicates negative autocorrelation.

---

### 3. **Homoscedasticity**
The variance of the residuals should remain constant across all values of the independent variables.

Mathematical representation:
**Var(εᵢ) = σ²**

Where:
- **σ²**: Constant variance.

---

### 4. **Normality**
The residuals should follow a normal distribution.

Mathematical representation:
**ε ∼ N(0, σ²)**

Where:
- **N(0, σ²)**: Residuals are normally distributed with mean 0 and variance σ².

---

### 5. **No Multicollinearity**
Independent variables should not be highly correlated with one another.

Mathematical test (Variance Inflation Factor - VIF):
**VIF = 1 / (1 - R²)**

Where:
- **R²**: Coefficient of determination for the regression of one independent variable on all others.

---

### 6. **Outlier Influence**
Outliers should not disproportionately affect the model's predictions.

Mathematical representation:
Outliers are detected using metrics like:
- **Cook's Distance**: Measures the influence of an individual data point.

Formula for Cook's Distance:
**Dᵢ = (RSSᵢ / p) × (hᵢ / (1 - hᵢ)²)**

Where:
- **RSSᵢ**: Residual Sum of Squares excluding observation i.
- **p**: Number of predictors.
- **hᵢ**: Leverage of observation i.

---

### Summary
These assumptions are essential for building a reliable and accurate linear regression model. Violating these assumptions can result in biased coefficients, poor predictions, and unreliable statistical inferences.

---

## Advanced Topics

### 1. **Regularization**
To prevent overfitting, techniques like **Lasso** and **Ridge Regression** modify the loss function by adding penalties:
- **Ridge Regression**: Adds an L2 penalty (sum of squared coefficients):
  **RSS + λΣβᵢ²**
- **Lasso Regression**: Adds an L1 penalty (sum of absolute coefficients):
  **RSS + λΣ|βᵢ|**

### 2. **Polynomial Regression**
When the relationship is not linear, we can use polynomial terms:
**y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε**

### 3. **Feature Selection**
To improve model interpretability, irrelevant predictors can be removed using techniques like stepwise regression or statistical tests.

### 4. **Evaluating Model Performance**
- **R² (Coefficient of Determination)**: Measures goodness of fit (higher is better).
- **Adjusted R²**: Adjusts R² for the number of predictors.
- **RMSE (Root Mean Squared Error)**: Measures prediction error.

---

## Applications of Linear Regression
1. Predicting housing prices.
2. Estimating stock market trends.
3. Forecasting sales and revenue.
4. Analyzing relationships in research studies.

---


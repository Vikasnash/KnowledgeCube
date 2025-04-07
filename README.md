## Machine Learning Algorithms

### **Supervised Learning**
- **Linear Regression**: Predicts continuous values based on input features. [Read it in Detail](#Linear-Regression-Explained-(Scratch-to-Advanced))
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


# Linear Regression Explained (Scratch to Advanced)

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
Linear regression relies on the following assumptions:
1. **Linearity**: Relationship between predictors and target is linear.
2. **Independence**: Residuals are independent.
3. **Homoscedasticity**: Constant variance of errors.
4. **Normality**: Residuals are normally distributed.
5. **No Multicollinearity**: Predictors are not highly correlated.

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

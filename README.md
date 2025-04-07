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

### 1. **Linearity**
The relationship between the independent variables (predictors) and the dependent variable (target) must be linear.

Mathematical representation:
**y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε**

Where:
- **y**: Dependent variable.
- **x₁, x₂, ..., xₖ**: Independent variables.
- **β₀, β₁, ..., βₖ**: Coefficients.
- **ε**: Error term.

---

### 2. **Independence**
The residuals (errors) should be independent of one another. Violating this assumption results in autocorrelation.

Mathematical test for independence:
**Cov(εᵢ, εⱼ) = 0, for all i ≠ j**

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

### 7. **No Autocorrelation**
Autocorrelation refers to the correlation between residuals of consecutive observations, common in time-series data. It violates the assumption of independence.

Mathematical representation:
Residuals should satisfy:
**Cov(εᵢ, εᵢ₋₁) = 0**

Test for autocorrelation:
- **Durbin-Watson Statistic**:
  **DW = Σ((εᵢ - εᵢ₋₁)²) / Σ(εᵢ²)**

Where:
- **DW ≈ 2**: Indicates no autocorrelation.
- **DW < 2**: Indicates positive autocorrelation.
- **DW > 2**: Indicates negative autocorrelation.

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







# Principal Component Analysis (PCA) Step-by-Step

PCA is essentially a linear algebra technique that transforms a dataset to a new coordinate system, where the axes (principal components) correspond to the directions of maximum variance in the data. Here's how it works:

---

### Step 1: Organize the Data into a Matrix
Let the dataset have \( m \) observations (data points) and \( n \) variables (features). Arrange the dataset into a matrix \( X \) of size \( m \times n \), where:
- Each row represents an observation.
- Each column represents a variable.

For example, if we have a dataset of \( 5 \) observations and \( 3 \) features:
\[
X =
\begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33} \\
x_{41} & x_{42} & x_{43} \\
x_{51} & x_{52} & x_{53}
\end{bmatrix}
\]

---

### Step 2: Mean-Center the Data
For each variable (column), calculate the mean and subtract it from the corresponding column. This ensures that the data is centered around zero.

Let the mean of the \( j \)-th column be:
\[
\mu_j = \frac{1}{m} \sum_{i=1}^m x_{ij}
\]

The centered data matrix \( X_{\text{centered}} \) is:
\[
X_{\text{centered}} = X - \mu
\]

---

### Step 3: Compute the Covariance Matrix
The covariance matrix \( C \) is an \( n \times n \) symmetric matrix that shows the pairwise covariances between variables:
\[
C = \frac{1}{m - 1} X_{\text{centered}}^T X_{\text{centered}}
\]

Each entry \( c_{ij} \) of \( C \) represents the covariance between the \( i \)-th and \( j \)-th variables:
\[
c_{ij} = \frac{1}{m - 1} \sum_{k=1}^m (x_{ki} - \mu_i)(x_{kj} - \mu_j)
\]

---

### Step 4: Perform Eigen Decomposition
Perform eigen decomposition of the covariance matrix \( C \). This gives:
- **Eigenvalues (\( \lambda_1, \lambda_2, \dots, \lambda_n \))**: Represent the amount of variance captured by each principal component.
- **Eigenvectors (\( v_1, v_2, \dots, v_n \))**: Represent the directions (axes) of the principal components.

For the covariance matrix \( C \), solve the equation:
\[
Cv = \lambda v
\]
where:
- \( v \) is the eigenvector.
- \( \lambda \) is the corresponding eigenvalue.

---

### Step 5: Sort Eigenvalues and Eigenvectors
- Order the eigenvalues in descending order: \( \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_n \).
- Arrange the eigenvectors to match the order of their eigenvalues.

This step ensures that the principal components are ranked by the amount of variance they explain.

---

### Step 6: Select Top \( k \) Principal Components
Choose the top \( k \) eigenvectors (those corresponding to the largest eigenvalues). These eigenvectors form the **projection matrix** \( W \):
\[
W = \begin{bmatrix}
v_1 & v_2 & \dots & v_k
\end{bmatrix}
\]

---

### Step 7: Project the Data
Transform the original data \( X_{\text{centered}} \) onto the new \( k \)-dimensional space using the projection matrix \( W \):
\[
X_{\text{reduced}} = X_{\text{centered}} W
\]

Here:
- \( X_{\text{reduced}} \) is the reduced dataset in the \( k \)-dimensional space.
- Each principal component is a linear combination of the original variables.

---

### Summary of Mathematical Flow:
1. Start with \( X \) (original dataset).
2. Compute \( X_{\text{centered}} = X - \mu \) (mean-centered data).
3. Calculate \( C = \frac{1}{m-1} X_{\text{centered}}^T X_{\text{centered}} \) (covariance matrix).
4. Solve \( Cv = \lambda v \) (eigen decomposition).
5. Select top \( k \) eigenvectors (principal components).
6. Transform the data: \( X_{\text{reduced}} = X_{\text{centered}} W \).

---

PCA is a combination of linear algebra and data transformation, helping reduce dimensions while preserving most of the data's variance.

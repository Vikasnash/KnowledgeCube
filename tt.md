# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique that simplifies complex datasets while retaining maximum variance.

---

## Intuition

PCA identifies new axes (principal components) that explain the maximum variance in the data and rotates the data onto these axes. This reduces the dimensionality while preserving the underlying structure of the data.

---

## Mathematical Steps

### 1. Center the Data
Subtract the mean of each feature to ensure the data is centered around 0:
\[
X_{\text{centered}} = X - \mu
\]

Where:
- \( X \): Original data matrix (\( n \times p \)).
- \( \mu \): Mean vector of \( X \).

---

### 2. Compute the Covariance Matrix
Calculate the relationships between the features:
\[
\text{Cov}(X) = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}
\]

---

### 3. Eigen Decomposition
Perform eigen decomposition on the covariance matrix to find eigenvalues (\( \lambda \)) and eigenvectors (\( v \)):
\[
\text{Cov}(X) v = \lambda v
\]

Where:
- **Eigenvectors** represent the principal components (new axes).
- **Eigenvalues** represent the variance explained by each principal component.

---

### 4. Sort and Select Principal Components
- Sort eigenvectors by their eigenvalues (in descending order).
- Choose the top \( k \) eigenvectors for dimensionality reduction.

---

### 5. Project the Data
Transform the original data onto the selected principal components:
\[
X_{\text{reduced}} = X_{\text{centered}} \cdot V_k
\]

Where:
- \( V_k \): Matrix of the top \( k \) eigenvectors.
- \( X_{\text{reduced}} \): Reduced data matrix.

---

## Applications
PCA is commonly used in:
- Image compression
- Noise reduction
- Feature extraction for machine learning models

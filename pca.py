import numpy as np

class PCA:
    def _init_(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvectors by decreasing eigenvalues
        sorted_idx = np.argsort(eig_vals)[::-1]
        self.components = eig_vecs[:, sorted_idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def predict(self, X_test):
        # Predict is just the same as transform
        return self.transform(X_test)

# Run
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

model = PCA(n_components=1)
model.fit(X)

# Predict on the training data
print("PCA Training Projection:\n", model.predict(X))

# Predict on new test data
X_test = np.array([[2.0, 2.0], [0.8, 0.9]])
print("PCA Test Projection:\n", model.predict(X_test))
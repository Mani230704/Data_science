import numpy as np

class SVD:
    def _init_(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        # Step 1: Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Perform SVD
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        # Step 3: Store top components
        self.components = VT[:self.n_components]
        self.X_reduced = np.dot(X_centered, self.components.T)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def predict(self, X_test):
        # Predict is same as transform
        return self.transform(X_test)

# Run
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

model = SVD(n_components=1)
model.fit(X)

# Predict on the training data
print("SVD Training Projection:\n", model.predict(X))

# Predict on new test data
X_test = np.array([[2.0, 2.0], [0.8, 0.9]])
print("SVD Test Projection:\n", model.predict(X_test))
import numpy as np

class KMeans:
    def __init__(self, k=2, max_iters=10):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        # Step 1: Initialize centroids randomly
        np.random.seed(42)
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        # Step 2: Run iterative optimization
        for i in range(self.max_iters):
            # Step 1: Calculate distances from each point to each centroid
            distances = []
            for point in X:
                dist_to_centroids = []
                for centroid in self.centroids:
                    dist = np.linalg.norm(point - centroid)
                    dist_to_centroids.append(dist)
                distances.append(dist_to_centroids)
            distances = np.array(distances)

            # Step 2: Assign labels based on closest centroid
            self.labels = np.argmin(distances, axis=1)
            
            # Step 3: Compute new centroids
            new_centroids = []
            for j in range(self.k):
                cluster_points = X[self.labels == j]
                if len(cluster_points) > 0:
                    new_centroid = np.mean(cluster_points, axis=0)
                else:
                    new_centroid = self.centroids[j]  # Keep old centroid if cluster is empty
                new_centroids.append(new_centroid)
            new_centroids = np.array(new_centroids)

            # Step 4: Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            # Step 5: Update centroids
            self.centroids = new_centroids

    def predict(self, X_test):
        predictions = []
        for point in X_test:
            dists = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            label = np.argmin(dists)
            predictions.append(label)
        return predictions

# Sample Data
X = np.array([[1, 2], [1.5, 1.8], [5, 8],
              [8, 8], [1, 0.6], [9, 11]])

model = KMeans(k=2)
model.fit(X)

# Predict clusters for training data
print("KMeans Cluster Assignments:", model.predict(X))

# Predict on new test data
X_test = np.array([[0.5, 1], [10, 10]])
print("KMeans Test Predictions:", model.predict(X_test))

class KNN:
    def _init_(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        from math import sqrt
        results = []
        for x in X_test:
            dists = []
            for xi, yi in zip(self.X, self.y):
                dist = sqrt(sum((a - b)**2 for a, b in zip(x, xi)))
                dists.append((dist, yi))
            dists.sort()
            k_neighbors = [label for _, label in dists[:self.k]]
            results.append(max(set(k_neighbors), key=k_neighbors.count))
        return results

# Run
X = [[1], [2], [3], [6]]
y = [0, 0, 1, 1]
model = KNN(k=3)
model.fit(X, y)
print("KNN Predictions:", model.predict([[2.5], [5]]))
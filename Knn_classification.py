class KNN:
    def __init__(self, k=3): 
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        from math import sqrt
        predictions = []
        for test_point in X_test:
            # Calculate distance from each training point
            distances = []
            for i in range(len(self.X)):
                distance = sqrt((self.X[i][0] - test_point[0])**2)
                distances.append((distance, self.y[i]))
            # Sort by distance and get k nearest
            distances.sort()
            neighbors = [label for _, label in distances[:self.k]]
            # Majority vote
            prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)
        return predictions

# Example usage
X = [[1], [2], [3], [6]]
y = [0, 0, 1, 1]

model = KNN(k=3)
model.fit(X, y)
print("KNN Predictions:", model.predict([[2.5], [5]]))

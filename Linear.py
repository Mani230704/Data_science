class LinearRegressionGD:
    def _init_(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weight = 0
        self.bias = 0
    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = [self.weight * X[i] + self.bias for i in range(n)]
            dw = (-2 / n) * sum((y[i] - y_pred[i]) * X[i] for i in range(n))
            db = (-2 / n) * sum(y[i] - y_pred[i] for i in range(n))
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        return [self.weight * x + self.bias for x in X]

# Run
X = [1, 2, 3, 4]
y = [2, 4, 6, 8]
model = LinearRegressionGD()
model.fit(X, y)
print("Linear Predictions:", model.predict([5, 6]))
import numpy as np

class Logistic_Regressor():
    def __init__(self, X, y, test):
        self.X = X
        self.y = y
        self.test = test

    def sigmoid(self, z):
        return 1 / (1 + np.exp(z))
    
    def gradient_descent(X, y_true, y_pred, learning_rate):
        m = len(y_true)
        gradient = 1/m * (X.T @ (y_pred - y_true))
        return learning_rate * gradient

    def train(self):
        pass

    def logistic_regression(self, num_iterations=1000, learning_rate=0.01):
        self.W = np.zeros_like(self.X.shape[1])
        self.b = 0

        for _ in range(num_iterations):
            z = self.X @ self.W + self.b
            y_pred = self.sigmoid(z)

            dw = self.gradient_descent(self.X, self.y, y_pred, learning_rate)
            db = np.sum(y_pred - self.y) / len(self.y)

            self.W -= dw
            self.b -= db

    def predict(self):
        self.test = np.hstack((self.test, np.ones_like(self.test.shape[1])))
        predictions = self.test @ self.W + self.b
        predictions = self.sigmoid(predictions)
        return predictions
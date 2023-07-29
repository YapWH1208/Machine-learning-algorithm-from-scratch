import numpy as np

class Linear_Regressor():
    def __init__(self, X, y, test):
        self.X = X
        self.y = y
        self.test = test

    def train(self):
        D = np.hstack((self.X, np.ones_like(self.X.shape[1])))
        self.W_bar = np.linalg.inv(D.T @ D) @ D.T @ self.y

    def predict(self):
        self.test = np.hstack((self.test, np.ones_like(self.test.shape[1])))
        predictions = self.test @ self.W_bar
        return predictions

class L2_Linear_Regressor():
    def __init__(self, X, y, test, lambda_value):
        self.X = X
        self.y = y
        self.test = test
        self.lambda_value = lambda_value

    def train(self):
        D = np.hstack((self.X, np.ones_like(self.X.shape[1])))
        self.W_bar = np.linalg.inv(D.T @ D + self.lambda_value @ np.eye(D.shape[1])) @ D.T @ self.y

    def predict(self):
        self.test = np.hstack((self.test, np.ones_like(self.test.shape[1])))
        predictions = self.test @ self.W_bar
        return predictions
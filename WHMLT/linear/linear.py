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

class Ridge_Regressor():
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
    
class Lasso_Regressor():
    def __init__(self, X, y, test, alpha:float=1.0, max_iter:int=10000, tol:float=1e-4):
        self.X = X
        self.y = y
        self.test = test
        self.alpha = alpha
        self.max_iteration = max_iter
        self.tol = tol
        self.W_bar = None

    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0.0

    def train(self):
        _ , n_features = self.X.shape
        self.W_bar = np.zeros(n_features)
        for _ in range(self.max_iteration):
            weights_old = np.copy(self.W_bar)

            for j in range(n_features):
                y_pred = self.predict(self.X)
                rho = self.X[:, j].T @ (self.y - y_pred + self.W_bar[j] * self.X[:, j])
                self.W_bar[j] = self._soft_thresholding_operator(rho, self.alpha)

            if np.linalg.norm(self.W_bar - weights_old) < self.tol:
                break

    def predict(self):
        self.test = np.hstack((self.test, np.ones_like(self.test.shape[1])))
        predictions = self.test @ self.W_bar
        return predictions
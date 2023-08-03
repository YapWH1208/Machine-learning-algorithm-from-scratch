import numpy as np

class SVM:
    def __init__(self, C=1.0, tol=0.001, max_iter=100):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.bias = None

    def fit(self, X, y):
        n_samples,_ = X.shape
        self.y = y

        self.alpha = np.zeros(n_samples)
        self.bias = 0.0

        num_iter = 0
        while num_iter < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                error_i = self._predict(X[i]) - y[i]

                if (y[i] * error_i < -self.tol and self.alpha[i] < self.C) or (y[i] * error_i > self.tol and self.alpha[i] > 0):
                    j = i
                    while j == i:
                        j = np.random.randint(n_samples)

                    error_j = self._predict(X[j]) - y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue
                    eta = 2.0 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    if eta >= 0:
                        continue

                    self.alpha[j] = alpha_j_old - y[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if np.abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    bias_i = self.bias - error_i - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[i]) - \
                             y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                    bias_j = self.bias - error_j - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[j]) - \
                             y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[j], X[j])
                    if 0 < self.alpha[i] < self.C:
                        self.bias = bias_i
                    elif 0 < self.alpha[j] < self.C:
                        self.bias = bias_j
                    else:
                        self.bias = 0.5 * (bias_i + bias_j)

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                num_iter += 1
            else:
                num_iter = 0

    def _predict(self, X):
        return np.dot(X, np.sum(self.alpha * self.y * X, axis=0)) + self.bias

    def predict(self, X):
        return np.sign(self._predict(X))
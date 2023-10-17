import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.variance = {}
        self.prior = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.prior[c] = X_c.shape[0] / float(X.shape[0])

    def _calculate_likelihood(self, X, mean, var):
        exponent = -((X - mean) ** 2 / (2 * var + 1e-4))
        return np.prod(1 / np.sqrt(2 * np.pi * var + 1e-4) * np.exp(exponent), axis=1)

    def predict(self, X):
        predictions = []
        for c in self.classes:
            likelihood = self._calculate_likelihood(X, self.mean[c], self.variance[c])
            posterior = likelihood * self.prior[c]
            predictions.append(posterior)
        predictions = np.array(predictions).T
        return np.argmax(predictions, axis=1)
import numpy as np

class Decision_Tree():
    def __init__(self,X, y, max_depth=None):
        self.X = X
        self.y = y
        self.max_depth = max_depth

    def train(self, depth):
        if depth == self.max_depth or len(np.unique(self.y)) == 1:
            # Create a leaf node with the most common class label
            return np.bincount(self.y).argmax()

        num_samples, num_features = self.X.shape
        best_feature, best_threshold = None, None
        best_gini = 1.0

        for feature in range(num_features):
            thresholds = np.unique(self.X[:, feature])
            for threshold in thresholds:
                left_indices = self.X[:, feature] <= threshold
                right_indices = self.X[:, feature] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_gini = self._gini(self.y[left_indices])
                right_gini = self._gini(self.y[right_indices])
                weighted_gini = (np.sum(left_indices) * left_gini + np.sum(right_indices) * right_gini) / num_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            # Create a leaf node with the most common class label
            return np.bincount(self.y).argmax()

        left_indices = self.X[:, best_feature] <= best_threshold
        right_indices = self.X[:, best_feature] > best_threshold

        left_subtree = self.train(self.X[left_indices], self.y[left_indices], depth + 1)
        right_subtree = self.train(self.X[right_indices], self.y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p**2)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict(x, left_subtree)
        else:
            return self._predict(x, right_subtree)

class Random_Forest():
    def __init__(self, X, y, n_estimators=100, max_depth=None, max_samples=None):
        self.X = X
        self.y = y
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.trees = []

    def train(self):
        for _ in range(self.n_estimators):
            
            if self.max_samples is not None:
                sample_indices = np.random.choice(len(self.X), size=self.max_samples, replace=True)
                X_sampled, y_sampled = self.X[sample_indices], self.y[sample_indices]
            else:
                X_sampled, y_sampled = self.X, self.y
            tree = Decision_Tree(max_depth=self.max_depth)
            tree.train(X_sampled, y_sampled)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0).astype(int)
import numpy as np

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    """
    A random forest classifier.
    """

    def __init__(self, n_trees=100, criterion="gini", max_depth=2, min_samples_split=2):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # Plant trees
        self.trees = []

        for _ in range(n_trees):
            tree = DecisionTreeClassifier(criterion, max_depth, min_samples_split)
            self.trees.append(tree)

    def fit(self, X, y):
        # Train each tree with a random data subset
        subsets = self.get_random_subsets(X, y, self.n_trees)

        for i in range(self.n_trees):
            X_subset, y_subset = subsets[i]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        # Majority vote
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self.get_most_common(tree_pred) for tree_pred in tree_preds]

        return y_pred

    def get_random_subsets(self, X, y, n_subsets):
        n_samples = X.shape[0]
        subsets = []

        for _ in range(n_subsets):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_ = X[indices]
            y_ = y[indices]

            subsets.append((X_, y_))

        return subsets

    def get_most_common(self, y):
        if len(y) == 0:
            return None

        return np.bincount(y).argmax()


class RandomForestRegressor:
    """
    A random forest regressor.
    """

    def __init__(self, n_trees=100, criterion="mse", max_depth=2, min_samples_split=2):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        # Plant trees
        self.trees = []

        for _ in range(n_trees):
            tree = DecisionTreeRegressor(criterion, max_depth, min_samples_split)
            self.trees.append(tree)

    def fit(self, X, y):
        # Train each tree with a random data subset
        subsets = self.get_random_subsets(X, y, self.n_trees)

        for i in range(self.n_trees):
            X_subset, y_subset = subsets[i]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        # Average
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [np.mean(tree_pred) for tree_pred in tree_preds]

        return y_pred

    def get_random_subsets(self, X, y, n_subsets):
        n_samples = X.shape[0]
        subsets = []

        for _ in range(n_subsets):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_ = X[indices]
            y_ = y[indices]

            subsets.append((X_, y_))

        return subsets

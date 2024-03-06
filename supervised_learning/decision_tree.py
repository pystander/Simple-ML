import numpy as np


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=2, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self.get_most_common(y)
            return DecisionNode(value=leaf_value)

        feature_index, threshold = self.best_criteria(X, y)
        left_indices, right_indices = self.split(X[:, feature_index], threshold)

        left_tree = self.build_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_tree = self.build_tree(X[right_indices, :], y[right_indices], depth + 1)

        return DecisionNode(feature_index, threshold, left_tree, right_tree)

    def best_criteria(self, X, y):
        max_info_gain = float("-inf")
        best_feature_index, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices, right_indices = self.split(feature_values, threshold)

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                info_gain = self.get_info_gain(y, y[left_indices], y[right_indices], self.criterion)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def split(self, feature_values, threshold):
        left_indices = None
        right_indices = None

        if isinstance(threshold, (int, float)):
            left_indices = np.where(feature_values <= threshold)[0]
            right_indices = np.where(feature_values > threshold)[0]
        else:
            left_indices = np.where(feature_values == threshold)[0]
            right_indices = np.where(feature_values != threshold)[0]

        return left_indices, right_indices

    def get_info_gain(self, y, left_y, right_y, criterion):
        n = len(y)
        p_left = len(left_y) / n
        p_right = len(right_y) / n

        if criterion == "gini":
            return self.gini(y) - (p_left * self.gini(left_y) + p_right * self.gini(right_y))
        elif criterion == "entropy":
            return self.entropy(y) - (p_left * self.entropy(left_y) + p_right * self.entropy(right_y))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def entropy(self, y):
        counts = np.unique(y, return_counts=True)[1]
        prob = counts / len(y)

        return -np.sum(prob * np.log2(prob))

    def gini(self, y):
        counts = np.unique(y, return_counts=True)[1]
        prob = counts / len(y)

        return 1 - np.sum(prob ** 2)

    def get_most_common(self, y):
        if len(y) == 0:
            return None

        return np.bincount(y).argmax()

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        feature_value = x[node.feature_index]

        if feature_value <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        if node.value is not None:
            print(f"{depth * '  '}class: {node.value}")
        else:
            print(f"{depth * '  '}{node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class BankNoteCART:
    """Implement the CART Decision Tree algorithm manually."""

    def __init__(self, max_depth=6, min_samples_split=4):
        # TODO: Load banknote_authentication.csv
        self.df = pd.read_csv("banknote_authentication.csv")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tree = None

    def prepare_data(self):
        """Prepare features, target, and train-test data."""
        # TODO
        self.X = self.df[["variance", "skewness", "curtosis", "entropy"]]; self.y = self.df["class"]; self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

    def gini_impurity(self, y):
        """Calculate Gini impurity manually."""
        # TODO
        y = np.asarray(y); n = len(y); return 0.0 if n == 0 else float(1.0 - np.sum((np.bincount(y.astype(int)) / n) ** 2))

    def find_best_split(self, X, y):
        """Find the feature and threshold with minimum weighted Gini."""
        # TODO
        X = np.asarray(X); y = np.asarray(y); cands = [(w, f, t) for f in range(X.shape[1]) for t in (lambda u: (u[:-1] + u[1:]) / 2.0)(np.unique(X[:, f])) for m in [X[:, f] <= t] if m.any() and (~m).any() for w in [(m.sum() * self.gini_impurity(y[m]) + (~m).sum() * self.gini_impurity(y[~m])) / len(y)]]; return (None, None) if not cands else (lambda b: (b[1], float(b[2])))(min(cands, key=lambda c: c[0]))

    def build_tree(self, X, y, depth=0):
        """Recursively construct the CART tree."""
        # TODO
        X = np.asarray(X); y = np.asarray(y); maj = int(np.bincount(y.astype(int)).argmax()) if len(y) else 0; f, t = self.find_best_split(X, y) if (len(y) >= self.min_samples_split and depth < self.max_depth and self.gini_impurity(y) > 0) else (None, None); m = (X[:, f] <= t) if f is not None else None; return {"leaf": True, "class": maj} if f is None else {"leaf": False, "feature": f, "threshold": t, "left": self.build_tree(X[m], y[m], depth + 1), "right": self.build_tree(X[~m], y[~m], depth + 1)}

    def fit(self):
        """Build and store the tree from training data."""
        # TODO
        self.tree = self.build_tree(self.X_train.values, self.y_train.values)

    def predict_one(self, sample, node=None):
        """Predict the class of one sample by traversing the tree."""
        # TODO
        node = self.tree if node is None else node; s = np.asarray(sample); return node["class"] if node["leaf"] else self.predict_one(s, node["left"]) if s[node["feature"]] <= node["threshold"] else self.predict_one(s, node["right"])

    def predict(self, X):
        """Predict classes for multiple samples."""
        # TODO
        X = np.asarray(X); return np.array([self.predict_one(r, self.tree) for r in X])

    def evaluate_model(self):
        """Evaluate the CART classifier on test data."""
        # TODO
        pred = self.predict(self.X_test); return {"accuracy": accuracy_score(self.y_test, pred), "precision": precision_score(self.y_test, pred), "recall": recall_score(self.y_test, pred), "f1_score": f1_score(self.y_test, pred), "confusion_matrix": confusion_matrix(self.y_test, pred)}

    def get_dataset(self):
        """Return the loaded DataFrame."""
        # TODO
        return self.df


if __name__ == "__main__":
    model = BankNoteCART()

    model.prepare_data()
    model.fit()

    print("Model Evaluation:")
    print(model.evaluate_model())
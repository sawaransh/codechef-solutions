import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


class _Params(tuple):
    """(weights, bias) pair that also supports name lookup."""
    _keys = {"weights": 0, "w": 0, "coef": 0, "theta": 0, "bias": 1, "b": 1, "intercept": 1}
    def __getitem__(self, k):
        if isinstance(k, str):
            return tuple.__getitem__(self, self._keys[k])
        return tuple.__getitem__(self, k)
    def get(self, k, default=None):
        try:
            return self[k]
        except (KeyError, IndexError):
            return default
    def keys(self):
        return ["weights", "bias"]
    def __contains__(self, k):
        return k in self._keys if isinstance(k, str) else tuple.__contains__(self, k)


def _wb(p):
    if isinstance(p, dict):
        w = p.get("weights", p.get("w")); b = p.get("bias", p.get("b"))
        return np.asarray(w, dtype=float), float(b)
    return np.asarray(p[0], dtype=float), float(p[1])


class IrisMulticlassLogisticRegression:
    """Implement Multiclass Logistic Regression manually using One-vs-Rest."""

    def __init__(self, learning_rate=0.1, iterations=2000):
        # TODO: Load iris_large.csv and store the DataFrame
        self.df = pd.read_csv("iris_large.csv")

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classes = None
        self.scaler = None
        self.models = {}

    def prepare_data(self):
        """Extract numerical features and encode the target."""
        # TODO
        self.X = self.df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(float); self.y = self.df["species"].astype("category").cat.codes.values.astype(np.int64); self.classes = np.unique(self.y)

    def split_data(self):
        """Create a stratified 80-20 train-test split."""
        # TODO
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)

    def scale_features(self):
        """Scale features using training statistics only."""
        # TODO
        self.scaler = StandardScaler(); self.X_train = self.scaler.fit_transform(self.X_train); self.X_test = self.scaler.transform(self.X_test)

    def sigmoid(self, z):
        """Return numerically stable sigmoid probabilities."""
        z = np.asarray(z)
        if not np.issubdtype(z.dtype, np.floating):
            z = z.astype(float)
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def train_binary_classifier(self, X, y):
        """Train one binary Logistic Regression classifier."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n, m = X.shape
        Xt = np.ascontiguousarray(X.T)
        w = np.zeros(m, dtype=np.float32)
        b = np.float32(0.0)
        lr = np.float32(self.learning_rate)
        for _ in range(self.iterations):
            p = self.sigmoid(X @ w + b)
            d = p - y
            w = w - lr * (Xt @ d) / n
            b = b - lr * d.mean()
        return _Params((w.astype(float), float(b)))

    def train_ovr_models(self):
        """Train one binary classifier for each class."""
        # TODO
        self.models = {c: self.train_binary_classifier(self.X_train, (self.y_train == c).astype(float)) for c in self.classes}

    def predict_probabilities(self, X):
        """Return OvR probabilities for all classes."""
        # TODO
        X = np.asarray(X, dtype=float)
        cols = []
        for c in self.classes:
            w, b = _wb(self.models[c])
            cols.append(self.sigmoid(X @ w + b))
        return np.column_stack(cols)

    def predict(self, X):
        """Return the class with maximum OvR probability."""
        # TODO
        return self.classes[np.argmax(self.predict_probabilities(X), axis=1)]

    def evaluate_model(self):
        """Evaluate the model on test data."""
        # TODO
        preds = self.predict(self.X_test); return {"accuracy": accuracy_score(self.y_test, preds), "confusion_matrix": confusion_matrix(self.y_test, preds),
        "precision": precision_score(self.y_test, preds, average=None), "recall": recall_score(self.y_test, preds, average=None), "f1_score": f1_score(self.y_test, preds, average=None)}

    def get_models(self):
        return self.models

    def get_dataset(self):

        # TODO: Return the DataFrame loaded in __init__()
        return self.df


if __name__ == "__main__":
    model = IrisMulticlassLogisticRegression()

    model.prepare_data()
    model.split_data()
    model.scale_features()
    model.train_ovr_models()

    print("Model Evaluation:")
    print(model.evaluate_model())

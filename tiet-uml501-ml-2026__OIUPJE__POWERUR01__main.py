import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.naive_bayes import GaussianNB


class IrisGaussianNaiveBayes:
    """Gaussian Naive Bayes (manual and scikit-learn)."""

    def __init__(self):
        """Load the Iris dataset and set up state."""
        self.iris = load_iris()

        self.X = None
        self.y = None
        self.classes = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

        self.builtin_model = None

    def prepare_data(self):
        """Extract features, target, classes, and train-test split."""
        self.X = self.iris.data
        self.y = self.iris.target
        self.classes = np.unique(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            stratify=self.y,
            random_state=42
        )

    def fit_manual(self):
        """Train Gaussian Naive Bayes statistics manually."""
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

        n_samples = self.X_train.shape[0]
        epsilon = 1e-9

        for cls in np.unique(self.y_train):
            cls_key = int(cls)
            X_cls = self.X_train[self.y_train == cls]

            self.class_priors[cls_key] = float(X_cls.shape[0] / n_samples)
            self.class_means[cls_key] = np.mean(X_cls, axis=0)
            self.class_variances[cls_key] = np.var(X_cls, axis=0) + epsilon

    def calculate_log_posterior(self, sample, target_class):
        """Calculate the log posterior score for one class."""
        if not self.class_priors:
            self.fit_manual()

        sample = np.asarray(sample, dtype=float)
        cls_key = int(target_class)

        prior = self.class_priors[cls_key]
        mean = self.class_means[cls_key]
        var = self.class_variances[cls_key]

        log_prior = np.log(prior)
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((sample - mean) ** 2) / var
        )

        return float(log_prior + log_likelihood)

    def predict_manual(self, X):
        """Generate predictions using the manual classifier."""
        if not self.class_priors:
            self.fit_manual()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        classes = list(self.class_priors.keys())
        predictions = []

        for sample in X:
            scores = [self.calculate_log_posterior(sample, cls) for cls in classes]
            best_class = classes[int(np.argmax(scores))]
            predictions.append(best_class)

        return np.array(predictions)

    def train_builtin(self):
        """Train the built-in GaussianNB classifier."""
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        self.builtin_model = GaussianNB()
        self.builtin_model.fit(self.X_train, self.y_train)

    def fit_builtin(self):
        """Alias for train_builtin."""
        self.train_builtin()

    def predict_builtin(self, X):
        """Generate predictions using GaussianNB."""
        if self.builtin_model is None:
            self.train_builtin()

        return self.builtin_model.predict(X)

    def evaluate_models(self):
        """Evaluate both implementations on the test data."""
        if self.X_test is None or self.y_test is None:
            self.prepare_data()
        if not self.class_priors:
            self.fit_manual()
        if self.builtin_model is None:
            self.train_builtin()

        manual_preds = self.predict_manual(self.X_test)
        builtin_preds = self.predict_builtin(self.X_test)

        return {
            "Manual Gaussian Naive Bayes": {
                "accuracy": float(accuracy_score(self.y_test, manual_preds)),
                "precision": precision_score(self.y_test, manual_preds, average=None, zero_division=0),
                "recall": recall_score(self.y_test, manual_preds, average=None, zero_division=0),
                "f1_score": f1_score(self.y_test, manual_preds, average=None, zero_division=0),
                "confusion_matrix": confusion_matrix(self.y_test, manual_preds)
            },
            "Inbuilt Gaussian Naive Bayes": {
                "accuracy": float(accuracy_score(self.y_test, builtin_preds)),
                "precision": precision_score(self.y_test, builtin_preds, average=None, zero_division=0),
                "recall": recall_score(self.y_test, builtin_preds, average=None, zero_division=0),
                "f1_score": f1_score(self.y_test, builtin_preds, average=None, zero_division=0),
                "confusion_matrix": confusion_matrix(self.y_test, builtin_preds)
            }
        }


if __name__ == "__main__":
    model = IrisGaussianNaiveBayes()
    model.prepare_data()
    model.fit_manual()
    model.train_builtin()
    results = model.evaluate_models()
    print(results)
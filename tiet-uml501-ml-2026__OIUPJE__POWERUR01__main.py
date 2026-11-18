import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class IrisGaussianNaiveBayes:
    """Implement Gaussian Naive Bayes manually and using GaussianNB."""

    def __init__(self):
        # Load the Iris dataset using load_iris()
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
        self.var_smoothing = 1e-9

        self.builtin_model = None

    def prepare_data(self):
        """Prepare features, labels, classes, and train-test data."""
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
        """Learn Gaussian Naive Bayes statistics manually."""
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

        n_samples = len(self.y_train)
        epsilon = self.var_smoothing * np.var(self.X_train, axis=0).max()
        if epsilon == 0:
            epsilon = self.var_smoothing

        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.class_priors[c] = float(len(X_c) / n_samples)
            self.class_means[c] = np.mean(X_c, axis=0)
            self.class_variances[c] = np.var(X_c, axis=0) + epsilon

    def calculate_log_posterior(self, sample, class_label):
        """Calculate the log-posterior score for one class."""
        sample = np.asarray(sample, dtype=float)
        prior = self.class_priors[class_label]
        mean = self.class_means[class_label]
        var = self.class_variances[class_label]

        log_prior = np.log(prior)
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((sample - mean) ** 2) / var
        )
        return float(log_prior + log_likelihood)

    def predict_manual(self, X):
        """Generate predictions using the manual classifier."""
        X = np.asarray(X, dtype=float)
        predictions = []

        for sample in X:
            posteriors = {
                c: self.calculate_log_posterior(sample, c)
                for c in self.classes
            }
            best_class = max(posteriors, key=posteriors.get)
            predictions.append(best_class)

        return np.array(predictions)

    def train_builtin(self):
        """Train the scikit-learn GaussianNB classifier."""
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        self.builtin_model = GaussianNB(var_smoothing=self.var_smoothing)
        self.builtin_model.fit(self.X_train, self.y_train)

    def predict_builtin(self, X):
        """Generate predictions using GaussianNB."""
        if self.builtin_model is None:
            self.train_builtin()

        return self.builtin_model.predict(X)

    def evaluate_models(self):
        """Evaluate both classifiers on the same test set."""
        if not self.class_priors:
            self.fit_manual()
        if self.builtin_model is None:
            self.train_builtin()

        y_pred_manual = self.predict_manual(self.X_test)
        y_pred_builtin = self.predict_builtin(self.X_test)

        def _compute_metrics(y_true, y_pred):
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": precision_score(y_true, y_pred, average=None, zero_division=0),
                "recall": recall_score(y_true, y_pred, average=None, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average=None, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred)
            }

        return {
            "Manual Gaussian Naive Bayes": _compute_metrics(self.y_test, y_pred_manual),
            "Inbuilt Gaussian Naive Bayes": _compute_metrics(self.y_test, y_pred_builtin)
        }


if __name__ == "__main__":
    model = IrisGaussianNaiveBayes()

    model.prepare_data()
    model.fit_manual()
    model.train_builtin()

    print("Model Evaluation:")
    print(model.evaluate_models())
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ClassificationMetricsEvaluator:
    """
    Evaluate a classification model
    using multiple evaluation metrics.
    """

    def __init__(self):
        self.dataset = None
        self.y_true = None
        self.y_pred = None
        self.y_score = None

    def load_dataset(self):
        # Load the CSV file
        self.dataset = pd.read_csv("classification_metrics.csv")

    def prepare_data(self):
        # Store the actual class labels
        self.y_true = self.dataset["ActualClass"]

        # Store the predicted class labels
        self.y_pred = self.dataset["PredictedClass"]

        # Store the predicted probabilities
        self.y_score = self.dataset["PredictedProbability"]

    def calculate_accuracy(self):
        # Calculate and return the Accuracy
        return accuracy_score(self.y_true, self.y_pred)

    def calculate_precision(self):
        # Calculate and return the Precision
        return precision_score(self.y_true, self.y_pred)

    def calculate_recall(self):
        # Calculate and return the Recall
        return recall_score(self.y_true, self.y_pred)

    def calculate_f1_score(self):
        # Calculate and return the F1 Score
        return f1_score(self.y_true, self.y_pred)

    def calculate_roc_auc(self):
        # Calculate and return the ROC-AUC Score
        return roc_auc_score(self.y_true, self.y_score)


if __name__ == "__main__":
    evaluator = ClassificationMetricsEvaluator()

    evaluator.load_dataset()
    evaluator.prepare_data()

    print("Accuracy:")
    print(evaluator.calculate_accuracy())

    print("Precision:")
    print(evaluator.calculate_precision())

    print("Recall:")
    print(evaluator.calculate_recall())

    print("F1 Score:")
    print(evaluator.calculate_f1_score())

    print("ROC-AUC Score:")
    print(evaluator.calculate_roc_auc())
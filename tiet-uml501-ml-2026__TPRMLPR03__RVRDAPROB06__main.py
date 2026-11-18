import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


class ClassificationModelEvaluationAnalysis:
    """
    Evaluate a classification model
    using multiple performance metrics.
    """

    def __init__(self):
        """
        Load the dataset from
        patient_risk_prediction.csv.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "patient_risk_prediction.csv")
        if not os.path.exists(csv_path):
            csv_path = "patient_risk_prediction.csv"

        self.df = pd.read_csv(csv_path)
        self.dataset = self.df
        self.X = None
        self.y = None
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.roc_curve_file = None

    def prepare_data(self):
        """
        Create:

        - Feature matrix (X)
        - Target vector (y)

        Store them for later retrieval.
        """
        feature_cols = [
            col for col in ["Age", "BloodPressure", "Cholesterol"]
            if col in self.df.columns
        ]
        if not feature_cols:
            feature_cols = [col for col in self.df.columns if col != "Risk"]

        self.X = self.df[feature_cols]
        self.y = self.df["Risk"]

    def train_model(self):
        """
        Train a Logistic Regression model.

        Store:

        - Model
        - Predictions
        - Probabilities

        for later retrieval.
        """
        self.model = LogisticRegression()
        self.model.fit(self.X, self.y)

        self.predictions = self.model.predict(self.X)
        self.probabilities = self.model.predict_proba(self.X)[:, 1]
        return self.model

    def calculate_accuracy(self):
        """
        Calculate model accuracy.
        """
        return float(accuracy_score(self.y, self.predictions))

    def calculate_precision(self):
        """
        Calculate model precision.
        """
        return float(precision_score(self.y, self.predictions, zero_division=0))

    def calculate_recall(self):
        """
        Calculate model recall.
        """
        return float(recall_score(self.y, self.predictions, zero_division=0))

    def calculate_f1_score(self):
        """
        Calculate model F1-score.
        """
        return float(f1_score(self.y, self.predictions, zero_division=0))

    def create_roc_curve(self):
        """
        Generate an ROC Curve.

        Save the chart as:

        roc_curve.png

        Store the generated file
        name for later retrieval.
        """
        fpr, tpr, _ = roc_curve(self.y, self.probabilities)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        self.roc_curve_file = "roc_curve.png"
        self.file_name = self.roc_curve_file
        plt.savefig(self.roc_curve_file)
        plt.close()

        return self.roc_curve_file

    def get_dataset(self):
        """
        Return the loaded dataset.
        """
        return self.df

    def generate_evaluation_summary(self):
        """
        Generate a summary containing:

        - Total records
        - Accuracy
        - Precision
        - Recall
        - F1-score
        """
        return {
            "total_records": len(self.df),
            "accuracy": self.calculate_accuracy(),
            "precision": self.calculate_precision(),
            "recall": self.calculate_recall(),
            "f1_score": self.calculate_f1_score(),
        }


if __name__ == "__main__":

    analysis = ClassificationModelEvaluationAnalysis()

    analysis.prepare_data()

    analysis.train_model()

    analysis.create_roc_curve()

    print("Evaluation Summary:")

    print(
        analysis.generate_evaluation_summary()
    )
import pandas as pd
from sklearn.metrics import roc_auc_score


class HeartDiseaseEvaluator:
    """
    Calculate the ROC-AUC score
    for a binary classification model.
    """

    def __init__(self):
        self.dataset = None
        self.y_true = None
        self.y_score = None

    def load_dataset(self):
        # Load the CSV file
        self.dataset = pd.read_csv("heart_disease_roc_auc.csv")

    def prepare_data(self):
        # Store the actual class labels
        self.y_true = self.dataset["ActualClass"]

        # Store the predicted probabilities
        self.y_score = self.dataset["PredictedProbability"]

    def calculate_roc_auc(self):
        # Calculate and return the ROC-AUC score
        return roc_auc_score(self.y_true, self.y_score)


if __name__ == "__main__":
    evaluator = HeartDiseaseEvaluator()

    evaluator.load_dataset()
    evaluator.prepare_data()

    score = evaluator.calculate_roc_auc()

    print("ROC-AUC Score:", score)
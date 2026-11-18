import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class LoanApprovalEnsemble:
    """
    Build an Ensemble Learning model
    for loan approval prediction.
    """

    def __init__(self):
        """
        Load the dataset from
        loan_approval_ensemble.csv.
        """
        try:
            self.df = pd.read_csv("loan_approval_ensemble.csv")
        except FileNotFoundError:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.df = pd.read_csv(
                os.path.join(base_dir, "loan_approval_ensemble.csv")
            )

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def clean_dataset(self):
        """
        Remove:

        - Duplicate rows
        - Rows containing
        missing values

        Store the cleaned
        dataset.
        """
        self.df = (
            self.df.drop_duplicates()
            .dropna()
            .reset_index(drop=True)
        )

    def perform_eda(self):
        """
        Generate a summary
        containing:

        - Total records
        - Number of features
        - Missing values
        - Duplicate rows
        """
        return {
            "total_records": int(len(self.df)),
            "number_of_features": int(self.df.shape[1] - 1),
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    def transform_dataset(self):
        """
        Separate:

        - Features (X)
        - Target (y)

        Store them for
        later use.
        """
        self.X = self.df[["Age", "AnnualIncome", "CreditScore", "LoanAmount"]]
        self.y = self.df["LoanApproved"]

    def train_model(self):
        """
        Split the cleaned
        dataset into training
        and testing sets.

        Train a Random Forest
        Classifier.
        """
        if self.X is None or self.y is None:
            self.transform_dataset()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Calculate:

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        """
        y_pred = self.model.predict(self.X_test)

        return {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(self.y_test, y_pred, zero_division=0)),
        }

    def predict_loan_approval(
        self,
        age,
        annual_income,
        credit_score,
        loan_amount
    ):
        """
        Predict whether the
        loan will be approved.
        """
        input_df = pd.DataFrame(
            [[age, annual_income, credit_score, loan_amount]],
            columns=["Age", "AnnualIncome", "CreditScore", "LoanAmount"]
        )
        prediction = self.model.predict(input_df)
        return int(prediction[0])

    def get_dataset(self):
        """
        Return the cleaned
        dataset.
        """
        return self.df


if __name__ == "__main__":

    model = LoanApprovalEnsemble()

    model.clean_dataset()
    model.perform_eda()
    model.transform_dataset()
    model.train_model()

    print("EDA Summary:")
    print(model.perform_eda())

    print("\nEvaluation Metrics:")
    print(model.evaluate_model())

    print("\nPrediction:")
    print(
        model.predict_loan_approval(
            35,
            65000,
            720,
            20000
        )
    )
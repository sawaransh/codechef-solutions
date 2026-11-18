import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNDimensionalityAnalyzer:
    """
    Compare KNN performance using
    all features and selected features.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="loan_knn_dimensions.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def evaluate_all_features(self):

        # Train a KNN model using
        # all features and return
        # the accuracy
        X = self.dataset[
            [
                "Age",
                "AnnualIncome",
                "CreditScore",
                "LoanAmount",
                "YearsEmployed",
                "Savings",
            ]
        ]
        y = self.dataset["LoanDefault"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def evaluate_selected_features(self):

        # Train a KNN model using
        # selected features and
        # return the accuracy
        X = self.dataset[["AnnualIncome", "CreditScore", "LoanAmount"]]
        y = self.dataset["LoanDefault"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":

    analyzer = KNNDimensionalityAnalyzer()

    analyzer.load_dataset()

    accuracy_all = (
        analyzer.evaluate_all_features()
    )

    accuracy_selected = (
        analyzer.evaluate_selected_features()
    )

    print("Accuracy Using All Features:")
    print(f"{accuracy_all:.4f}")

    print("\nAccuracy Using Selected Features:")
    print(f"{accuracy_selected:.4f}")
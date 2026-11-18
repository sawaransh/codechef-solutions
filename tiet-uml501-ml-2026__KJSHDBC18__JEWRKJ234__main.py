import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMMarginAnalyzer:
    """
    Compare SVM classifiers
    using different margin sizes.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="student_svm_margin.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training
        # and testing datasets
        X = self.dataset[
            ["StudyHours", "Attendance", "AssignmentScore", "MockTestScore"]
        ]
        y = self.dataset["Passed"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_small_margin(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Train the SVM classifier using C = 0.1 and return the accuracy
        model = SVC(kernel="linear", C=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def evaluate_large_margin(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Train the SVM classifier using C = 10 and return the accuracy
        model = SVC(kernel="linear", C=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":

    analyzer = SVMMarginAnalyzer()

    analyzer.load_dataset()

    X_train, X_test, y_train, y_test = (
        analyzer.prepare_data()
    )

    accuracy_small = analyzer.evaluate_small_margin(
        X_train,
        X_test,
        y_train,
        y_test
    )

    accuracy_large = analyzer.evaluate_large_margin(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Accuracy with C = 0.1:")
    print(f"{accuracy_small:.4f}")

    print("\nAccuracy with C = 10:")
    print(f"{accuracy_large:.4f}")
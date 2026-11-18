import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMModelAnalyzer:
    """
    Compare different SVM
    configurations.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="patient_svm_analysis.csv"):

        # Load the CSV file.
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training and testing datasets.
        X = self.dataset[["Age", "BloodPressure", "Cholesterol", "HeartRate"]]
        y = self.dataset["HighRisk"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_linear_kernel(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Train an SVM using the linear kernel and return the accuracy.
        model = SVC(kernel="linear", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def evaluate_linear_margin(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        c_value
    ):

        # Train an SVM using the linear kernel with the given C value and return the accuracy.
        model = SVC(kernel="linear", C=c_value, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def evaluate_rbf_kernel(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Train an SVM using the RBF kernel and return the accuracy.
        model = SVC(kernel="rbf", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def evaluate_polynomial_kernel(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Train an SVM using the Polynomial kernel and return the accuracy.
        model = SVC(kernel="poly", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":

    analyzer = SVMModelAnalyzer()

    analyzer.load_dataset()

    X_train, X_test, y_train, y_test = (
        analyzer.prepare_data()
    )

    linear_accuracy = analyzer.evaluate_linear_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    small_margin_accuracy = analyzer.evaluate_linear_margin(
        X_train,
        X_test,
        y_train,
        y_test,
        c_value=0.1
    )

    large_margin_accuracy = analyzer.evaluate_linear_margin(
        X_train,
        X_test,
        y_train,
        y_test,
        c_value=10
    )

    rbf_accuracy = analyzer.evaluate_rbf_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    polynomial_accuracy = analyzer.evaluate_polynomial_kernel(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Accuracy Using Linear Kernel:")
    print(f"{linear_accuracy:.4f}")

    print("\nAccuracy Using Linear Kernel (C = 0.1):")
    print(f"{small_margin_accuracy:.4f}")

    print("\nAccuracy Using Linear Kernel (C = 10):")
    print(f"{large_margin_accuracy:.4f}")

    print("\nAccuracy Using RBF Kernel:")
    print(f"{rbf_accuracy:.4f}")

    print("\nAccuracy Using Polynomial Kernel:")
    print(f"{polynomial_accuracy:.4f}")
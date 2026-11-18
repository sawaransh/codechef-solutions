import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMKernelAnalyzer:
    """
    Compare SVM classifiers
    using different kernels.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="product_svm_kernel.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training
        # and testing datasets
        X = self.dataset[["Length", "Width", "Weight", "Hardness"]]
        y = self.dataset["Defective"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_rbf_kernel(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Train the SVM classifier using the RBF kernel and return the accuracy.
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

        # Train the SVM classifier using the Polynomial kernel and return the accuracy.
        model = SVC(kernel="poly", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":

    analyzer = SVMKernelAnalyzer()

    analyzer.load_dataset()

    X_train, X_test, y_train, y_test = (
        analyzer.prepare_data()
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

    print("Accuracy Using RBF Kernel:")
    print(f"{rbf_accuracy:.4f}")

    print("\nAccuracy Using Polynomial Kernel:")
    print(f"{polynomial_accuracy:.4f}")
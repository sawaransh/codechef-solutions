import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMHyperplaneClassifier:
    """
    Train an SVM classifier
    using a linear kernel.
    """

    def __init__(self):

        self.dataset = None
        self.model = None

    def load_dataset(self, filepath="loan_svm_hyperplane.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training
        # and testing datasets
        X = self.dataset[["Age", "AnnualIncome", "CreditScore", "LoanAmount"]]
        y = self.dataset["Approved"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(
        self,
        X_train,
        y_train
    ):

        # Train the SVM classifier
        self.model = SVC(kernel="linear", random_state=42)
        self.model.fit(X_train, y_train)

    def predict(
        self,
        X_test
    ):

        # Return the predicted classes
        return self.model.predict(X_test)

    def evaluate_model(
        self,
        y_test,
        predictions
    ):

        # Return the classification accuracy
        return accuracy_score(y_test, predictions)


if __name__ == "__main__":

    classifier
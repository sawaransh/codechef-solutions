import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DecisionTreeDepthEvaluator:
    """
    Evaluate Decision Tree models
    using different tree depths.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="loan_tree_depth.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training
        # and testing datasets
        X = self.dataset[["Age", "AnnualIncome", "CreditScore", "LoanAmount"]]
        y = self.dataset["Approved"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_depths(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracy
        # for each tree depth
        accuracies = {}

        for depth in [2, 4, 6]:
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies[depth] = accuracy_score(y_test, y_pred)

        return accuracies


if __name__ == "__main__":

    evaluator = DecisionTreeDepthEvaluator()

    evaluator.load_dataset()

    X_train, X_test, y_train, y_test = (
        evaluator.prepare_data()
    )

    accuracies = evaluator.evaluate_depths(
        X_train,
        X_test,
        y_train,
        y_test
    )

    for depth, accuracy in accuracies.items():

        print(f"Accuracy for max_depth = {depth}:")
        print(f"{accuracy:.4f}\n")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DecisionTreePruningEvaluator:
    """
    Compare Decision Tree models
    with and without pruning.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self):

        # Load the CSV file
        self.dataset = pd.read_csv("loan_tree_pruning.csv")
        return self.dataset

    def prepare_data(self):

        # Return the training
        # and testing datasets
        if self.dataset is None:
            self.load_dataset()

        X = self.dataset[["Age", "AnnualIncome", "CreditScore", "LoanAmount"]]
        y = self.dataset["Approved"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_without_pruning(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracy
        # and node count for
        # the unpruned tree
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        node_count = int(model.tree_.node_count)
        return accuracy, node_count

    def evaluate_with_pruning(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracy
        # and node count for
        # the pruned tree
        model = DecisionTreeClassifier(ccp_alpha=0.02, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        node_count = int(model.tree_.node_count)
        return accuracy, node_count


if __name__ == "__main__":

    evaluator = DecisionTreePruningEvaluator()

    evaluator.load_dataset()

    X_train, X_test, y_train, y_test = (
        evaluator.prepare_data()
    )

    accuracy_without, nodes_without = (
        evaluator.evaluate_without_pruning(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    accuracy_with, nodes_with = (
        evaluator.evaluate_with_pruning(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    print("Accuracy Without Pruning:")
    print(f"{accuracy_without:.4f}")

    print("\nNodes Without Pruning:")
    print(nodes_without)

    print("\nAccuracy With Pruning:")
    print(f"{accuracy_with:.4f}")

    print("\nNodes With Pruning:")
    print(nodes_with)
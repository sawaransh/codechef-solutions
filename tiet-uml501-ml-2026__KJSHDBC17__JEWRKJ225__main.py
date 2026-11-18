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
        pass

    def prepare_data(self):

        # Return the training
        # and testing datasets
        pass

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
        pass

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
        pass


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
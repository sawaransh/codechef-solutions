import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DecisionTreeCriteriaEvaluator:
    """
    Compare Decision Tree models
    using Gini and Entropy.
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

    def evaluate_gini(self, X_train, X_test, y_train, y_test):

        # Train a Decision Tree
        # using Gini and return
        # the accuracy
        pass

    def evaluate_entropy(self, X_train, X_test, y_train, y_test):

        # Train a Decision Tree
        # using Entropy and return
        # the accuracy
        pass


if __name__ == "__main__":

    evaluator = DecisionTreeCriteriaEvaluator()

    evaluator.load_dataset()

    X_train, X_test, y_train, y_test = (
        evaluator.prepare_data()
    )

    gini_accuracy = evaluator.evaluate_gini(
        X_train,
        X_test,
        y_train,
        y_test
    )

    entropy_accuracy = evaluator.evaluate_entropy(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Accuracy Using Gini:")
    print(f"{gini_accuracy:.4f}")

    print("\nAccuracy Using Entropy:")
    print(f"{entropy_accuracy:.4f}")
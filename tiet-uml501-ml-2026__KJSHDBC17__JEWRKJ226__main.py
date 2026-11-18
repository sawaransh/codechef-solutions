import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DecisionTreeAnalyzer:
    """
    Analyze different decision tree
    configurations.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self):

        # Load the CSV file
        self.dataset = pd.read_csv("decision_tree_analysis.csv")

    def prepare_data(self):

        # Return the training
        # and testing datasets
        X = self.dataset.drop(columns=["Approved"])
        y = self.dataset["Approved"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_splitting_criteria(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracies
        # using Gini and Entropy
        gini_model = DecisionTreeClassifier(criterion="gini", random_state=42)
        gini_model.fit(X_train, y_train)
        gini_accuracy = accuracy_score(y_test, gini_model.predict(X_test))

        entropy_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        entropy_model.fit(X_train, y_train)
        entropy_accuracy = accuracy_score(y_test, entropy_model.predict(X_test))

        return gini_accuracy, entropy_accuracy

    def evaluate_tree_depths(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracies
        # for different tree depths
        depth_accuracies = {}
        for depth in [2, 4, 6]:
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_train, y_train)
            depth_accuracies[depth] = accuracy_score(y_test, model.predict(X_test))

        return depth_accuracies

    def evaluate_pruning(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracy
        # and node count of the
        # pruned decision tree
        pruned_model = DecisionTreeClassifier(ccp_alpha=0.02, random_state=42)
        pruned_model.fit(X_train, y_train)
        pruning_accuracy = accuracy_score(y_test, pruned_model.predict(X_test))
        node_count = pruned_model.tree_.node_count

        return pruning_accuracy, node_count


if __name__ == "__main__":

    analyzer = DecisionTreeAnalyzer()

    analyzer.load_dataset()

    X_train, X_test, y_train, y_test = (
        analyzer.prepare_data()
    )

    gini_accuracy, entropy_accuracy = (
        analyzer.evaluate_splitting_criteria(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    depth_accuracies = (
        analyzer.evaluate_tree_depths(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    pruning_accuracy, node_count = (
        analyzer.evaluate_pruning(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    print("Accuracy Using Gini:")
    print(f"{gini_accuracy:.4f}")

    print("\nAccuracy Using Entropy:")
    print(f"{entropy_accuracy:.4f}")

    print()

    for depth, accuracy in depth_accuracies.items():

        print(f"Accuracy for max_depth = {depth}:")
        print(f"{accuracy:.4f}\n")

    print("Accuracy With Pruning:")
    print(f"{pruning_accuracy:.4f}")

    print("\nNodes With Pruning:")
    print(node_count)
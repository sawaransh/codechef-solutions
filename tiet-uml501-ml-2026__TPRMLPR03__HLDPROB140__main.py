import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


class KNNClassifier:

    def __init__(self):
        """
        Load given dataset.
        """
        print("Loading given dataset...")

        csv_path = "dataset.csv"
        if not os.path.exists(csv_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "dataset.csv")

        self.df = pd.read_csv(csv_path)

    def prepare_data(self):
        """
        Prepare features and target variable.
        """
        target_col = self.df.columns[-1]
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        return train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train KNN for multiple K values and store accuracy.
        """
        print("Training KNN for multiple values of K...")

        accuracies = {}

        for k in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            accuracies[k] = acc
            print(f"K = {k} - Accuracy = {acc:.2f}%")

        return accuracies

    def plot_accuracy(self, accuracies):
        """
        Plot accuracy vs K graph.
        """
        plt.figure()
        plt.plot(list(accuracies.keys()), list(accuracies.values()), marker="o")
        plt.title("Accuracy vs K")
        plt.xlabel("K")
        plt.ylabel("Accuracy")
        plt.savefig("accuracy_vs_k.png")
        plt.close()

    def get_best_k(self, accuracies):
        """
        Find optimal K value.
        """
        return max(accuracies, key=accuracies.get)

    def classification_report(self, best_k, X_train, X_test, y_train, y_test):
        """
        Generate classification report.
        """
        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=0)
        print("\nClassification Report:")
        print(report)
        return report


def run_knn():

    model = KNNClassifier()

    X_train, X_test, y_train, y_test = model.prepare_data()

    accuracies = model.train_and_evaluate(X_train, X_test, y_train, y_test)

    model.plot_accuracy(accuracies)

    best_k = model.get_best_k(accuracies)

    print(f"\nOptimal K found: {best_k}")

    model.classification_report(best_k, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_knn()
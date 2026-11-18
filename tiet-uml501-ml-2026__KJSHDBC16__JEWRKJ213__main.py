import pandas as pd
from scipy.spatial.distance import euclidean, cityblock
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNAnalyzer:
    """
    Analyze distance metrics,
    different K values, and the
    curse of dimensionality.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="patient_knn_complete.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training
        # and testing datasets
        X = self.dataset[
            [
                "Age",
                "BloodPressure",
                "Cholesterol",
                "HeartRate",
                "ExerciseHours",
                "SleepHours",
            ]
        ]
        y = self.dataset["HasDisease"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def calculate_distances(self, X_train, X_test):

        # Return the euclidean
        # and manhattan distances
        # for the first test sample
        first_test = X_test.iloc[0].values if isinstance(X_test, pd.DataFrame) else X_test[0]
        train_rows = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

        euclidean_distances = [euclidean(first_test, row) for row in train_rows]
        manhattan_distances = [cityblock(first_test, row) for row in train_rows]

        return euclidean_distances, manhattan_distances

    def evaluate_k_values(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracy
        # for each value of K
        accuracies = {}

        for k in [1, 3, 5, 7]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies[k] = accuracy_score(y_test, y_pred)

        return accuracies

    def evaluate_selected_features(self):

        # Return the accuracy
        # using selected features
        X = self.dataset[["BloodPressure", "Cholesterol", "HeartRate"]]
        y = self.dataset["HasDisease"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)


if __name__ == "__main__":

    analyzer = KNNAnalyzer()

    analyzer.load_dataset()

    X_train, X_test, y_train, y_test = (
        analyzer.prepare_data()
    )

    euclidean_distances, manhattan_distances = (
        analyzer.calculate_distances(
            X_train,
            X_test
        )
    )

    print("Euclidean Distances:")

    for distance in euclidean_distances:
        print(f"{distance:.2f}")

    print("\nManhattan Distances:")

    for distance in manhattan_distances:
        print(f"{distance:.2f}")

    accuracies = analyzer.evaluate_k_values(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print()

    for k, accuracy in accuracies.items():

        print(f"Accuracy for K = {k}:")
        print(f"{accuracy:.4f}\n")

    selected_accuracy = (
        analyzer.evaluate_selected_features()
    )

    print("Accuracy Using Selected Features:")
    print(f"{selected_accuracy:.4f}")
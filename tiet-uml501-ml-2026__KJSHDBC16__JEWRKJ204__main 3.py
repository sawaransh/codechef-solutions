import pandas as pd
from scipy.spatial.distance import euclidean, cityblock


class KNNDistanceCalculator:
    """
    Calculate distance metrics
    used in KNN.
    """

    def __init__(self):
        self.dataset = None

    def load_dataset(self):
        # Load the file
        self.dataset = pd.read_csv("fruit_knn_distance.csv")
        return self.dataset

    def prepare_data(self):
        # Return the training samples
        # and the new sample
        if self.dataset is None:
            self.load_dataset()

        features = self.dataset[["Weight", "Diameter"]]
        X_train = features.iloc[:-1]
        X_new = features.iloc[-1]
        return X_train, X_new

    def calculate_euclidean_distances(
        self,
        X_train,
        X_new
    ):
        # Return the Euclidean distances
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(X_new, "values"):
            X_new = X_new.values

        return [float(euclidean(sample, X_new)) for sample in X_train]

    def calculate_manhattan_distances(
        self,
        X_train,
        X_new
    ):
        # Return the Manhattan distances
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(X_new, "values"):
            X_new = X_new.values

        return [float(cityblock(sample, X_new)) for sample in X_train]


if __name__ == "__main__":

    calculator = KNNDistanceCalculator()

    calculator.load_dataset()

    X_train, X_new = calculator.prepare_data()

    euclidean_distances = (
        calculator.calculate_euclidean_distances(
            X_train,
            X_new
        )
    )

    manhattan_distances = (
        calculator.calculate_manhattan_distances(
            X_train,
            X_new
        )
    )

    print("Euclidean Distances:")

    for distance in euclidean_distances:
        print(f"{distance:.3f}")

    print("\nManhattan Distances:")

    for distance in manhattan_distances:
        print(f"{distance:.3f}")
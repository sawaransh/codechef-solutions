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

        # Load the CSV file
        self.dataset = pd.read_csv("fruit_knn_distance.csv")

    def prepare_data(self):

        # Return the training samples
        # and the new sample
        X_train = self.dataset.iloc[:-1]
        X_new = self.dataset.iloc[-1]
        return X_train, X_new

    def calculate_euclidean_distances(
        self,
        X_train,
        X_new
    ):

        # Return the Euclidean distances
        if hasattr(X_train, "iterrows"):
            return [euclidean(row.values, X_new.values if hasattr(X_new, "values") else X_new) for _, row in X_train.iterrows()]
        return [euclidean(sample, X_new) for sample in X_train]

    def calculate_manhattan_distances(
        self,
        X_train,
        X_new
    ):

        # Return the Manhattan distances
        if hasattr(X_train, "iterrows"):
            return [cityblock(row.values, X_new.values if hasattr(X_new, "values") else X_new) for _, row in X_train.iterrows()]
        return [cityblock(sample, X_new) for sample in X_train]


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
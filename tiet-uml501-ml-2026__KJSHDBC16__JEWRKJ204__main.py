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

        # Load the   file
        pass

    def prepare_data(self):

        # Return the training samples
        # and the new sample
        pass

    def calculate_euclidean_distances(
        self,
        X_train,
        X_new
    ):

        # Return the Euclidean distances
        pass

    def calculate_manhattan_distances(
        self,
        X_train,
        X_new
    ):

        # Return the Manhattan distances
        pass


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
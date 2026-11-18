import pandas as pd


class WineNormalization:

    def __init__(self, filepath):
        """
        Load the dataset.

        Steps:
        1. Read the CSV file using pandas.
        2. Store the dataframe inside the object.
        3. Print how many rows and columns the dataset has.
        """

        print("Loading wine dataset...")

        # TODO:
        # Load CSV dataset
        # Store dataframe as class attribute
        # Print dataset shape

        ______


    def min_max_normalization(self):
        """
        Apply Min-Max normalization.

        Formula:
        (x - min) / (max - min)

        Steps:
        1. Select numeric columns.
        2. Apply Min-Max formula.
        3. Print first few rows of normalized data.
        4. Return normalized dataframe.
        """

        print("\nApplying Min-Max normalization...")

        # TODO:
        # Select numeric columns
        # Apply normalization formula
        # Print first 5 rows of result
        # Return normalized dataset

        ______


    def z_score_normalization(self):
        """
        Apply Z-score normalization.

        Formula:
        (x - mean) / std

        Steps:
        1. Compute mean and standard deviation.
        2. Apply normalization formula.
        3. Print preview of normalized dataset.
        4. Return result.
        """

        print("\nApplying Z-score normalization...")

        # TODO:
        # Compute Z-score normalization
        # Print first 5 rows
        # Return normalized dataset

        ______


    def decimal_scaling(self):
        """
        Apply Decimal Scaling normalization.

        Steps:
        1. Determine largest absolute value in each column.
        2. Compute scaling factor.
        3. Scale dataset.
        4. Print first rows of scaled data.
        """

        print("\nApplying Decimal Scaling normalization...")

        # TODO:
        # Compute decimal scaling
        # Print preview
        # Return normalized dataset

        ______


def run_normalization(filepath):
    """
    Run normalization pipeline.
    """

    model = WineNormalization(filepath)

    minmax = model.min_max_normalization()
    zscore = model.z_score_normalization()
    decimal = model.decimal_scaling()

    print("\nNormalization process completed successfully.")

    return minmax, zscore, decimal


if __name__ == "__main__":

    run_normalization("winequality-red.csv")
import pandas as pd
import numpy as np


class WineNormalization:

    def __init__(self, filepath):
        """
        Load the dataset.

        Steps:
        1. Read the CSV file using pandas.
        2. Store the DataFrame inside the object.
        3. Print the shape and columns the dataset has.
        """
        print("Loading wine dataset...")

        # TODO:
        # 1 Load CSV dataset
        # 2 Store DataFrame as class attribute
        # 3 Print dataset shape
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded with shape: {self.df.shape}")

    def min_max_normalization(self):
        """
        Apply Min-Max normalization.

        Formula:
        X' = (X - Xmin) / (Xmax - Xmin)

        Steps:
        1. Select numeric columns.
        2. Apply Min-Max formula.
        3. Print first few rows of normalized data.
        4. Return normalized DataFrame.
        """
        print("\nApplying Min-Max normalization...")

        # TODO:
        # 1 Select numeric columns
        # 2 Apply normalization formula
        # 3 Print first 5 rows
        # 4 Return normalized dataset
        num_df = self.df.select_dtypes(include=[np.number])
        minmax_df = (num_df - num_df.min()) / (num_df.max() - num_df.min())
        minmax_df.to_csv("minmax_normalized.csv", index=False)
        print("Min-Max normalized data preview:")
        print(minmax_df.head())
        return minmax_df

    def z_score_normalization(self):
        """
        Apply Z-score normalization.

        Formula:
        X' = (X - mean) / std

        Steps:
        1. Compute mean and standard deviation.
        2. Apply normalization formula.
        3. Print preview of normalized dataset.
        4. Return result.
        """
        print("\nApplying Z-score normalization...")

        # TODO:
        # 1 Compute Z-score normalization
        # 2 Print first 5 rows
        # 3 Return normalized dataset
        num_df = self.df.select_dtypes(include=[np.number])
        zscore_df = (num_df - num_df.mean()) / num_df.std()
        zscore_df.to_csv("zscore_normalized.csv", index=False)
        print("Z-score normalized data preview:")
        print(zscore_df.head())
        return zscore_df

    def decimal_scaling(self):
        """
        Apply Decimal Scaling normalization.

        Steps:
        1. Determine largest absolute value in each column.
        2. Find smallest power of 10 >= max(|X|).
        3. Scale values.
        4. Print first rows of scaled data.
        """
        print("\nApplying Decimal Scaling normalization...")

        # TODO:
        # 1 Compute decimal scaling
        # 2 Print preview
        # 3 Return scaled dataset
        num_df = self.df.select_dtypes(include=[np.number])
        decimal_df = num_df.copy()
        for col in decimal_df.columns:
            max_abs = decimal_df[col].abs().max()
            if max_abs > 0:
                j = int(np.ceil(np.log10(max_abs)))
                if max_abs == 10 ** j:
                    j += 1
                decimal_df[col] = decimal_df[col] / (10 ** j)
        decimal_df.to_csv("decimal_scaled.csv", index=False)
        print("Decimal scaled data preview:")
        print(decimal_df.head())
        return decimal_df


def run_normalization(filepath):
    """
    Run normalization pipeline.
    """
    model = WineNormalization(filepath)

    minmax = model.min_max_normalization()
    zscore = model.z_score_normalization()
    decimal = model.decimal_scaling()

    print("\nAll normalization techniques applied successfully.")

    return minmax, zscore, decimal


if __name__ == "__main__":

    run_normalization("winequality-red.csv")

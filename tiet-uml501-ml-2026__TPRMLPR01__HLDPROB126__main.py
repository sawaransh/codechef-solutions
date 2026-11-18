import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


class IrisPreprocessing:

    def __init__(self):
        """
        Initialize preprocessing.

        You must:
        - Load Iris dataset
        - Convert dataset to a Pandas DataFrame
        - Store dataset in self.df
        """
        # TODO: Load dataset and create DataFrame
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)

    def introduce_missing_values(self):
        """
        Introduce missing values to simulate real-world data.

        Return modified DataFrame.
        """
        # TODO: Insert NaN values into dataset
        self.df.loc[1, "sepal length (cm)"] = np.nan
        return self.df

    def clean_data(self):
        """
        Identify and fill missing values.

        Return cleaned DataFrame.
        """
        # TODO: Fill missing values using appropriate method
        self.df = self.df.fillna(self.df.median())
        return self.df

    def compute_correlation(self):
        """
        Compute correlation matrix of numeric features.

        Return correlation DataFrame.
        """
        # TODO: Compute correlation matrix
        return self.df.corr()

    def save_heatmap(self, corr_matrix):
        """
        Generate heatmap visualization and save it as an image file.

        You must:
        - Create heatmap
        - Label axes clearly
        - Save image file
        - Print confirmation message
        """
        # TODO: Create heatmap and save image
        filename = "correlation_heatmap.png"
        plt.figure(figsize=(8, 6))
        plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title("Iris Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Heatmap saved as {filename}")


def run_preprocessing():

    processor = IrisPreprocessing()

    df_missing = processor.introduce_missing_values()
    cleaned = processor.clean_data()
    corr_matrix = processor.compute_correlation()

    processor.save_heatmap(corr_matrix)

    return {
        "data_with_missing": df_missing,
        "cleaned_data": cleaned,
        "correlation": corr_matrix
    }


if __name__ == "__main__":

    results = run_preprocessing()

    print("\n--- Dataset with Missing Values ---")
    print(results["data_with_missing"].head())

    print("\n--- Cleaned Dataset ---")
    print(results["cleaned_data"].head())

    print("\n--- Correlation Matrix ---")
    print(results["correlation"])

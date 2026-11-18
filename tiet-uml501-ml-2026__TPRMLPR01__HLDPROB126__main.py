import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


class IrisPreprocessing:

    def __init__(self):
        """
        Initialize preprocessing.
        """
        # Load Iris dataset
        iris = load_iris()
        # Convert dataset to a Pandas DataFrame
        # Use only numeric feature columns
        self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    def introduce_missing_values(self):
        """
        Introduce missing values to simulate real-world data.
        """
        # Inject NaN values into dataset
        self.df.loc[0, "sepal length (cm)"] = np.nan
        return self.df.copy()

    def clean_dataset(self):
        """
        Identify and fill missing values.
        """
        # Fill missing values with column mean
        if self.df.isna().sum().sum() > 0:
            self.df = self.df.fillna(self.df.mean(numeric_only=True))
        return self.df.copy()

    def compute_correlation(self):
        """
        Compute correlation matrix of numeric features.
        """
        # Compute correlation matrix
        return self.df.corr(numeric_only=True)

    def save_heatmap(self, corr_matrix):
        """
        Generate heatmap visualization and save it as an image file.

        Parameters:
        - corr_matrix:
          Pandas DataFrame
        - Output image file:
          'correlation_heatmap.png'
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)

        columns = list(corr_matrix.columns)
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticklabels(columns)

        for i in range(len(columns)):
            for j in range(len(columns)):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black"
                )

        ax.set_title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close(fig)
        print("Heatmap saved as correlation_heatmap.png")


def run_preprocessing():

    processor = IrisPreprocessing()

    df_missing = processor.introduce_missing_values()
    cleaned_df = processor.clean_dataset()
    corr_matrix = processor.compute_correlation()

    processor.save_heatmap(corr_matrix)

    return {
        "data_with_missing": df_missing,
        "cleaned_data": cleaned_df,
        "correlation": corr_matrix
    }


if __name__ == "__main__":

    results = run_preprocessing()

    print("\n=== Dataset with Missing Values ===")
    print(results["data_with_missing"].head())

    print("\n=== Cleaned Dataset ===")
    print(results["cleaned_data"].head())

    print("\n=== Correlation Matrix ===")
    print(results["correlation"])
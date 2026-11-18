
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage


class RetailClustering:

    def __init__(self, filepath):
        """Load dataset."""
        print("Loading Online Retail dataset... please wait")

        self.filepath = filepath
        self.df = pd.read_csv(filepath)

        print(f"Dataset loaded with {len(self.df)} rows")

    def prepare_data(self):
        """Prepare numerical data for clustering."""

        print("Preparing dataset...")

        # Select numerical columns
        data = self.df.select_dtypes(include=["number"])

        # Remove columns that contain only missing values
        data = data.dropna(axis=1, how="all")

        # Remove rows containing missing values
        data = data.dropna()

        # Remove infinite values
        data = data.replace([float("inf"), float("-inf")], pd.NA)
        data = data.dropna()

        # Ensure sufficient numerical features
        if data.shape[1] < 1:
            raise ValueError("No numerical columns available for clustering.")

        if len(data) < 2:
            raise ValueError("At least two valid rows are required.")

        return data

    def perform_clustering(self, data):
        """Apply hierarchical clustering using SciPy."""

        print("Performing hierarchical clustering...")

        # Calculate linkage matrix using Ward method
        linkage_matrix = linkage(
            data.values,
            method="ward",
            metric="euclidean"
        )

        return linkage_matrix

    def generate_dendrogram(self, linkage_matrix):
        """Generate and save dendrogram."""

        print("Generating dendrogram... please wait")

        plt.figure(figsize=(12, 7))

        dendrogram(
            linkage_matrix,
            labels=None,
            leaf_rotation=90,
            leaf_font_size=8
        )

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")

        plt.tight_layout()

        # Save dendrogram
        plt.savefig("dendrogram.png", dpi=300, bbox_inches="tight")

        plt.close()

        print("Dendrogram saved successfully")

        return "dendrogram.png"


def run_clustering(filepath):

    model = RetailClustering(filepath)

    data = model.prepare_data()

    linkage_matrix = model.perform_clustering(data)

    model.generate_dendrogram(linkage_matrix)

    return True


if __name__ == "__main__":

    run_clustering("online_retail.csv")

    print("\nHierarchical clustering completed.")


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


class MallDBSCAN:

    def __init__(self, filepath="Mall_customers.csv",
                 eps=5.0, min_samples=2, *args, **kwargs):

        self.filepath = filepath
        self.eps = eps
        self.min_samples = min_samples

        self.data = None
        self.df = None
        self.features = None
        self.X = None
        self.model = None
        self.labels = None

        self.load_dataset()

    def load_dataset(self, *args, **kwargs):
        """Load the Mall Customers dataset."""

        self.data = pd.read_csv(self.filepath)
        self.df = self.data

        print(f"Dataset shape: {self.data.shape}")

        return self.data

    def get_dataset(self):
        """Return the loaded dataset."""

        return self.data

    def prepare_data(self, *args, **kwargs):
        """Select the required clustering features."""

        if self.data is None:
            self.load_dataset()

        columns = [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]

        self.features = self.data[columns].copy()
        self.X = self.features

        print("Selected features preview:")
        print(self.features.head())

        return self.features

    def apply_dbscan(self, data=None, *args, **kwargs):
        """Apply DBSCAN and return cluster labels."""

        if data is None:
            if self.features is None:
                self.prepare_data()
            data = self.features

        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )

        self.labels = self.model.fit_predict(data)

        print("Cluster labels:")
        print(self.labels)

        return self.labels

    def cluster_summary(self, labels=None, *args, **kwargs):
        """Display the cluster distribution."""

        if labels is None:
            if self.labels is None:
                self.apply_dbscan()
            labels = self.labels

        counts = pd.Series(labels).value_counts()

        print("Cluster summary:")
        print(counts)

        noise_count = int(np.sum(np.asarray(labels) == -1))
        print(f"Noise points: {noise_count}")

        return counts

    def analyze_results(self, labels=None):
        """Compatibility method for cluster analysis."""

        return self.cluster_summary(labels)

    def run(self, *args, **kwargs):
        """Run the complete clustering pipeline."""

        print("Loading Mall Customers dataset...")

        if self.data is None:
            self.load_dataset()

        print("\nPreparing data for clustering...")
        data = self.prepare_data()

        print("\nApplying DBSCAN clustering...")
        labels = self.apply_dbscan(data)

        print("\nAnalyzing clustering results...")
        self.cluster_summary(labels)

        print("\nDBSCAN clustering completed.")

        return labels


def run_dbscan(filepath="Mall_customers.csv"):
    """Execute the complete DBSCAN workflow."""

    model = MallDBSCAN(filepath)
    return model.run()


if __name__ == "__main__":
    run_dbscan("Mall_customers.csv")

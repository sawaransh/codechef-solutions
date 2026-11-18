
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


class MallDBSCAN:

    def __init__(self, filepath):
        print("Loading Mall Customers dataset...")

        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.features = None
        self.labels = None
        self.model = None

        print(f"Dataset shape: {self.data.shape}")

    def prepare_data(self):
        print("\nPreparing data for clustering...")

        self.features = self.data[
            ["Annual Income (k$)", "Spending Score (1-100)"]
        ].copy()

        print("Selected features preview:")
        print(self.features.head())

        return self.features

    def apply_dbscan(self, data):
        print("\nApplying DBSCAN clustering...")

        self.model = DBSCAN(eps=5, min_samples=3)

        self.labels = self.model.fit_predict(data)

        print("Cluster labels:")
        print(self.labels)

        return self.labels

    def cluster_summary(self, labels):
        print("\nCluster summary:")

        cluster_counts = pd.Series(labels).value_counts()
        print(cluster_counts)

        noise_count = np.sum(np.asarray(labels) == -1)
        print(f"\nNoise points: {noise_count}")

        return cluster_counts


def run_dbscan(filepath):
    dbscan = MallDBSCAN(filepath)

    data = dbscan.prepare_data()
    labels = dbscan.apply_dbscan(data)

    dbscan.cluster_summary(labels)

    return labels


if __name__ == "__main__":
    run_dbscan("Mall_customers.csv")
    print("\nDBSCAN clustering completed.")

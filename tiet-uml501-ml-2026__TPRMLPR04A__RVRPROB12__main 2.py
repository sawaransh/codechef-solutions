import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering


class HierarchicalClusteringProject:

    def __init__(self):
        """
        Load dataset.
        """
        print("Loading dataset...")

        csv_path = "dataset.csv"
        if not os.path.exists(csv_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "dataset.csv")

        self.df = pd.read_csv(csv_path)

    def prepare_data(self):
        """
        Extract feature columns.
        """
        # Extract all feature columns
        features = self.df.select_dtypes(include=["number"]).values
        if features.size == 0:
            features = self.df.values
        return features

    def generate_dendrogram(self, features):
        """
        Generate dendrogram.
        """
        print("Generating Dendrogram...")

        # Compute Linkage
        linked = linkage(features, method="ward")

        # Generate dendrogram
        plt.figure(figsize=(10, 6))
        dendrogram(linked)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")

        # Save as dendrogram.png
        plt.savefig("dendrogram.png")
        plt.close()

    def apply_clustering(self, features):
        """
        Apply Hierarchical Clustering.
        """
        print("Applying Hierarchical Clustering...")

        # Create AgglomerativeClustering model
        model = AgglomerativeClustering(n_clusters=2)

        # Fit model & Generate cluster labels
        labels = model.fit_predict(features)
        return labels

    def plot_clusters(self, features, labels):
        """
        Generate cluster visualization.
        """
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            features[:, 0],
            features[:, 1] if features.shape[1] > 1 else [0] * len(features),
            c=labels,
            cmap="viridis"
        )
        plt.title("Hierarchical Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2" if features.shape[1] > 1 else "Value")

        # Save as hierarchical_clusters.png
        plt.savefig("hierarchical_clusters.png")
        plt.close()

    def display_results(self, labels):
        """
        Display cluster labels.
        """
        print("\nCluster Labels:")
        print(labels)


def run_hierarchical_clustering():
    model = HierarchicalClusteringProject()
    features = model.prepare_data()
    model.generate_dendrogram(features)
    labels = model.apply_clustering(features)
    model.plot_clusters(features, labels)
    model.display_results(labels)


if __name__ == "__main__":
    run_hierarchical_clustering()
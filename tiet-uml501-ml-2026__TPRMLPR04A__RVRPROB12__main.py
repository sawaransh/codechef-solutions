
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering


class HierarchicalClusteringProject:

    def __init__(self):
        print("Loading dataset...")

        self.data = pd.read_csv("dataset.csv")

    def prepare_data(self):
        """
        Extract feature columns and remove missing values.
        """

        features = self.data.select_dtypes(include=["number"])

        features = features.dropna()

        return features

    def generate_dendrogram(self, features):
        """
        Generate and save dendrogram.
        """

        print("Generating dendrogram...")

        linkage_matrix = linkage(features, method="ward")

        plt.figure(figsize=(10, 6))

        dendrogram(linkage_matrix)

        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")

        plt.tight_layout()
        plt.savefig("dendrogram.png")
        plt.close()

        return linkage_matrix

    def apply_clustering(self, features):
        """
        Apply Agglomerative Clustering.
        """

        print("Applying Hierarchical Clustering...")

        model = AgglomerativeClustering(
            n_clusters=2,
            linkage="ward"
        )

        labels = model.fit_predict(features)

        return labels

    def plot_clusters(self, features, labels):
        """
        Visualize clusters and save the scatter plot.
        """

        plt.figure(figsize=(8, 6))

        if features.shape[1] >= 2:
            x = features.iloc[:, 0]
            y = features.iloc[:, 1]
        else:
            x = range(len(features))
            y = features.iloc[:, 0]

        plt.scatter(x, y, c=labels, cmap="viridis")

        plt.title("Hierarchical Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        plt.tight_layout()
        plt.savefig("hierarchical_clusters.png")
        plt.close()

    def display_results(self, labels):
        """
        Display generated cluster labels.
        """

        print("Cluster Labels:")
        print(labels)

    def run_hierarchical_clustering(self):

        features = self.prepare_data()

        self.generate_dendrogram(features)

        labels = self.apply_clustering(features)

        self.plot_clusters(features, labels)

        self.display_results(labels)


if __name__ == "__main__":

    model = HierarchicalClusteringProject()

    model.run_hierarchical_clustering()

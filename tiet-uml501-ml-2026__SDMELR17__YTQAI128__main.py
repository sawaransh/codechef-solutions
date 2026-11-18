from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Generate a toy dataset
# TODO: Create a dataset with 150 samples and 4 clusters
X, _ = make_blobs(n_samples=150, centers=4, random_state=42)

# Step 2: Train KMeans
# TODO: Initialize KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# TODO: Fit the model and get cluster labels
labels = kmeans.fit_predict(X)

# Step 3: Visualize the clusters
# TODO: Plot the points with cluster colors
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

# TODO: Plot the cluster centers (centroids)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')

plt.legend()
plt.savefig("kmeans_practice.png")
plt.close()
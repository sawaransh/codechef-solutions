# TODO: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Step 1: Load the dataset
# TODO: Load the Iris dataset
iris = load_iris()
# TODO: Select features
X = iris.data
# TODO: Select labels
y = iris.target

# Step 2: Apply PCA
# TODO: Create a PCA model with 2 components
pca = PCA(n_components=2)
# TODO: Transform the dataset into 2D
X_pca = pca.fit_transform(X)

# Step 3: Visualize the PCA result
# TODO: Create scatter plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')

# TODO: Label the axes
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Practice Problem - Iris Dataset")

# Step 4: Save and show the plot
plt.savefig("pca_iris.png")
plt.show()
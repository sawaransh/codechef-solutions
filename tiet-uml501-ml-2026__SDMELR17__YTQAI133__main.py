# Import required libraries
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# TODO: Load the Iris dataset
iris = load_iris()

# TODO: Select features and labels
X = iris.data  # TODO: Select features
y = iris.target  # TODO: Select labels

# Step 2: Apply PCA
# TODO: Create a PCA model with 2 components
pca = PCA(n_components=2)

# TODO: Transform the dataset into 2D
X_pca = pca.fit_transform(X)

# Step 3: Visualize the PCA result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Practice Problem")

# Save the figure
plt.savefig("pca_practice.png")

# Show the plot
plt.show()
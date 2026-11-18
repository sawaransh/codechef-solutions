from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def load_mnist_data(test_size=0.3, random_state=42):
    """
    Load the MNIST digits dataset and perform train-test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=test_size, random_state=random_state, stratify=digits.target
    )
    return X_train, X_test, y_train, y_test

def apply_pca(X_train, X_test, variance_ratio=0.95, random_state=42):
    """
    Apply PCA to reduce dimensionality while retaining specified variance.

    Returns:
        X_train_pca, X_test_pca, pca_object
    """
    pca = PCA(n_components=variance_ratio, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def train_logistic_regression(X_train, y_train, X_test=None, y_test=None, max_iter=2000):
    """
    Train a Logistic Regression classifier.

    Returns:
        clf: trained classifier
        acc: accuracy on test set if provided, else None
    """
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) if (X_test is not None and y_test is not None) else None
    return clf, acc

def pca_explained_variance(X):
    """
    Compute cumulative explained variance for all PCA components.

    Returns:
        cumulative_variance: numpy array of cumulative variance ratios
    """
    pca = PCA()
    pca.fit(X)
    return np.cumsum(pca.explained_variance_ratio_)

def main():
    # 1. Load MNIST digits dataset
    X_train, X_test, y_train, y_test = load_mnist_data()
    print("Original dataset shape:", X_train.shape)

    # 2. Apply PCA to retain 95% variance
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test)
    print("PCA reduced shape:", X_train_pca.shape)

    # 3. Train Logistic Regression on original data
    clf_orig, acc_orig = train_logistic_regression(X_train, y_train, X_test, y_test)
    print("Accuracy on original data:", acc_orig)

    # 4. Train Logistic Regression on PCA-reduced data
    clf_pca, acc_pca = train_logistic_regression(X_train_pca, y_train, X_test_pca, y_test)
    print("Accuracy on PCA-reduced data:", acc_pca)

    # 5. Get cumulative explained variance
    cum_var = pca_explained_variance(X_train)
    print(cum_var)

# Run the script
if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_iris_data():
    """
    Load the Iris dataset into a DataFrame, encode the target, and handle missing values.
    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Encoded target
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target_names[iris.target]

    df.fillna(df.mean(numeric_only=True), inplace=True)

    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])

    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Fraction for test set
        random_state (int): Random seed
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    Args:
        X_train, X_test: Feature matrices for train and test
    Returns:
        X_train_scaled, X_test_scaled: Scaled feature matrices
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_knn(X_train, y_train, X_test, k=5):
    """
    Train K-Nearest Neighbors classifier.
    Args:
        X_train, y_train: Training data
        X_test: Test features
        k (int): Number of neighbors
    Returns:
        y_pred: Predictions on X_test
        accuracy: Accuracy score on training set
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    train_pred = knn.predict(X_train)
    accuracy = accuracy_score(y_train, train_pred)

    return y_pred, accuracy


def evaluate_knn_range(X_train, y_train, X_test, y_test, k_values=[1, 3, 5, 7, 9]):
    """
    Evaluate KNN classifier across a list of k values.
    Args:
        X_train, y_train, X_test, y_test: Train and test data
        k_values (list): List of k values
    Returns:
        results (dict): Accuracy for each k
    """
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        results[k] = accuracy_score(y_test, y_pred)
    return results


if __name__ == "__main__":
    # Load and split data
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train KNN (k=5)
    y_pred, train_acc = train_knn(X_train_scaled, y_train, X_test_scaled, k=5)
    print("KNN Accuracy:", accuracy_score(y_test, y_pred))

    # Evaluate multiple k values
    k_results = evaluate_knn_range(X_train_scaled, y_train, X_test_scaled, y_test)
    for k, acc in k_results.items():
        print(f"k={k} Accuracy:", acc)
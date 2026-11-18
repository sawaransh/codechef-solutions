import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class TitanicModel:

    def __init__(self, filepath):

        print("Loading Titanic dataset...")

        if not os.path.exists(filepath):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(base_dir, filepath)
            if os.path.exists(alt_path):
                filepath = alt_path

        self.df = pd.read_csv(filepath)

    def preprocess(self):
        """
        Clean and prepare dataset
        """
        df = self.df.copy()

        # Drop unnecessary columns
        cols_to_drop = [
            col for col in ["PassengerId", "Name", "Ticket", "Cabin"]
            if col in df.columns
        ]
        df = df.drop(columns=cols_to_drop)

        # Handle missing values
        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(df["Age"].median())
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        # Encode categorical columns (Sex, Embarked)
        for col in ["Sex", "Embarked"]:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Split into X and y
        X = df.drop(columns=["Survived"])
        y = df["Survived"]

        return X, y

    def split_data(self, X, y):

        return train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_decision_tree(self, X_train, y_train):

        print("Training Decision Tree...")

        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_model.fit(X_train, y_train)
        return dt_model

    def train_random_forest(self, X_train, y_train):

        print("Training Random Forest...")

        rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
        rf_model.fit(X_train, y_train)
        return rf_model

    def evaluate(self, model, X_test, y_test):

        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred) * 100

    def visualize_tree(self, model, feature_names):

        plt.figure(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=["Not Survived", "Survived"],
            filled=True,
            rounded=True,
            max_depth=3,
        )
        plt.savefig("decision_tree.png", bbox_inches="tight")
        plt.close()
        print("Decision tree visualization saved as decision_tree.png")


def run_model(filepath):

    model = TitanicModel(filepath)

    X, y = model.preprocess()

    X_train, X_test, y_train, y_test = model.split_data(X, y)

    dt = model.train_decision_tree(X_train, y_train)
    rf = model.train_random_forest(X_train, y_train)

    dt_acc = model.evaluate(dt, X_test, y_test)
    rf_acc = model.evaluate(rf, X_test, y_test)

    model.visualize_tree(dt, X.columns)

    return {
        "DecisionTree": dt_acc,
        "RandomForest": rf_acc
    }


if __name__ == "__main__":

    results = run_model("Titanic.csv")

    print("\nModel Comparison:")

    for k, v in results.items():
        print(f"{k}: {v:.2f}%")

    best = max(results, key=results.get)

    print(f"\nBest Model: {best}")
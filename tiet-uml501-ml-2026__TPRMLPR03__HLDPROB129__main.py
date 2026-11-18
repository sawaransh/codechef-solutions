import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


class CreditClassifier:

    def __init__(self, filepath):
        """
        Initialize classifier.

        Tasks:
        - Load CSV dataset
        - Store dataframe
        """
        print("Loading dataset... please wait")

        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded with {len(self.df)} records")

    def encode_features(self):
        """
        Encode categorical columns.

        Returns:
        Return encoded DataFrame.
        """
        print("Encoding categorical features...")

        df_encoded = self.df.copy()
        self.encoders = {}

        categorical_cols = df_encoded.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.encoders[col] = le

        self.df = df_encoded
        return self.df

    def train_model(self, df):
        """
        Train classification model.

        Steps:
        - Separate features and target
        - Split dataset
        - Scale numeric features
        - Train Logistic Regression

        Returns:
        model, X_test, y_test
        """
        print("Training classifier...")

        X = df.drop(columns=["status"])
        y = df["status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)

        return self.model, X_test, y_test

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.

        Returns:
        confusion matrix and accuracy percentage
        """
        print("Evaluating model...")

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred) * 100

        return cm, acc


def run_pipeline(filepath):

    classifier = CreditClassifier(filepath)

    df = classifier.encode_features()

    model, X_test, y_test = classifier.train_model(df)

    cm, acc = classifier.evaluate_model(model, X_test, y_test)

    return {
        "confusion_matrix": cm,
        "accuracy": acc
    }


if __name__ == "__main__":

    filepath = "credit_data.csv"

    results = run_pipeline(filepath)

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print(f"\nAccuracy: {results['accuracy']:.2f}%")
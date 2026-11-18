import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class ID3Classifier:

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
        self.encoders = {}

    def prepare_data(self):
        """
        Prepare dataset for training.
        """
        df_encoded = self.df.copy()

        categorical_cols = df_encoded.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.encoders[col] = le

        target_col = "PlayTennis" if "PlayTennis" in df_encoded.columns else df_encoded.columns[-1]
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train Decision Tree model.
        """
        print("\nTraining Decision Tree...")

        clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
        clf.fit(X_train, y_train)
        return clf

    def predict(self, model, X_test):
        """
        Generate predictions.
        """
        return model.predict(X_test)

    def predict_new_sample(self, model):
        """
        Predict a new sample.
        """
        print("\nPrediction for New Sample:")

        sample = pd.DataFrame({
            "Outlook": ["Sunny"],
            "Temperature": ["Mild"],
            "Humidity": ["Normal"],
            "Wind": ["Strong"]
        })

        encoded_sample = sample.copy()
        for col in encoded_sample.columns:
            if col in self.encoders:
                encoded_sample[col] = self.encoders[col].transform(
                    encoded_sample[col].astype(str)
                )

        pred = model.predict(encoded_sample)

        target_col = "PlayTennis" if "PlayTennis" in self.encoders else list(self.encoders.keys())[-1]
        predicted_label = self.encoders[target_col].inverse_transform(pred)[0]

        print(predicted_label)
        return predicted_label

    def evaluate(self, y_test, predictions):
        """
        Evaluate model performance.
        """
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        print("\nModel Accuracy:")
        print(acc)

        print("\nClassification Report:")
        print(report)

        return acc, report


def run_id3():

    model = ID3Classifier()

    X_train, X_test, y_train, y_test = model.prepare_data()

    classifier = model.train_model(X_train, y_train)

    predictions = model.predict(classifier, X_test)

    model.evaluate(y_test, predictions)

    model.predict_new_sample(classifier)


if __name__ == "__main__":
    run_id3()
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

class FlowerSpeciesClassification:
    """
    Build a Bayesian Classification model
    for flower species prediction.
    """
    def __init__(self):
        """
        Load the dataset from
        flower_species_classification.csv.
        """
        self.df = pd.read_csv('flower_species_classification.csv')
        self.X = None
        self.y = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def clean_dataset(self):
        """
        Remove:
        - Duplicate rows
        - Rows containing missing values
        Store the cleaned dataset.
        """
        self.df = self.df.drop_duplicates().dropna().copy()
        return self.df

    def perform_eda(self):
        """
        Generate a summary containing:
        - Total records
        - Number of features
        - Missing values
        - Duplicate rows
        """
        summary = {
            "total_records": len(self.df),
            "number_of_features": len(self.df.columns) - 1 if 'Species' in self.df.columns else len(self.df.columns),
            "missing_values": int(self.df.isnull().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum())
        }
        return summary

    def transform_dataset(self):
        """
        Separate:
        - Features (X)
        - Target (y)
        Store them for later use
        """
        self.X = self.df.drop("Species", axis=1)
        self.y = self.df["Species"]
        return self.X, self.y

    def train_model(self):
        """
        Split the cleaned dataset into training and testing sets
        Train a Gaussian Naive Bayes classifier
        """
        if self.X is None or self.y is None:
            self.transform_dataset()

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(self.X_test)
        return self.model

    def evaluate_model(self):
        """
        Calculate the following using the test dataset:
        - Accuracy
        - Classification Report
        """
        if self.model is None:
            self.train_model()

        acc = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        return {
            "accuracy": acc,
            "classification_report": report
        }

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Predict the flower species
        """
        if self.model is None:
            self.train_model()

        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]], 
            columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
        )
        prediction = self.model.predict(input_data)
        return prediction[0]

    def get_dataset(self):
        """
        Return the dataset
        """
        return self.df

# Do not modify the __main__ block below
if __name__ == "__main__":
    classifier = FlowerSpeciesClassification()
    eda_summary = classifier.perform_eda()
    print("EDA Summary:")
    for k, v in eda_summary.items():
        print(f"{k}: {v}")
        
    classifier.clean_dataset()
    classifier.transform_dataset()
    classifier.train_model()
    eval_metrics = classifier.evaluate_model()
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {eval_metrics['accuracy']}")
    print(eval_metrics['classification_report'])
    
    sample_pred = classifier.predict_species(5.1, 3.5, 1.4, 0.2)
    print("\nPrediction:")
    print(sample_pred)
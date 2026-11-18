import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class BuildingDataPreparation:

    def __init__(self, filepath):
        """
        Initialize data preparation pipeline.

        Tasks:
        - Load CSV dataset into a DataFrame
        """
        print("Loading dataset... please wait")

        # TODO: Load CSV dataset
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded with {len(self.df)} records.")

    def clean_columns(self):
        """
        Clean numeric columns that contain extra characters.

        Tasks:
        - Extract numeric value from Height_m column
        - Extract numeric value from Year column
        - Convert both columns to numeric datatype
        """
        print("Cleaning numeric columns...")

        # TODO: Clean Height_m and Year columns
        self.df["Height_m"] = pd.to_numeric(
            self.df["Height_m"].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
            errors="coerce",
        )
        self.df["Year"] = pd.to_numeric(
            self.df["Year"].astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
            errors="coerce",
        )

    def handle_missing_values(self):
        """
        Identify and fill missing numeric values.
        """
        print("Handling missing values...")

        # TODO: Fill missing numeric values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        for col in ["City", "Country"]:
            if col in self.df.columns and self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def encode_categorical_features(self):
        """
        Encode categorical columns:
        - City
        - Country
        """
        print("Encoding categorical features...")

        # TODO: Apply label encoding
        for col in ["City", "Country"]:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])

    def scale_features(self):
        """
        Normalize numeric features using MinMax scaling.

        Columns:
        - Height_m
        - Floors
        - Year
        """
        print("Scaling numeric features... please wait")

        # TODO: Apply scaling
        cols_to_scale = ["Height_m", "Floors", "Year"]
        scaler = MinMaxScaler()
        self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])


def run_pipeline(filepath):

    processor = BuildingDataPreparation(filepath)

    processor.clean_columns()

    processor.handle_missing_values()

    processor.encode_categorical_features()

    processor.scale_features()

    return processor.df


if __name__ == "__main__":

    filepath = "tallest_buildings.csv"

    df = run_pipeline(filepath)

    print("\n=== Prepared Dataset ===")
    print(df.head())
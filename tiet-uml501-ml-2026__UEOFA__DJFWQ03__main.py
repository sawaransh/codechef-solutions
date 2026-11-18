import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


class CarPriceRegression:
    """
    Build car price regression models
    before and after PCA.
    """

    def __init__(self):
        """
        Load the automobile dataset
        while treating ? as missing data.
        """
        self.dataset = pd.read_csv("automobile_price_prediction.csv", na_values="?")
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.model = None
        self.pca = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.pca_model = None

    def clean_dataset(self):
        """
        Remove missing target values and
        impute the remaining missing data.
        """
        numeric_cols = [
            "symboling",
            "normalized_losses",
            "wheel_base",
            "length",
            "width",
            "height",
            "curb_weight",
            "engine_size",
            "bore",
            "stroke",
            "compression_ratio",
            "horsepower",
            "peak_rpm",
            "city_mpg",
            "highway_mpg",
            "price",
        ]

        for col in numeric_cols:
            if col in self.dataset.columns:
                self.dataset[col] = pd.to_numeric(self.dataset[col], errors="coerce")

        # Remove rows where price is missing
        self.dataset = self.dataset.dropna(subset=["price"]).reset_index(drop=True)

        # Fill remaining missing numerical and categorical values
        for col in self.dataset.columns:
            if pd.api.types.is_numeric_dtype(self.dataset[col]):
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].median())
            else:
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0])

    def encode_number_words(self):
        """
        Convert textual number values in
        door and cylinder columns to numbers.
        """
        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
        }

        for col in ["num_doors", "num_cylinders"]:
            if col in self.dataset.columns:
                mapped = self.dataset[col].astype(str).str.lower().str.strip().map(word_to_num)
                numeric = pd.to_numeric(self.dataset[col], errors="coerce")
                self.dataset[col] = mapped.fillna(numeric).astype(int)

    def encode_categorical_features(self):
        """
        Convert the required categorical
        columns into numerical features.
        """
        # Binary encoding for fuel_system (1 if contains 'pfi', else 0)
        self.dataset["fuel_system"] = (
            self.dataset["fuel_system"].astype(str).str.lower().str.contains("pfi").astype(int)
        )

        # Binary encoding for engine_type (1 if contains 'ohc', else 0)
        self.dataset["engine_type"] = (
            self.dataset["engine_type"].astype(str).str.lower().str.contains("ohc").astype(int)
        )

        # Label encoding for make, aspiration, engine_location, fuel_type
        label_cols = ["make", "aspiration", "engine_location", "fuel_type"]
        for col in label_cols:
            le = LabelEncoder()
            self.dataset[col] = le.fit_transform(self.dataset[col].astype(str))

        # Dummy encoding for body_style and drive_wheels
        self.dataset = pd.get_dummies(
            self.dataset,
            columns=["body_style", "drive_wheels"],
            drop_first=True,
            dtype=int,
        )

    def prepare_data(self):
        """
        Separate the numerical features
        from the price target.
        """
        self.X = self.dataset.drop(columns=["price"])
        self.y = self.dataset["price"]

    def split_data(self):
        """
        Divide the prepared dataset into
        70% training and 30% testing data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def scale_features(self):
        """
        Scale features using statistics
        learned only from training data.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_train_scaled = self.X_train
        self.X_test_scaled = self.X_test

    def train_regression_model(self):
        """
        Train Multiple Linear Regression
        using the processed training data.
        """
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the baseline regression
        model on the test data.
        """
        y_pred = self.model.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def apply_pca(self):
        """
        Reduce the scaled feature space
        using PCA fitted on training data.
        """
        self.pca = PCA(n_components=8, random_state=42)
        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.transform(self.X_test)

    def train_pca_model(self):
        """
        Train Multiple Linear Regression
        using PCA-reduced training data.
        """
        self.pca_model = LinearRegression()
        self.pca_model.fit(self.X_train_pca, self.y_train)

    def evaluate_pca_model(self):
        """
        Evaluate the PCA regression model
        on the reduced test data.
        """
        y_pred_pca = self.pca_model.predict(self.X_test_pca)
        return r2_score(self.y_test, y_pred_pca)

    def compare_models(self):
        """
        Determine whether PCA improves
        test-set regression performance.
        """
        base_r2 = self.evaluate_model()
        pca_r2 = self.evaluate_pca_model()
        return bool(pca_r2 > base_r2)

    def get_dataset(self):
        """
        Return the current processed dataset.
        """
        return self.dataset


if __name__ == "__main__":

    model = CarPriceRegression()

    model.clean_dataset()
    model.encode_number_words()
    model.encode_categorical_features()
    model.prepare_data()
    model.split_data()
    model.scale_features()
    model.train_regression_model()
    model.apply_pca()
    model.train_pca_model()

    print("Regression R2 Score:")
    print(model.evaluate_model())

    print("\nPCA Regression R2 Score:")
    print(model.evaluate_pca_model())

    print("\nPCA Improved Performance:")
    print(model.compare_models())
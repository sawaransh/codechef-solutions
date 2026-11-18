
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


class BiasVarianceAnalyzer:

    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.linear_model = None
        self.degree2_model = None
        self.degree8_model = None

    # Load dataset
    def load_dataset(self):
        self.dataset = pd.read_csv("house_price_bias_variance.csv")

    # Prepare data
    def prepare_data(self):
        self.X = self.dataset[["Area"]]
        self.y = self.dataset["Price"]

    # Train all three models
    def train_models(self):

        self.linear_model = LinearRegression()

        self.degree2_model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression())
        ])

        self.degree8_model = Pipeline([
            ("poly", PolynomialFeatures(degree=8, include_bias=False)),
            ("linear", LinearRegression())
        ])

        self.linear_model.fit(self.X, self.y)
        self.degree2_model.fit(self.X, self.y)
        self.degree8_model.fit(self.X, self.y)

    # Linear predictions
    def predict_linear(self):
        return self.linear_model.predict(self.X)

    # Degree 2 predictions
    def predict_degree2(self):
        return self.degree2_model.predict(self.X)

    # Degree 8 predictions
    def predict_degree8(self):
        return self.degree8_model.predict(self.X)

    # Calculate Linear Regression MSE
    def calculate_linear_mse(self):
        predictions = self.predict_linear()
        return mean_squared_error(self.y, predictions)

    # Calculate Degree 2 MSE
    def calculate_degree2_mse(self):
        predictions = self.predict_degree2()
        return mean_squared_error(self.y, predictions)

    # Calculate Degree 8 MSE
    def calculate_degree8_mse(self):
        predictions = self.predict_degree8()
        return mean_squared_error(self.y, predictions)


if __name__ == "__main__":

    analyzer = BiasVarianceAnalyzer()

    analyzer.load_dataset()
    analyzer.prepare_data()
    analyzer.train_models()

    print("Linear Regression MSE:")
    print(analyzer.calculate_linear_mse())

    print("\nPolynomial Regression (Degree 2) MSE:")
    print(analyzer.calculate_degree2_mse())

    print("\nPolynomial Regression (Degree 8) MSE:")
    print(analyzer.calculate_degree8_mse())

    print("\nLinear Regression Predictions:")
    print(analyzer.predict_linear())

    print("\nPolynomial Regression (Degree 2) Predictions:")
    print(analyzer.predict_degree2())

    print("\nPolynomial Regression (Degree 8) Predictions:")
    print(analyzer.predict_degree8())

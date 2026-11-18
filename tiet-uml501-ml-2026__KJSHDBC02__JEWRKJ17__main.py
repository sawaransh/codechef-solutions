import pandas as pd
from sklearn.linear_model import LinearRegression


class HousePricePredictor:
    """
    Train a Multiple Linear Regression model
    and predict house prices.
    """

    def __init__(self):

        self.dataset = pd.read_csv("house_price_prediction.csv")

        self.X = self.dataset[["Area", "Bedrooms", "HouseAge"]]
        self.y = self.dataset["Price"]
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def load_dataset(self):
        self.dataset=pd.read_csv("house_price_prediction.csv")

    def prepare_data(self):
        self.X = self.dataset[["Area", "Bedrooms", "HouseAge"]]
        self.y = self.dataset["Price"]

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def get_model_parameters(self):
        intercept = self.model.intercept_
        coefficients = self.model.coef_
        return intercept, coefficients

    def predict_prices(self):
        predictions = self.model.predict(self.X)
        return predictions

if __name__ == "__main__":
    predictor = HousePricePredictor()
    predictor.load_dataset()
    predictor.prepare_data()
    predictor.train_model()
    intercept, coefficients = predictor.get_model_parameters()
    print("Intercept (b₀):", intercept)
    print("\nFeature Coefficients:")
    for feature, coefficient in zip(predictor.X.columns, coefficients):
        print(f"{feature}: {coefficient}")
    predictions = predictor.predict_prices()
    print("\nFirst Five Predicted House Prices:")
    for price in predictions[:5]:
        print(price)
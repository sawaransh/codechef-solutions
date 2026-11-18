import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class HousePriceAnalyzer:
    """
    Train a Multiple Linear Regression model
    and perform a complete model analysis.
    """

    def __init__(self):

        self.dataset = None

        self.X = None
        self.y = None

        self.model = None

    def load_dataset(self, filepath="house_price_analysis.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Store the input features
        self.X = self.dataset[["Area", "Bedrooms", "HouseAge", "DistanceFromCity"]]

        # Store the target value
        self.y = self.dataset["Price"]

    def train_model(self):

        # Create the Multiple Linear Regression model
        self.model = LinearRegression()

        # Train the model
        self.model.fit(self.X, self.y)

    def predict_prices(self, X=None):

        # Predict the house prices
        if X is None:
            X = self.X
        return self.model.predict(X)

    def get_intercept(self):

        # Return the learned intercept
        return self.model.intercept_

    def get_coefficients(self):

        # Return the regression coefficients
        return self.model.coef_

    def get_regression_equation(self):

        # Return the regression equation
        intercept = round(self.model.intercept_, 2)
        c1, c2, c3, c4 = [round(c, 2) for c in self.model.coef_]

        return (
            f"Price = {intercept} + "
            f"({c1} * Area) + "
            f"({c2} * Bedrooms) + "
            f"({c3} * HouseAge) + "
            f"({c4} * DistanceFromCity)"
        )

    def get_feature_effects(self):

        # Return whether each feature has
        # a positive or negative effect
        effects = []
        for feature, coef in zip(self.X.columns, self.model.coef_):
            effect = "Positive" if coef > 0 else "Negative"
            effects.append((feature, coef, effect))
        return effects

    def calculate_residuals(self):

        # Calculate the residuals
        y_pred = self.predict_prices()
        return self.y - y_pred

    def calculate_r2_score(self):

        # Calculate the R² Score
        y_pred = self.predict_prices()
        return r2_score(self.y, y_pred)

    def calculate_adjusted_r2(self):

        # Calculate the Adjusted R²
        r2 = self.calculate_r2_score()
        n = self.X.shape[0]
        p = self.X.shape[1]
        return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)


if __name__ == "__main__":

    analyzer = HousePriceAnalyzer()

    analyzer.load_dataset()

    analyzer.prepare_data()

    analyzer.train_model()

    print("Intercept:")
    print(analyzer.get_intercept())

    print("\nRegression Equation:")
    print(analyzer.get_regression_equation())

    print("\nFeature Effects:")

    effects = analyzer.get_feature_effects()

    for feature, coefficient, effect in effects:

        print(
            f"\n{feature}: "
            f"{coefficient:.2f} "
            f"({effect})"
        )

    print("\nFirst Five Residuals:")

    residuals = analyzer.calculate_residuals()

    for residual in residuals[:5]:

        print(f"{residual:.2f}")

    print("\nR² Score:")
    print(analyzer.calculate_r2_score())

    print("\nAdjusted R²:")
    print(analyzer.calculate_adjusted_r2())
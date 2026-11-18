
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class HousePriceAnalyzer:

    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.model = None

    # 1. Load dataset
    def load_dataset(self):
        self.dataset = pd.read_csv("house_price_analysis.csv")

    # 2. Prepare input and output data
    def prepare_data(self):
        self.X = self.dataset[["Area", "Bedrooms", "HouseAge", "DistanceFromCity"]]
        self.y = self.dataset["Price"]

    # 3. Train the model
    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    # 4. Predict house prices
    def predict_prices(self):
        predictions = self.model.predict(self.X)
        return predictions

    # 5. Get intercept
    def get_intercept(self):
        return self.model.intercept_

    # 6. Get coefficients
    def get_coefficients(self):
        return self.model.coef_

    # 7. Get regression equation
    def get_regression_equation(self):

        intercept = self.model.intercept_
        coefficients = self.model.coef_

        equation = (
            f"Price = {intercept} "
            f"+ ({coefficients[0]} * Area) "
            f"+ ({coefficients[1]} * Bedrooms) "
            f"+ ({coefficients[2]} * HouseAge) "
            f"+ ({coefficients[3]} * DistanceFromCity)"
        )

        return equation

    # 8. Get feature effects
    def get_feature_effects(self):

        feature_names = self.X.columns
        coefficients = self.model.coef_

        effects = []

        for feature, coefficient in zip(feature_names, coefficients):

            if coefficient > 0:
                effect = "Positive"
            elif coefficient < 0:
                effect = "Negative"
            else:
                effect = "Zero"

            effects.append((feature, coefficient, effect))

        return effects

    # 9. Calculate residuals
    def calculate_residuals(self):

        predictions = self.predict_prices()

        residuals = self.y - predictions

        return residuals

    # 10. Calculate R2 score
    def calculate_r2_score(self):

        predictions = self.predict_prices()

        score = r2_score(self.y, predictions)

        return score

    # 11. Calculate Adjusted R2
    def calculate_adjusted_r2(self):

        r2 = self.calculate_r2_score()

        n = len(self.y)
        p = self.X.shape[1]

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

        return adjusted_r2


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
            f"{feature}: "
            f"Coefficient: {coefficient:.2f} "
            f"({effect})"
        )

    print("\nFirst Five Residuals:")

    residuals = analyzer.calculate_residuals()

    for residual in residuals[:5]:
        print(f"{residual:.2f}")

    print("\nR2 Score:")
    print(analyzer.calculate_r2_score())

    print("\nAdjusted R2:")
    print(analyzer.calculate_adjusted_r2())

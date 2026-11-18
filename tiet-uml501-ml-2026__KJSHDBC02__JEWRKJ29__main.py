import os
import pandas as pd
from sklearn.linear_model import LinearRegression


class CustomerCreditLimitPredictor:
    """
    Train a Multiple Linear Regression model
    and interpret the regression results.
    """

    def __init__(self):

        self.dataset = None

        self.X = None
        self.y = None

        self.model = None

    def load_dataset(self):

        # Load the CSV file
        csv_path = "customer_credit_limit.csv"
        if not os.path.exists(csv_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "customer_credit_limit.csv")

        self.dataset = pd.read_csv(csv_path)
        return self.dataset

    def prepare_data(self):

        # Store the input features
        self.X = self.dataset[["MonthlyIncome", "CreditScore", "YearsEmployed"]]

        # Store the target value
        self.y = self.dataset["CreditLimit"]

        return self.X, self.y

    def train_model(self):

        # Create the Multiple Linear Regression model
        self.model = LinearRegression()

        # Train the model
        self.model.fit(self.X, self.y)

        return self.model

    def get_intercept(self):

        # Return the learned intercept
        return float(self.model.intercept_)

    def get_coefficients(self):

        # Return the regression coefficients
        return self.model.coef_

    def get_regression_equation(self):

        # Return the regression equation
        intercept = float(self.model.intercept_)
        coef1, coef2, coef3 = [float(c) for c in self.model.coef_]

        return (
            f"CreditLimit = {intercept:.2f} "
            f"+ ({coef1:.2f} * MonthlyIncome) "
            f"+ ({coef2:.2f} * CreditScore) "
            f"+ ({coef3:.2f} * YearsEmployed)"
        )

    def get_feature_effects(self):

        # Return the effect of each feature
        # as Positive or Negative
        effects = []
        for feature, coef in zip(self.X.columns, self.model.coef_):
            coef_val = float(coef)
            effect = "Positive" if coef_val >= 0 else "Negative"
            effects.append((feature, coef_val, effect))

        return effects


if __name__ == "__main__":

    predictor = CustomerCreditLimitPredictor()

    predictor.load_dataset()

    predictor.prepare_data()

    predictor.train_model()

    print("Intercept:")
    print(predictor.get_intercept())

    print("\nRegression Equation:")
    print(predictor.get_regression_equation())

    print("\nFeature Effects:")

    effects = predictor.get_feature_effects()

    for feature, coefficient, effect in effects:

        print(
            f"\n{feature}: "
            f"{coefficient:.2f} "
            f"({effect})"
        )
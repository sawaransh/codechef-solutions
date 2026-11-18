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

    def load_dataset(self, filepath="customer_credit_limit.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Store the input features
        self.X = self.dataset[["MonthlyIncome", "CreditScore", "YearsEmployed"]]

        # Store the target value
        self.y = self.dataset["CreditLimit"]

    def train_model(self):

        # Create the Multiple Linear Regression model
        self.model = LinearRegression()

        # Train the model
        self.model.fit(self.X, self.y)

    def get_intercept(self):

        # Return the learned intercept
        return self.model.intercept_

    def get_coefficients(self):

        # Return the regression coefficients
        return self.model.coef_

    def get_regression_equation(self):

        # Return the regression equation
        intercept = round(self.model.intercept_, 2)
        c1, c2, c3 = [round(c, 2) for c in self.model.coef_]

        return (
            f"CreditLimit = {intercept} + "
            f"({c1} * MonthlyIncome) + "
            f"({c2} * CreditScore) + "
            f"({c3} * YearsEmployed)"
        )

    def get_feature_effects(self):

        # Return the effect of each feature
        # as Positive or Negative
        effects = []
        for feature, coef in zip(self.X.columns, self.model.coef_):
            effect = "Positive" if coef > 0 else "Negative"
            effects.append((feature, coef, effect))
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
    for feature, coef, effect in predictor.get_feature_effects():
        print(f"\n{feature}: {round(coef, 2)} ({effect})")
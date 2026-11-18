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
        pass

    def prepare_data(self):

        # Store the input features
        self.X = None

        # Store the target value
        self.y = None

    def train_model(self):

        # Create the Multiple Linear Regression model
        self.model = None

        # Train the model
        pass

    def get_intercept(self):

        # Return the learned intercept
        pass

    def get_coefficients(self):

        # Return the regression coefficients
        pass

    def get_regression_equation(self):

        # Return the regression equation
        pass

    def get_feature_effects(self):

        # Return the effect of each feature
        # as Positive or Negative
        pass


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
            f"{feature}: "
            f"{coefficient:.2f} "
            f"({effect})"
        )
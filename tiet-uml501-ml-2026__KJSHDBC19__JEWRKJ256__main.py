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
    self.dataset = pd.read_csv("customer_credit_limit.csv")

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
    return round(float(self.model.intercept_), 2)

  def get_coefficients(self):
    # Return the regression coefficients
    return self.model.coef_

  def get_regression_equation(self):
    # Return the regression equation
    intercept = self.get_intercept()
    coefs = self.get_coefficients()
    return (
        f"CreditLimit = {intercept:.2f} + ({coefs[0]:.2f} * MonthlyIncome) + "
        f"({coefs[1]:.2f} * CreditScore) + ({coefs[2]:.2f} * YearsEmployed)"
    )

  def get_feature_effects(self):
    # Return the effect of each feature
    # as Positive or Negative
    effects = []
    for feature, coef in zip(self.X.columns, self.get_coefficients()):
      effect = "Positive" if coef >= 0 else "Negative"
      effects.append((feature, float(coef), effect))
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
    print(f"\n{feature}: {coefficient:.2f} ({effect})")
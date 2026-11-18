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

  def load_dataset(self):
    # Load the CSV file
    self.dataset = pd.read_csv("house_price_analysis.csv")

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

  def predict_prices(self):
    # Predict the house prices
    return self.model.predict(self.X)

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
        f"Price = {intercept:.2f} + ({coefs[0]:.2f} * Area) + ({coefs[1]:.2f} *"
        f" Bedrooms) + ({coefs[2]:.2f} * HouseAge) + ({coefs[3]:.2f} *"
        " DistanceFromCity)"
    )

  def get_feature_effects(self):
    # Return whether each feature has
    # a positive or negative effect
    effects = []
    for feature, coef in zip(self.X.columns, self.get_coefficients()):
      effect = "Positive" if coef >= 0 else "Negative"
      effects.append((feature, float(coef), effect))
    return effects

  def calculate_residuals(self):
    # Calculate the residuals
    predictions = self.predict_prices()
    return self.y - predictions

  def calculate_r2_score(self):
    # Calculate the R2 score
    predictions = self.predict_prices()
    return round(float(r2_score(self.y, predictions)), 4)

  def calculate_adjusted_r2(self):
    # Calculate the Adjusted R2
    r2 = r2_score(self.y, self.predict_prices())
    n = len(self.y)
    p = self.X.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    return round(float(adjusted_r2), 4)


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
    print(f"\n{feature}: {coefficient:.2f} ({effect})")

  print("\nFirst Five Residuals:")

  residuals = analyzer.calculate_residuals()

  for residual in residuals[:5]:
    print(f"{residual:.2f}")

  print("\nR2 Score:")
  print(analyzer.calculate_r2_score())

  print("\nAdjusted R2:")
  print(analyzer.calculate_adjusted_r2())
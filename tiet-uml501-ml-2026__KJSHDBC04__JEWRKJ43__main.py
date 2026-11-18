import pandas as pd
from sklearn.linear_model import Ridge


class LoanApprovalAnalyzer:
    """
    Train a Ridge Regression model
    and analyze the learned model.
    """

    def __init__(self):

        self.dataset = None

        self.X = None
        self.y = None

        self.model = None

    def load_dataset(self):

        # Load the CSV file
        self.dataset = pd.read_csv("loan_approval_ridge.csv")

    def prepare_data(self):

        # Store the input features
        self.X = self.dataset.drop(columns=["ApprovedLoanAmount"])

        # Store the target value
        self.y = self.dataset["ApprovedLoanAmount"]

    def train_model(self):

        # Create the Ridge Regression model
        self.model = Ridge()

        # Train the model
        self.model.fit(self.X, self.y)

    def predict_loan_amounts(self):

        # Predict the approved loan amounts
        return self.model.predict(self.X)

    def get_intercept(self):

        # Return the learned intercept
        return self.model.intercept_

    def get_coefficients(self):

        # Return the regression coefficients
        return self.model.coef_

    def get_regression_equation(self):

        # Return the regression equation
        terms = [
            f"({coef:.2f} * {feature})"
            for feature, coef in zip(self.X.columns, self.model.coef_)
        ]
        return f"ApprovedLoanAmount = {self.model.intercept_:.2f} + " + " + ".join(terms)

    def get_feature_effects(self):

        # Return whether each feature has
        # a positive or negative effect
        effects = []
        for feature, coef in zip(self.X.columns, self.model.coef_):
            effect = "Positive" if coef >= 0 else "Negative"
            effects.append((feature, coef, effect))
        return effects


if __name__ == "__main__":

    analyzer = LoanApprovalAnalyzer()

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
            f"{coefficient:.2f} "
            f"({effect})"
        )

    print("\nPredicted Loan Amounts:")

    predictions = analyzer.predict_loan_amounts()

    for prediction in predictions:

        print(f"{prediction:.2f}")
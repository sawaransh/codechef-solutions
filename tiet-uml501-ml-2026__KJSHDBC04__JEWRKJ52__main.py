import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


class RegularizationAnalyzer:
    """
    Train Ridge, Lasso, and ElasticNet
    Regression models and compare them.
    """

    def __init__(self):

        self.dataset = None

        self.X = None
        self.y = None

        self.ridge_model = None

        self.lasso_model = None

        self.elasticnet_model = None

    def load_dataset(self):

        # Load the CSV file
        self.dataset = pd.read_csv("loan_regularization.csv")

    def prepare_data(self):

        # Store the input features
        self.X = self.dataset.drop(columns=["ApprovedLoanAmount"])

        # Store the target value
        self.y = self.dataset["ApprovedLoanAmount"]

    def train_models(self):

        # Create the Ridge Regression model
        self.ridge_model = Ridge()

        # Create the Lasso Regression model
        self.lasso_model = Lasso()

        # Create the ElasticNet Regression model
        self.elasticnet_model = ElasticNet()

        # Train all three models
        self.ridge_model.fit(self.X, self.y)
        self.lasso_model.fit(self.X, self.y)
        self.elasticnet_model.fit(self.X, self.y)

    def predict_ridge(self):

        # Predict using the Ridge model
        return self.ridge_model.predict(self.X)

    def predict_lasso(self):

        # Predict using the Lasso model
        return self.lasso_model.predict(self.X)

    def predict_elasticnet(self):

        # Predict using the ElasticNet model
        return self.elasticnet_model.predict(self.X)

    def get_ridge_coefficients(self):

        # Return Ridge coefficients
        return self.ridge_model.coef_

    def get_lasso_coefficients(self):

        # Return Lasso coefficients
        return self.lasso_model.coef_

    def get_elasticnet_coefficients(self):

        # Return ElasticNet coefficients
        return self.elasticnet_model.coef_


if __name__ == "__main__":

    analyzer = RegularizationAnalyzer()

    analyzer.load_dataset()

    analyzer.prepare_data()

    analyzer.train_models()

    print("Ridge Regression Coefficients:")
    print(
        analyzer.get_ridge_coefficients()
    )

    print("\nLasso Regression Coefficients:")
    print(
        analyzer.get_lasso_coefficients()
    )

    print("\nElasticNet Regression Coefficients:")
    print(
        analyzer.get_elasticnet_coefficients()
    )

    print("\nRidge Predictions:")
    print(analyzer.predict_ridge())

    print("\nLasso Predictions:")
    print(analyzer.predict_lasso())

    print("\nElasticNet Predictions:")
    print(analyzer.predict_elasticnet())
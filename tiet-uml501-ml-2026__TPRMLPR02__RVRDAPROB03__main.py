
import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score
)


class RegressionModelComparisonAnalysis:
    """
    Compare Linear Regression and
    Logistic Regression models.
    """

    def __init__(self):
        """
        Load the dataset from customer_purchase_prediction.csv.
        """

        self.df = pd.read_csv(
            'customer_purchase_prediction.csv'
        )

        self.X = None
        self.y = None

        self.linear_model = None
        self.logistic_model = None

        self.linear_regression_model = None
        self.logistic_regression_model = None

        self.linear_predictions = None
        self.logistic_predictions = None

    def prepare_data(self):
        """
        Create feature matrix (X)
        and target vector (y).
        """

        self.X = self.df.drop(
            columns=['Purchased']
        )

        self.y = self.df['Purchased']

        # Ensure features are numerical
        self.X = self.X.apply(
            pd.to_numeric,
            errors='coerce'
        )

        # Handle missing feature values
        self.X = self.X.fillna(
            self.X.median()
        )

        # Convert target to numerical values
        self.y = pd.to_numeric(
            self.y,
            errors='coerce'
        )

        # Remove rows with missing target values
        valid_rows = self.y.notna()

        self.X = self.X.loc[valid_rows]
        self.y = self.y.loc[valid_rows]

        self.y = self.y.astype(int)

        return self.X, self.y

    def train_linear_regression_model(self):
        """
        Train a Linear Regression model
        and store it for later retrieval.
        """

        self.linear_model = LinearRegression()

        self.linear_model.fit(
            self.X,
            self.y
        )

        self.linear_regression_model = self.linear_model

        self.linear_predictions = self.linear_model.predict(
            self.X
        )

        return self.linear_model

    def train_logistic_regression_model(self):
        """
        Train a Logistic Regression model
        and store it for later retrieval.
        """

        self.logistic_model = LogisticRegression(
            max_iter=2000
        )

        self.logistic_model.fit(
            self.X,
            self.y
        )

        self.logistic_regression_model = self.logistic_model

        self.logistic_predictions = self.logistic_model.predict(
            self.X
        )

        return self.logistic_model

    def evaluate_linear_regression(self):
        """
        Calculate:
        - Mean Squared Error (MSE)
        - R2 Score
        """

        predictions = self.linear_model.predict(
            self.X
        )

        mse = mean_squared_error(
            self.y,
            predictions
        )

        r2 = r2_score(
            self.y,
            predictions
        )

        return {
            'mse': mse,
            'r2': r2
        }

    def evaluate_logistic_regression(self):
        """
        Calculate classification accuracy.
        """

        predictions = self.logistic_model.predict(
            self.X
        )

        accuracy = accuracy_score(
            self.y,
            predictions
        )

        return {
            'accuracy': accuracy
        }

    def compare_models(self):
        """
        Generate a comparison between
        Linear Regression and Logistic Regression.
        """

        linear_results = self.evaluate_linear_regression()

        logistic_results = self.evaluate_logistic_regression()

        comparison = {
            'linear_regression': linear_results,
            'logistic_regression': logistic_results
        }

        return comparison

    def get_dataset(self):
        """
        Return the loaded dataset.
        """

        return self.df

    def generate_analysis_summary(self):
        """
        Generate a summary containing:
        - Total records
        - Number of features
        - Linear Regression MSE
        - Linear Regression R2
        - Logistic Regression Accuracy
        """

        linear_results = self.evaluate_linear_regression()

        logistic_results = self.evaluate_logistic_regression()

        summary = {
            'Total Records': len(self.df),

            'Number of Features': self.X.shape[1],

            'Linear Regression MSE': linear_results['mse'],

            'Linear Regression R2': linear_results['r2'],

            'Logistic Regression Accuracy': logistic_results['accuracy']
        }

        return summary


if __name__ == "__main__":

    analysis = RegressionModelComparisonAnalysis()

    # Prepare dataset
    analysis.prepare_data()

    # Train both models
    analysis.train_linear_regression_model()

    analysis.train_logistic_regression_model()

    # Generate analysis summary
    print(
        "Analysis Summary:"
    )

    print(
        analysis.generate_analysis_summary()
    )

    # Compare models
    print(
        "\nModel Comparison:"
    )

    print(
        analysis.compare_models()
    )

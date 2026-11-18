import pandas as pd
from sklearn.linear_model import LinearRegression


class ElectricityBillPrediction:
    """
    Train a Simple Linear Regression
    model and use the learned
    hypothesis to predict the
    electricity bill.
    """

    def __init__(self):

        self.dataset = pd.read_csv(
            "electricity_bill.csv"
        )

    def prepare_data(self):

        # Store the input feature
        self.X = self.dataset[["UnitsConsumed"]]

        # Store the target value
        self.y = self.dataset["ElectricityBill"]

    def train_model(self):

        # Create the Linear Regression model
        self.model = LinearRegression()

        # Train the model
        self.model.fit(self.X, self.y)

    def get_hypothesis(self):

        # Retrieve the intercept
        intercept = self.model.intercept_

        # Retrieve the slope
        slope = self.model.coef_[0]

        return {
            "intercept": intercept,
            "slope": slope
        }

    def predict_bill(self):

        # Predict the electricity bill
        bill=pd.DataFrame({"UnitsConsumed":[350]})
        prediction = self.model.predict(bill)

        return prediction[0]


if __name__ == "__main__":

    predictor = ElectricityBillPrediction()

    predictor.prepare_data()

    predictor.train_model()

    hypothesis = predictor.get_hypothesis()

    print("Intercept (b₀):", hypothesis["intercept"])
    print("Slope (b₁):", hypothesis["slope"])

    print("\nHypothesis:")
    print(
        f"ŷ = {hypothesis['intercept']} + "
        f"({hypothesis['slope']} × x)"
    )

    print("\nUnits Consumed: 350")
    print("Predicted Electricity Bill:", predictor.predict_bill())
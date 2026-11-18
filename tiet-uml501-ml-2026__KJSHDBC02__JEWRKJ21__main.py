import pandas as pd
from sklearn.linear_model import LinearRegression


class EmployeeIncomePredictor:
    """
    Train a Multiple Linear Regression model
    and interpret the learned coefficients.
    """

    def __init__(self):

        self.dataset = pd.read_csv("employee_income.csv")

        self.X = self.dataset[["Experience", "EducationLevel", "WorkingHours"]]
        self.y = self.dataset["AnnualIncome"]

        self.model = LinearRegression()

    def load_dataset(self):
        self.dataset=pd.read_csv("employee_income.csv")
    def prepare_data(self):
        self.X = self.dataset[["Experience", "EducationLevel", "WorkingHours"]]
        self.y = self.dataset["AnnualIncome"]

    def train_model(self):

        # Create the Multiple Linear Regression model
        self.model = LinearRegression()

        # Train the model
        self.model.fit(self.X, self.y)

    def get_model_parameters(self):

        intercept = self.model.intercept_
        coefficients = self.model.coef_
        return intercept, coefficients

    def get_coefficient_effects(self):
        coefficients=self.model.coef_
        effects = []
        for c in coefficients:
            if c > 0:
                effects.append("Positive")
            elif c < 0:
                effects.append("Negative")
            else:
                effects.append("Zero") 

        return effects


if __name__ == "__main__":

    predictor = EmployeeIncomePredictor()

    predictor.load_dataset()

    predictor.prepare_data()

    predictor.train_model()

    intercept, coefficients = predictor.get_model_parameters()

    print("Intercept (b₀):", intercept)

    print("\nFeature Coefficients:")

    effects = predictor.get_coefficient_effects()

    for feature, coefficient, effect in zip(
        predictor.X.columns,
        coefficients,
        effects
    ):
        print(f"{feature}: {coefficient} ({effect})")
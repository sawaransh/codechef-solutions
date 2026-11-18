import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class MedicalExpensePredictor:
    """
    Train a Multiple Linear Regression model
    and evaluate it using model diagnostics.
    """

    def __init__(self):

        self.dataset = None

        self.X = None
        self.y = None

        self.model = None

    def load_dataset(self):
        self.dataset=pd.read_csv("medical_expenses.csv")
    def prepare_data(self):

        self.X = self.dataset[["Age", "BMI", "HospitalVisits"]]
        self.y = self.dataset["MedicalExpenses"]

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def predict_expenses(self):
        predictions = self.model.predict(self.X)
        return predictions

    def calculate_residuals(self):
        predictions=self.model.predict(self.X)
        residuals = self.y-predictions
        return residuals

    def calculate_r2_score(self):
        predictions=self.model.predict(self.X)
        r2 = r2_score(self.y, predictions)
        return r2

    def calculate_adjusted_r2(self):
        predictions=self.model.predict(self.X)
        n=len(self.y)
        p=self.X.shape[1]
        r2=r2_score(self.y, predictions)
        adjusted_r2 = 1-((1-r2)*(n-1)/(n-p-1)) 
        return adjusted_r2


if __name__ == "__main__":

    predictor = MedicalExpensePredictor()

    predictor.load_dataset()

    predictor.prepare_data()

    predictor.train_model()

    residuals = predictor.calculate_residuals()

    print("First Five Residuals:")
    for residual in residuals[:5]:
        print(residual)

    print("\nR² Score:")
    print(predictor.calculate_r2_score())

    print("\nAdjusted R²:")
    print(predictor.calculate_adjusted_r2())
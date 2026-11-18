import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
class ApartmentRentCostCalculator:
    def __init__(self):
        self.dataset = pd.read_csv("apartment_rent.csv")

    def prepare_data(self):
        self.X = self.dataset [["ApartmentSize"]]
        self.y = self.dataset ["MonthlyRent"]
        print ("Total records:", len(self.dataset))

    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def predict_rent(self):
        predictions = self.model.predict(self.X)
        return predictions

    def calculate_cost(self):
        predictions = self.predict_rent()
        squared_errors = (predictions-self.y)**2
        n=len(self.y)
        cost = np.sum(squared_errors)/(2*n)
        return cost
if __name__ == "__main__":
    calculator = ApartmentRentCostCalculator()
    calculator.prepare_data()
    calculator.train_model()
    print("Total records:", len(calculator.dataset))
    predictions = calculator.predict_rent()
    print("\nPredictions generated.")
    print("\nCost:", calculator.calculate_cost())
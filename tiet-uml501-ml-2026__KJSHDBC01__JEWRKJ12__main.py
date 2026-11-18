import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
class WorkoutCaloriesAnalyzer:
    def __init__(self):
        self.dataset = pd.read_csv("workout_calories.csv")
        self.X = self.dataset[["WorkoutMinutes"]]
        self.y = self.dataset["CaloriesBurned"]
        self.model = LinearRegression
    def load_dataset(self):
        self.dataset=pd.read_csv("workout_calories.csv")
    def prepare_data(self):
        self.X = self.dataset[["WorkoutMinutes"]]
        self.y = self.dataset["CaloriesBurned"]
    def train_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
    def get_hypothesis(self):
        intercept = self.model.intercept_
        slope = self.model.coef_[0]
        return intercept, slope
    def predict_calories(self):
        predictions = self.model.predict(self.X)
        return predictions
    def calculate_cost(self):
        predictions=self.model.predict(self.X)
        squared_errors=(predictions-self.y)**2
        n=len(self.y)
        cost = np.sum(squared_errors)/(2*n)
        return cost
if __name__ == "__main__":
    analyzer = WorkoutCaloriesAnalyzer()
    analyzer.load_dataset()
    analyzer.prepare_data()
    analyzer.train_model()
    intercept, slope = analyzer.get_hypothesis()
    print("Intercept (b₀):", intercept)
    print("Slope (b₁):", slope)
    print("\nHypothesis:")
    print(f"ŷ = {intercept} + ({slope} × x)")
    print("\nCost:")
    print(analyzer.calculate_cost())
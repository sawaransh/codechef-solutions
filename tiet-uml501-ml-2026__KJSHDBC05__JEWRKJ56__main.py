import math
import pandas as pd


class SigmoidProbabilityCalculator:
    """
    Calculate probabilities using
    the Sigmoid Function.
    """

    def __init__(self):

        self.dataset = None

        self.intercept = -8

        self.coefficient = 0.08

    def load_dataset(self):

        # Load the CSV file
        self.dataset = pd.read_csv("patient_disease_probability.csv")

    def calculate_sigmoid(self, z):

        # Calculate the Sigmoid Function
        return 1 / (1 + math.exp(-z))

    def calculate_probabilities(self):

        # Calculate the predicted
        # probabilities
        probabilities = []
        for x in self.dataset["BloodSugarLevel"]:
            z = self.intercept + self.coefficient * x
            probabilities.append(self.calculate_sigmoid(z))
        return probabilities


if __name__ == "__main__":

    calculator = SigmoidProbabilityCalculator()

    calculator.load_dataset()

    probabilities = (
        calculator.calculate_probabilities()
    )

    print("Predicted Probabilities:")

    for probability in probabilities:

        print(
            f"{probability:.4f}"
        )
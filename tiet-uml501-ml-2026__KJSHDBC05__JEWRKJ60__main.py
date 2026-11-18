import math
import pandas as pd


class LoanRepaymentClassifier:
    """
    Calculate probabilities using
    the Sigmoid Function and
    classify using a decision boundary.
    """

    def __init__(self):

        self.dataset = None

        self.intercept = -8

        self.coefficient = 0.015

    def load_dataset(self):

        # Load the CSV file
        self.dataset = pd.read_csv("loan_repayment_prediction.csv")

    def calculate_sigmoid(self, z):

        # Calculate the Sigmoid Function
        return 1 / (1 + math.exp(-z))

    def predict_probabilities(self):

        # Return the predicted
        # probabilities
        probabilities = []
        for x in self.dataset["CreditScore"]:
            z = self.intercept + self.coefficient * x
            probabilities.append(self.calculate_sigmoid(z))
        return probabilities

    def predict_classes(self):

        # Return the predicted
        # classes using the
        # decision boundary
        probabilities = self.predict_probabilities()
        return [1 if p >= 0.5 else 0 for p in probabilities]


if __name__ == "__main__":

    classifier = LoanRepaymentClassifier()

    classifier.load_dataset()

    probabilities = (
        classifier.predict_probabilities()
    )

    classes = (
        classifier.predict_classes()
    )

    print("Predicted Probabilities:")

    for probability in probabilities:

        print(
            f"{probability:.4f}"
        )

    print("\nPredicted Classes:")

    for predicted_class in classes:

        print(predicted_class)
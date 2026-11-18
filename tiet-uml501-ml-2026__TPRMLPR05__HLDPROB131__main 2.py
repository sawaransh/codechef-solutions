import pandas as pd

class NaiveBayesClassifier:

    def __init__(self, filepath):
        """
        Load dataset from CSV.
        """
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)
        self.target_col = self.df.columns[-1]

    def prior_probability(self, class_label):
        """
        Compute P(class).
        """
        total_rows = len(self.df)
        class_rows = len(self.df[self.df[self.target_col] == class_label])
        return class_rows / total_rows

    def conditional_probability(self, column, value, class_label):
        """
        Compute P(feature=value | class).
        """
        class_subset = self.df[self.df[self.target_col] == class_label]
        matching_rows = len(class_subset[class_subset[column] == value])
        return matching_rows / len(class_subset)

    def predict(self, sample):
        """
        Predict class label using Naïve Bayes.
        """
        # Calculate probability for class 'yes'
        prob_yes = self.prior_probability("yes")
        for column, value in sample.items():
            prob_yes *= self.conditional_probability(column, value, "yes")

        # Calculate probability for class 'no'
        prob_no = self.prior_probability("no")
        for column, value in sample.items():
            prob_no *= self.conditional_probability(column, value, "no")

        # Return class with highest probability
        return "yes" if prob_yes > prob_no else "no"


def run_classifier():
    model = NaiveBayesClassifier("allelectronics.csv")

    sample = {
        "age": "youth",
        "income": "medium",
        "student": "yes",
        "credit_rating": "fair"
    }

    prediction = model.predict(sample)
    print(f"\nPredicted Class: {prediction}")
    return prediction


if __name__ == "__main__":
    run_classifier()
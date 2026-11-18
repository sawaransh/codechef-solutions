import pandas as pd
from imblearn.over_sampling import SMOTE


class LoanDefaultBalancer:
    """
    Balance an imbalanced dataset
    using SMOTE.
    """

    def __init__(self):

        self.dataset = None

        self.balanced_dataset = None

    def load_dataset(self, filepath="loan_default_imbalanced.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def apply_smote(self):

        # Create the balanced dataset
        # using SMOTE
        X = self.dataset[["AnnualIncome", "CreditScore"]]
        y = self.dataset["LoanDefault"]

        min_samples = y.value_counts().min()
        k_neighbors = min(5, min_samples - 1)

        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.balanced_dataset = pd.concat([X_resampled, y_resampled], axis=1)

    def get_balanced_dataset(self):

        # Return the balanced dataset
        return self.balanced_dataset

    def get_class_distribution(self):

        # Return the class distribution
        # after applying SMOTE
        return self.balanced_dataset["LoanDefault"].value_counts().sort_index()


if __name__ == "__main__":

    balancer = LoanDefaultBalancer()

    balancer.load_dataset()

    balancer.apply_smote()

    print("Balanced Dataset:")

    print(
        balancer.get_balanced_dataset()
    )

    print("\nClass Distribution:")

    print(
        balancer.get_class_distribution()
    )
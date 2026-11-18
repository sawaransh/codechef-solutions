import pandas as pd
from sklearn.utils import resample


class InsuranceDatasetBalancer:
    """
    Balance an imbalanced dataset
    using Random Undersampling.
    """

    def __init__(self):
        self.dataset = None
        self.balanced_dataset = None

    def load_dataset(self, filepath="insurance_claim_imbalanced.csv"):
        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def undersample_dataset(self):
        # Create the balanced dataset
        # using Random Undersampling
        class_counts = self.dataset["FiledClaim"].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        majority_df = self.dataset[self.dataset["FiledClaim"] == majority_class]
        minority_df = self.dataset[self.dataset["FiledClaim"] == minority_class]

        majority_undersampled = resample(
            majority_df,
            replace=False,
            n_samples=len(minority_df),
            random_state=42
        )

        self.balanced_dataset = pd.concat(
            [majority_undersampled, minority_df]
        ).reset_index(drop=True)

    def get_balanced_dataset(self):
        # Return the balanced dataset
        return self.balanced_dataset

    def get_class_distribution(self):
        # Return the class distribution
        # after undersampling
        return self.balanced_dataset["FiledClaim"].value_counts().sort_index()


if __name__ == "__main__":

    balancer = InsuranceDatasetBalancer()

    balancer.load_dataset()

    balancer.undersample_dataset()

    print("Balanced Dataset:")

    print(
        balancer.get_balanced_dataset()
    )

    print("\nClass Distribution:")

    print(
        balancer.get_class_distribution()
    )
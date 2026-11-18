import pandas as pd
from sklearn.utils import resample


class DiseaseDatasetBalancer:
    """
    Balance an imbalanced dataset
    using Random Oversampling.
    """

    def __init__(self):
        self.dataset = None
        self.balanced_dataset = None
        self.target_col = None

    def load_dataset(self, filepath="patient_disease_imbalanced.csv"):
        # Load the CSV file
        self.dataset = pd.read_csv(filepath)
        self.target_col = self.dataset.columns[-1]

    def oversample_dataset(self):
        # Create the balanced dataset
        # using Random Oversampling
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        max_count = self.dataset[self.target_col].value_counts().max()
        balanced_groups = []

        for _, group in self.dataset.groupby(self.target_col):
            if len(group) < max_count:
                group = resample(
                    group,
                    replace=True,
                    n_samples=max_count,
                    random_state=42
                )
            balanced_groups.append(group)

        self.balanced_dataset = pd.concat(balanced_groups).reset_index(drop=True)

    def get_balanced_dataset(self):
        # Return the balanced dataset
        return self.balanced_dataset

    def get_class_distribution(self):
        # Return the class distribution
        # after oversampling
        if self.balanced_dataset is None:
            raise ValueError("Dataset not balanced yet. Call oversample_dataset() first.")
        return self.balanced_dataset[self.target_col].value_counts()


if __name__ == "__main__":
    balancer = DiseaseDatasetBalancer()
    balancer.load_dataset()
    balancer.oversample_dataset()

    print("Balanced Dataset:")
    print(balancer.get_balanced_dataset())

    print("\nClass Distribution:")
    print(balancer.get_class_distribution())
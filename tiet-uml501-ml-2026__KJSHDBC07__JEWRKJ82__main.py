import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


class ClassImbalanceAnalyzer:
    """
    Compare Random Oversampling, Random Undersampling,
    and SMOTE on an imbalanced dataset.
    """

    def __init__(self):
        self.dataset = None
        self.oversampled_dataset = None
        self.undersampled_dataset = None
        self.smote_dataset = None

    def load_dataset(self, filepath="loan_default_techniques.csv"):
        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def apply_oversampling(self):
        # Create the balanced dataset
        # using Random Oversampling
        class_counts = self.dataset["LoanDefault"].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        majority_df = self.dataset[self.dataset["LoanDefault"] == majority_class]
        minority_df = self.dataset[self.dataset["LoanDefault"] == minority_class]

        minority_oversampled = resample(
            minority_df,
            replace=True,
            n_samples=len(majority_df),
            random_state=42
        )

        self.oversampled_dataset = pd.concat(
            [majority_df, minority_oversampled]
        ).reset_index(drop=True)

    def apply_undersampling(self):
        # Create the balanced dataset
        # using Random Undersampling
        class_counts = self.dataset["LoanDefault"].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        majority_df = self.dataset[self.dataset["LoanDefault"] == majority_class]
        minority_df = self.dataset[self.dataset["LoanDefault"] == minority_class]

        majority_undersampled = resample(
            majority_df,
            replace=False,
            n_samples=len(minority_df),
            random_state=42
        )

        self.undersampled_dataset = pd.concat(
            [majority_undersampled, minority_df]
        ).reset_index(drop=True)

    def apply_smote(self):
        # Create the balanced dataset
        # using SMOTE
        X = self.dataset[["AnnualIncome", "CreditScore"]]
        y = self.dataset["LoanDefault"]

        min_samples = y.value_counts().min()
        k_neighbors = max(1, min(5, min_samples - 1))

        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.smote_dataset = pd.concat([X_resampled, y_resampled], axis=1)

    def get_oversampled_dataset(self):
        # Return the oversampled dataset
        return self.oversampled_dataset

    def get_undersampled_dataset(self):
        # Return the undersampled dataset
        return self.undersampled_dataset

    def get_smote_dataset(self):
        # Return the SMOTE dataset
        return self.smote_dataset

    def get_oversampling_distribution(self):
        # Return the class distribution
        # after oversampling
        return self.oversampled_dataset["LoanDefault"].value_counts().sort_index()

    def get_undersampling_distribution(self):
        # Return the class distribution
        # after undersampling
        return self.undersampled_dataset["LoanDefault"].value_counts().sort_index()

    def get_smote_distribution(self):
        # Return the class distribution
        # after applying SMOTE
        return self.smote_dataset["LoanDefault"].value_counts().sort_index()


if __name__ == "__main__":

    analyzer = ClassImbalanceAnalyzer()

    analyzer.load_dataset()

    analyzer.apply_oversampling()

    analyzer.apply_undersampling()

    analyzer.apply_smote()

    print("Oversampling Class Distribution:")

    print(
        analyzer.get_oversampling_distribution()
    )

    print("\nUndersampling Class Distribution:")

    print(
        analyzer.get_undersampling_distribution()
    )

    print("\nSMOTE Class Distribution:")

    print(
        analyzer.get_smote_distribution()
    )
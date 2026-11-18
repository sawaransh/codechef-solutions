import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)


class CustomerSimilarityAnalysis:
    """
    Perform customer similarity and
    correlation analysis using a
    transformed customer dataset.
    """

    def __init__(self):
        """
        Load the dataset from
        AWCustomers.csv.
        """
        self.df = pd.read_csv("AWCustomers.csv")
        self.onehot_cols = []

    def preprocess_data(self):
        """
        Prepare the dataset by:

        - Removing duplicate rows
        - Filling missing numeric
          values using the median
        - Filling missing categorical
          values using the mode
        - Applying Min-Max
          Normalization to
          YearlyIncome
        - Applying Z-score
          standardization to:
          - NumberCarsOwned
          - NumberChildrenAtHome
          - TotalChildren
        - Applying One-Hot Encoding
          to:
          - Gender
          - MaritalStatus
          - Education
          - Occupation
          - CountryRegionName
        """
        self.df = self.df.drop_duplicates().reset_index(drop=True)

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val[0])

        minmax = MinMaxScaler()
        self.df["YearlyIncome"] = minmax.fit_transform(self.df[["YearlyIncome"]])

        std_cols = ["NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren"]
        scaler = StandardScaler()
        self.df[std_cols] = scaler.fit_transform(self.df[std_cols])

        cat_cols = [
            "Gender",
            "MaritalStatus",
            "Education",
            "Occupation",
            "CountryRegionName"
        ]
        before_cols = set(self.df.columns)
        self.df = pd.get_dummies(self.df, columns=cat_cols, dtype=int)
        self.onehot_cols = [col for col in self.df.columns if col not in before_cols]

    def simple_matching_similarity(self):
        """
        Calculate the Simple Matching
        Similarity between row 0 and
        row 1 of the transformed
        dataset.

        Return the similarity value.
        """
        row0 = self.df.iloc[0]
        row1 = self.df.iloc[1]
        return float((row0 == row1).mean())

    def jaccard_similarity(self):
        """
        Calculate the Jaccard
        Similarity between row 0 and
        row 1 using only the one-hot
        encoded binary features.

        Return the similarity value.
        """
        v0 = self.df.loc[0, self.onehot_cols].values
        v1 = self.df.loc[1, self.onehot_cols].values
        intersection = np.sum((v0 == 1) & (v1 == 1))
        union = np.sum((v0 == 1) | (v1 == 1))
        return float(intersection / union) if union != 0 else 0.0

    def cosine_similarity(self):
        """
        Calculate the Cosine
        Similarity between row 0 and
        row 1 using only the numeric
        features.

        Return the similarity value.
        """
        num_df = self.df.select_dtypes(include=[np.number])
        v0 = num_df.iloc[0].values
        v1 = num_df.iloc[1].values
        denom = np.linalg.norm(v0) * np.linalg.norm(v1)
        return float(np.dot(v0, v1) / denom) if denom != 0 else 0.0

    def feature_correlation(self):
        """
        Calculate the Pearson
        Correlation Coefficient
        between:

        - NumberCarsOwned
        - YearlyIncome

        Return the correlation value.
        """
        return float(self.df["NumberCarsOwned"].corr(self.df["YearlyIncome"]))

    def get_dataset(self):
        """
        Return the transformed
        dataset.
        """
        return self.df


if __name__ == "__main__":

    analyzer = CustomerSimilarityAnalysis()

    analyzer.preprocess_data()

    print("Simple Matching Similarity:")
    print(analyzer.simple_matching_similarity())

    print("\nJaccard Similarity:")
    print(analyzer.jaccard_similarity())

    print("\nCosine Similarity:")
    print(analyzer.cosine_similarity())

    print("\nCorrelation:")
    print(analyzer.feature_correlation())

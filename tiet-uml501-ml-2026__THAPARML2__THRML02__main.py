import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)


class CustomerDataTransformation:
    """
    Implement data preprocessing and
    transformation techniques to
    prepare customer data for
    data mining.
    """

    def __init__(self):
        """
        Load the dataset from
        AWCustomers.csv.
        """
        self.df = pd.read_csv("AWCustomers.csv")

    def handle_null_values(self):
        """
        Handle missing values by:

        - Removing duplicate rows.
        - Filling missing numeric
          values using the median.
        - Filling missing categorical
          values using the mode.
        """
        self.df = self.df.drop_duplicates()

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val[0])

    def normalize_data(self):
        """
        Apply Min-Max Normalization
        to the YearlyIncome attribute.

        Store the normalized values
        in a new column named:

        - NormalizedIncome
        """
        scaler = MinMaxScaler()
        self.df["NormalizedIncome"] = scaler.fit_transform(self.df[["YearlyIncome"]])

    def discretize_data(self):
        """
        Discretize the YearlyIncome
        attribute into 4 equal-width
        bins.

        Store the result in a new
        column named:

        - IncomeBin
        """
        self.df["IncomeBin"] = pd.cut(self.df["YearlyIncome"], bins=4)

    def standardize_data(self):
        """
        Apply Z-score
        standardization to:

        - NumberCarsOwned
        - NumberChildrenAtHome
        - TotalChildren
        """
        cols = ["NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren"]
        scaler = StandardScaler()
        self.df[cols] = scaler.fit_transform(self.df[cols])

    def one_hot_encode(self):
        """
        Apply One Hot Encoding to:

        - Occupation
        - CountryRegionName
        """
        self.df = pd.get_dummies(self.df, columns=["Occupation", "CountryRegionName"])

    def get_dataset(self):
        """
        Return the transformed
        dataset.
        """
        return self.df


if __name__ == "__main__":

    transformer = (
        CustomerDataTransformation()
    )

    transformer.handle_null_values()

    transformer.normalize_data()

    transformer.discretize_data()

    transformer.standardize_data()

    transformer.one_hot_encode()

    print(
        "Transformed Dataset:"
    )

    print(
        transformer.get_dataset()
    )

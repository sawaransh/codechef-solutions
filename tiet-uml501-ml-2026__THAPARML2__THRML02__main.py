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

        # Load the dataset
        pass

    def handle_null_values(self):
        """
        Handle missing values by:

        - Removing duplicate rows.
        - Filling missing numeric
          values using the median.
        - Filling missing categorical
          values using the mode.
        """

        pass

    def normalize_data(self):
        """
        Apply Min-Max Normalization
        to the YearlyIncome attribute.

        Store the normalized values
        in a new column named:

        - NormalizedIncome
        """

        pass

    def discretize_data(self):
        """
        Discretize the YearlyIncome
        attribute into 4 equal-width
        bins.

        Store the result in a new
        column named:

        - IncomeBin
        """

        pass

    def standardize_data(self):
        """
        Apply Z-score
        standardization to:

        - NumberCarsOwned
        - NumberChildrenAtHome
        - TotalChildren
        """

        pass

    def one_hot_encode(self):
        """
        Apply One Hot Encoding to:

        - Occupation
        - CountryRegionName
        """

        pass

    def get_dataset(self):
        """
        Return the transformed
        dataset.
        """

        pass


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
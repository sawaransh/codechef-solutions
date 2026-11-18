import pandas as pd
import numpy as np


class CustomerFeatureSelection:
    """
    Implement feature selection and
    preprocessing analysis to prepare
    customer data for data mining.
    """

    def __init__(self):
        """
        Load the dataset from
        AWCustomers.csv.
        """

        # Load the dataset
        pass

    def clean_dataset(self):
        """
        Clean the dataset by:

        - Removing duplicate rows.
        - Filling missing numeric
          values using the median.
        - Filling missing categorical
          values using the mode.
        """

        pass

    def analyze_attributes(self):
        """
        Return a dictionary containing:

        - total_records
        - total_attributes
        - missing_values
        - duplicate_rows
        - data_types
        """

        pass

    def select_features(self):
        """
        Select the required attributes
        and store the selected dataset.
        """

        pass

    def get_selected_dataset(self):
        """
        Return the selected dataset.
        """

        pass

    def analyze_attribute_types(self):
        """
        This helper method is already
        implemented.

        Do not modify this method.
        """

        return {
            "BirthDate": {
                "value_type": "Discrete",
                "measurement_scale": "Interval"
            },
            "Education": {
                "value_type": "Discrete",
                "measurement_scale": "Ordinal"
            },
            "Occupation": {
                "value_type": "Discrete",
                "measurement_scale": "Nominal"
            },
            "Gender": {
                "value_type": "Discrete",
                "measurement_scale": "Nominal"
            },
            "MaritalStatus": {
                "value_type": "Discrete",
                "measurement_scale": "Nominal"
            },
            "HomeOwnerFlag": {
                "value_type": "Discrete",
                "measurement_scale": "Nominal"
            },
            "NumberCarsOwned": {
                "value_type": "Discrete",
                "measurement_scale": "Ratio"
            },
            "NumberChildrenAtHome": {
                "value_type": "Discrete",
                "measurement_scale": "Ratio"
            },
            "TotalChildren": {
                "value_type": "Discrete",
                "measurement_scale": "Ratio"
            },
            "YearlyIncome": {
                "value_type": "Continuous",
                "measurement_scale": "Ratio"
            },
            "CountryRegionName": {
                "value_type": "Discrete",
                "measurement_scale": "Nominal"
            }
        }

    def generate_preprocessing_plan(self):
        """
        This helper method is already
        implemented.

        Do not modify this method.
        """

        return {
            "BirthDate": "date",
            "Education": "ordinal",
            "Occupation": "onehot",
            "Gender": "label",
            "MaritalStatus": "label",
            "HomeOwnerFlag": "none",
            "NumberCarsOwned": "standardize",
            "NumberChildrenAtHome": "standardize",
            "TotalChildren": "standardize",
            "YearlyIncome": "normalize",
            "CountryRegionName": "onehot"
        }

    def get_dataset(self):
        """
        Return the cleaned dataset.
        """

        pass


if __name__ == "__main__":

    analyzer = (
        CustomerFeatureSelection()
    )

    analyzer.clean_dataset()

    print(
        "Attribute Summary:"
    )

    print(
        analyzer.analyze_attributes()
    )

    analyzer.select_features()

    print(
        "\nSelected Dataset:"
    )

    print(
        analyzer.get_selected_dataset()
    )

    print(
        "\nAttribute Types:"
    )

    print(
        analyzer.analyze_attribute_types()
    )

    print(
        "\nPreprocessing Plan:"
    )

    print(
        analyzer.generate_preprocessing_plan()
    )
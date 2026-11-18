import pandas as pd
import numpy as np


class CustomerFeatureSelection:
    """
    Implements feature selection and
    attribute type analysis to prepare
    customer data for data mining.
    """

    def __init__(self):
        """
        Load the dataset from
        AWCustomers.csv.
        """
        # Load the dataset
        self.df = pd.read_csv("AWCustomers.csv")
        self.selected_df = None

    def clean_dataset(self):
        """
        Clean the dataset by:

        - Removing duplicate rows
        - Filling missing numeric
          values using the median
        - Filling missing categorical
          values using the mode
        """
        self.df = self.df.drop_duplicates()

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val[0])

    def analyze_attributes(self):
        """
        Return a dictionary containing:

        - total_records
        - total_attributes
        - missing_values
        - duplicate_rows
        - data_types
        """
        return {
            "total_records": len(self.df),
            "total_attributes": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "data_types": self.df.dtypes.astype(str).to_dict()
        }

    def select_features(self):
        """
        Select the required attributes
        and store the selected dataset.
        """
        selected_columns = [
            "BirthDate",
            "Education",
            "Occupation",
            "Gender",
            "MaritalStatus",
            "HomeOwnerFlag",
            "NumberCarsOwned",
            "NumberChildrenAtHome",
            "TotalChildren",
            "YearlyIncome",
            "CountryRegionName"
        ]
        self.selected_df = self.df[selected_columns].copy()

    def get_selected_dataset(self):
        """
        Return the selected dataset.
        """
        return self.selected_df

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
            "BirthDate": "Date",
            "Education": "Ordinal",
            "Occupation": "OneHot",
            "Gender": "Label",
            "MaritalStatus": "Label",
            "HomeOwnerFlag": "None",
            "NumberCarsOwned": "Standardize",
            "NumberChildrenAtHome": "Standardize",
            "TotalChildren": "Standardize",
            "YearlyIncome": "Normalize",
            "CountryRegionName": "OneHot"
        }

    def get_dataset(self):
        """
        Return the cleaned dataset.
        """
        return self.df


if __name__ == "__main__":
    analyzer = CustomerFeatureSelection()
    analyzer.clean_dataset()
    print("Attribute Summary:")
    print(analyzer.analyze_attributes())
    analyzer.select_features()
    print("\nSelected Dataset:")
    print(analyzer.get_selected_dataset())
    print("\nAttribute Types:")
    print(analyzer.analyze_attribute_types())
    print("\nPreprocessing Plan:")
    print(analyzer.generate_preprocessing_plan())

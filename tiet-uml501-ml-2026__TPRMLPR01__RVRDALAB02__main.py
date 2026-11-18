import pandas as pd


class EmployeeSalaryTransformationReport:
    """
    Transform employee salary data
    using Pandas.
    """

    def __init__(self):
        """
        Load the dataset and store it
        for further processing.
        """
        # TODO: Implement constructor
        ______

    def apply_salary_normalization(self):
        """
        Apply Min-Max Normalization
        to the Salary column and create
        the NormalizedSalary column.
        """
        # TODO: Implement logic
        ______

    def create_performance_efficiency(self):
        """
        Create the PerformanceEfficiency
        column using the required formula.
        """
        # TODO: Implement logic
        ______

    def get_transformed_data(self):
        """
        Return the transformed dataset.
        """
        # TODO: Implement logic
        return ______

    def generate_transformation_summary(self):
        """
        Generate a summary containing
        transformation details.
        """
        # TODO: Implement logic
        return ______


if __name__ == "__main__":

    report = (
        EmployeeSalaryTransformationReport()
    )

    report.apply_salary_normalization()

    report.create_performance_efficiency()

    print(
        "Transformation Summary:"
    )

    print(
        report.generate_transformation_summary()
    )

    print(
        "\nTransformed Employee Data:"
    )

    print(
        report.get_transformed_data()
    )
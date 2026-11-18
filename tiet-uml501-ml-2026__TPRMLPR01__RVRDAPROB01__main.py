import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EmployeePerformanceEDA:
    """
    Perform Exploratory Data Analysis
    on employee performance data.
    """

    def __init__(self):
        """
        Load the dataset from
        employee_performance_eda.csv.
        """
        # TODO: Implement constructor
        ______

    def generate_descriptive_statistics(self):
        """
        Generate descriptive statistics
        for all numerical columns.

        Store the generated statistics
        for later retrieval.
        """
        # TODO: Implement logic
        return ______

    def calculate_correlation_matrix(self):
        """
        Generate a correlation matrix
        using numerical columns.

        Store the generated correlation
        matrix for later retrieval.
        """
        # TODO: Implement logic
        return ______

    def create_salary_histogram(self):
        """
        Create a histogram for Salary.

        Save the chart as:

        salary_histogram.png

        Store the generated file name
        for later retrieval.
        """
        # TODO: Implement logic
        ______

    def create_experience_vs_salary_scatter_plot(self):
        """
        Create a scatter plot using:

        - ExperienceYears
        - Salary

        Save the chart as:

        experience_salary_scatter.png

        Store the generated file name
        for later retrieval.
        """
        # TODO: Implement logic
        ______

    def create_department_salary_box_plot(self):
        """
        Create a box plot showing salary
        distribution by department.

        Save the chart as:

        department_salary_boxplot.png

        Store the generated file name
        for later retrieval.
        """
        # TODO: Implement logic
        ______

    def get_dataset(self):
        """
        Return the loaded dataset.
        """
        # TODO: Implement logic
        return ______

    def generate_analysis_summary(self):
        """
        Generate a summary containing:

        - Total records
        - Total departments
        - Average salary
        - Average performance score
        """
        # TODO: Implement logic
        return ______


if __name__ == "__main__":

    analysis = (
        EmployeePerformanceEDA()
    )

    statistics = (
        analysis
        .generate_descriptive_statistics()
    )   

    correlation_matrix = (
        analysis
        .calculate_correlation_matrix()
    )

    analysis.create_salary_histogram()

    analysis.create_experience_vs_salary_scatter_plot()

    analysis.create_department_salary_box_plot()

    print(
        "Analysis Summary:"
    )

    print(
        analysis
        .generate_analysis_summary()
    )

    print(
        "\nDescriptive Statistics:"
    )

    print(
        statistics
    )

    print(
        "\nCorrelation Matrix:"
    )

    print(
        correlation_matrix
    )
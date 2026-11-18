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
        self.df = pd.read_csv("employee_performance_eda.csv")
        self.statistics = None
        self.correlation_matrix = None
        self.generated_charts = []

    def generate_descriptive_statistics(self):
        """
        Generate descriptive statistics
        for all numerical columns.

        Store the generated statistics
        for later retrieval.
        """
        self.statistics = self.df.describe()
        return self.statistics

    def calculate_correlation_matrix(self):
        """
        Generate a correlation matrix
        using numerical columns.

        Store the generated correlation
        matrix for later retrieval.
        """
        self.correlation_matrix = self.df.corr(numeric_only=True)
        return self.correlation_matrix

    def create_salary_histogram(self):
        """
        Create a histogram for Salary.

        Save the chart as:

        salary_histogram.png

        Store the generated file name
        for later retrieval.
        """
        filename = "salary_histogram.png"
        plt.figure()
        plt.hist(self.df["Salary"])
        plt.title("Salary Distribution")
        plt.xlabel("Salary")
        plt.ylabel("Frequency")
        plt.savefig(filename)
        plt.close()
        self.generated_charts.append(filename)
        return filename

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
        filename = "experience_salary_scatter.png"
        plt.figure()
        plt.scatter(self.df["ExperienceYears"], self.df["Salary"])
        plt.title("Experience vs Salary")
        plt.xlabel("ExperienceYears")
        plt.ylabel("Salary")
        plt.savefig(filename)
        plt.close()
        self.generated_charts.append(filename)
        return filename

    def create_department_salary_box_plot(self):
        """
        Create a box plot showing salary
        distribution by department.

        Save the chart as:

        department_salary_boxplot.png

        Store the generated file name
        for later retrieval.
        """
        filename = "department_salary_boxplot.png"
        plt.figure()
        sns.boxplot(data=self.df, x="Department", y="Salary")
        plt.title("Salary Distribution by Department")
        plt.savefig(filename)
        plt.close()
        self.generated_charts.append(filename)
        return filename

    def get_dataset(self):
        """
        Return the loaded dataset.
        """
        return self.df

    def generate_analysis_summary(self):
        """
        Generate a summary containing:

        - total records
        - total departments
        - average salary
        - average performance score
        """
        return {
            "total_records": len(self.df),
            "total_departments": int(self.df["Department"].nunique()),
            "average_salary": float(self.df["Salary"].mean()),
            "average_performance_score": float(self.df["PerformanceScore"].mean())
        }


if __name__ == "__main__":

    analysis = EmployeePerformanceEDA()

    statistics = analysis.generate_descriptive_statistics()

    correlation_matrix = analysis.calculate_correlation_matrix()

    analysis.create_salary_histogram()
    analysis.create_experience_vs_salary_scatter_plot()
    analysis.create_department_salary_box_plot()

    print("Analysis Summary:")
    print(analysis.generate_analysis_summary())

    print("\nDescriptive Statistics:")
    print(statistics)

    print("\nCorrelation Matrix:")
    print(correlation_matrix)

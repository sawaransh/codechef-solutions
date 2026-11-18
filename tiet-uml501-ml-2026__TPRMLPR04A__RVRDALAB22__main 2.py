
import pandas as pd
import matplotlib.pyplot as plt


class EmployeeSalaryDistributionVisualization:
    """
    Visualize salary distributions using histogram and density plots.
    """

    def __init__(self):
        """Load the dataset from employee_salaries.csv."""
        self.data = pd.read_csv("employee_salaries.csv")

        self.histogram_filename = "salary_histogram.png"
        self.density_filename = "salary_density_plot.png"

    def create_histogram(self):
        """Create and save a histogram using the Salary column."""

        plt.figure(figsize=(8, 5))

        plt.hist(
            self.data["Salary"].dropna(),
            bins=10,
            edgecolor="black"
        )

        plt.title("Employee Salary Distribution")
        plt.xlabel("Salary")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(self.histogram_filename)
        plt.close()

        return self.histogram_filename

    def create_density_plot(self):
        """Create and save a density plot using the Salary column."""

        plt.figure(figsize=(8, 5))

        self.data["Salary"].dropna().plot(
            kind="density"
        )

        plt.title("Employee Salary Density Plot")
        plt.xlabel("Salary")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.savefig(self.density_filename)
        plt.close()

        return self.density_filename

    def get_dataset(self):
        """Return the loaded dataset."""
        return self.data

    def get_average_salary(self):
        """Return the average salary."""
        return self.data["Salary"].mean()

    def get_median_salary(self):
        """Return the median salary."""
        return self.data["Salary"].median()

    def generate_visualization_summary(self):
        """Generate a summary of salary distribution."""

        summary = {
            "total_employees": len(self.data),
            "average_salary": self.get_average_salary(),
            "median_salary": self.get_median_salary()
        }

        return summary


if __name__ == "__main__":

    visualization = (
        EmployeeSalaryDistributionVisualization()
    )

    visualization.create_histogram()

    visualization.create_density_plot()

    print(
        "\nVisualization Summary:"
    )

    print(
        visualization.generate_visualization_summary()
    )

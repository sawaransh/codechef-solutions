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
        self.df = pd.read_csv("employee_salary_data.csv")

    def apply_salary_normalization(self):
        """
        Apply Min-Max Normalization
        to the Salary column and create
        the NormalizedSalary column.
        """
        min_sal = self.df["Salary"].min()
        max_sal = self.df["Salary"].max()
        self.df["NormalizedSalary"] = (self.df["Salary"] - min_sal) / (max_sal - min_sal)

    def create_performance_efficiency(self):
        """
        Create the PerformanceEfficiency
        column using the required formula.
        """
        score_col = next(c for c in self.df.columns if "performance" in c.lower() and "score" in c.lower())
        self.df["PerformanceEfficiency"] = self.df[score_col] / self.df["Salary"]

    def get_transformed_data(self):
        """
        Return the transformed dataset.
        """
        return self.df

    def generate_transformation_summary(self):
        """
        Generate a summary containing
        transformation details.
        """
        return {
            "total_employees": len(self.df),
            "min_normalized_salary": float(self.df["NormalizedSalary"].min()),
            "max_normalized_salary": float(self.df["NormalizedSalary"].max())
        }


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

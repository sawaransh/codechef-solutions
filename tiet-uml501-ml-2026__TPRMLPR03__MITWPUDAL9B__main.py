import os
import pandas as pd
from scipy import stats


class EmployeeSalaryDistributionReport:
    """
    Generate salary distribution
    statistics using SciPy.
    """

    def __init__(self):
        """
        Load the dataset and store it
        for further processing.
        """
        csv_path = "employee_salaries.csv"
        if not os.path.exists(csv_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "employee_salaries.csv")

        self.df = pd.read_csv(csv_path)

    def calculate_salary_mode(self):
        """
        Calculate the mode
        of employee salaries.
        """
        mode_result = stats.mode(self.df["Salary"])
        mode_val = (
            mode_result.mode.flat[0]
            if hasattr(mode_result.mode, "flat")
            else mode_result.mode
        )
        return mode_val.item() if hasattr(mode_val, "item") else mode_val

    def calculate_salary_skewness(self):
        """
        Calculate the skewness
        of employee salaries.
        """
        skew_val = stats.skew(self.df["Salary"])
        return float(skew_val)

    def calculate_salary_kurtosis(self):
        """
        Calculate the kurtosis
        of employee salaries.
        """
        kurtosis_val = stats.kurtosis(self.df["Salary"])
        return float(kurtosis_val)

    def generate_distribution_report(self):
        """
        Generate a distribution report
        containing all calculated metrics.
        """
        return {
            "salary_mode": self.calculate_salary_mode(),
            "salary_skewness": self.calculate_salary_skewness(),
            "salary_kurtosis": self.calculate_salary_kurtosis(),
        }


if __name__ == "__main__":

    report = (
        EmployeeSalaryDistributionReport()
    )

    print(
        "Employee Salary Distribution Report:"
    )

    print(
        report.generate_distribution_report()
    )
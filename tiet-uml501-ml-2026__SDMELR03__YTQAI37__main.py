import _________ as pd   # TODO: Import pandas library
import matplotlib.pyplot as plt

def analyze_salaries(_________):   # TODO: Add function parameter (file path)

    # Step 1: Load csv file
    df = ____________   # TODO: Load the CSV file using pandas
    print("\nFirst 5 rows of the dataset:\n")
    print(______________)   # TODO: Print the first 5 rows

    # Step 2: Visualize with outliers
    plt.boxplot(df["Salary"])   
    plt.title("Salary Distribution (with Outliers)")
    plt.savefig("salary_outliers.png")   
    plt.close()
    print("\nBoxplot saved as 'salary_outliers.png'")

    # Step 3: Remove outliers (keep only 30000–100000)
    # TODO: Filter dataset to keep only rows where Salary is between 30000 and 100000
    cleaned_df = ______________________   
    print("\nDataset after removing outliers:\n")
    print(cleaned_df)

    # TODO return the cleaned DataFrame
    _______________


def main():
    # TODO: Call the function with the dataset path
    analyze_salaries("employee_salaries.csv")   

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt

def analyze_data(filePath):

    # Step 1: Load csv file
    df = pd.read_csv(filePath)
    print("\nFirst 5 rows of the dataset:\n")
    print(df.head())

    # Step 2: Visualize with outliers
    plt.boxplot(df["Hours_Worked"])
    plt.title("Working Hours Distribution (with Outliers)")
    plt.savefig("hours_outliers.png")
    plt.close()
    print("\nBoxplot saved as 'hours_outliers.png'")

    # Step 3: Remove outliers (keep only 30-60)
    cleaned_df = df[(df["Hours_Worked"] >= 30) & (df["Hours_Worked"] <= 60)]
    print("\nDataset after removing outliers:\n")
    print(cleaned_df)

def main():
    analyze_data("working_hours.csv")

if __name__ == "__main__":
    main()

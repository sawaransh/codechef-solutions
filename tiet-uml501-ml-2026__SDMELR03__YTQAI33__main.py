import pandas as pd

def handle_missing_values(file_path):
    """
    Load the dataset, handle missing values, round numerical columns, 
    and display first 5 rows and final dataset.
    """
    # Step 1: Load dataset
    df = pd.read_csv(file_path)
    
    # Step 2: Display the first 5 rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Step 3: Check missing values
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    
    # Step 4: Fill missing numerical values with mean
    df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].mean())
    df['English_Score'] = df['English_Score'].fillna(df['English_Score'].mean())
    df['Science_Score'] = df['Science_Score'].fillna(df['Science_Score'].mean())
    
    # Step 5: Fill missing categorical values with mode
    df['Class'] = df['Class'].fillna(df['Class'].mode()[0])
    
    # Step 6: Round numerical columns to 1 decimal place
    df[['Math_Score', 'English_Score', 'Science_Score']] = df[['Math_Score', 'English_Score', 'Science_Score']].round(1)
    
    # Step 7: Display dataset after handling missing values
    print("\nDataset after Handling Missing Values (rounded to 1 decimal):")
    print(df)
    
    return df  # return the DataFrame if needed later

# Call the function in main
if __name__ == "__main__":
    handle_missing_values("students.csv")

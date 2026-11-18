import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # TODO: read the csv into a DataFrame
    df = pd.read_csv(file_path)
    return df


def explore_data(df: pd.DataFrame):
    """
    Perform basic data exploration, printing key details about the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to explore.
    """
    # TODO: Print dataset shape
    print("Dataset Shape:", df.shape)
    # TODO: Print data types of each column
    print("\nData Types:\n", df.dtypes)
    # TODO: Print number of missing values
    print("\nMissing Values:\n", df.isnull().sum())
    # TODO: Print value counts for key categorical columns (e.g., 'Gender', 'Course', 'Tuition fees up to date', 'Debtor' etc.)
    print("\nGender Distribution:\n", df["Gender"].value_counts())
    print("\nCourse Distribution:\n", df["Course"].value_counts())
    print("\nTuition Fees Up to Date Distribution:\n", df["Tuition fees up to date"].value_counts())
    print("\nDebtor Distribution:\n", df["Debtor"].value_counts())


def create_approval_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 'approval_rate' feature based on 'Curricular units 1st sem (approved)' and 'Curricular units 1st sem (enrolled)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'approval_rate' column.
    """
    # TODO: Create 'approval_rate' column by dividing 'Curricular units 1st sem (approved)' by 'Curricular units 1st sem (enrolled)'
    df["approval_rate"] = df["Curricular units 1st sem (approved)"] / df["Curricular units 1st sem (enrolled)"]
    return df


def create_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 'performance_score' feature based on 'Curricular units 1st sem (approved)' and 'Curricular units 1st sem (evaluations)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'performance_score' column.
    """
    # TODO: Create 'performance_score' column by dividing 'Curricular units 1st sem (approved)' by 'Curricular units 1st sem (evaluations)'
    df["performance_score"] = df["Curricular units 1st sem (approved)"] / df["Curricular units 1st sem (evaluations)"]
    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features needed for the analysis.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # TODO: Add engineered features by calling the functions to create 'approval_rate' and 'performance_score'
    df = create_approval_rate(df)
    df = create_performance_score(df)

    # TODO: Print the first few rows of the DataFrame to verify that the new columns are correctly added ('Course', 'Gender', 'approval_rate', 'performance_score')
    print("\nData with new columns:\n", df[["Course", "Gender", "approval_rate", "performance_score"]].head())

    return df


# --- Main execution block ---
if __name__ == "__main__":
    # Path to the dataset
    file_path = "dataset.csv"

    df = load_data(file_path)

    explore_data(df)

    df = create_engineered_features(df)

import pandas as pd
import numpy as np


def create_target_variable(df):
    """
    Creates the binary target variable 'is_cancelled' and filters the DataFrame.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the new 'is_cancelled' column and
                      filtered for relevant booking statuses.
    """
    df_copy = df.copy()
    cancelled_statuses = ['Cancelled by Customer', 'Cancelled by Driver', 'No Driver Found']

    df_copy['is_cancelled'] = np.where(
        df_copy['Booking Status'].isin(cancelled_statuses),
        1,
        np.where(df_copy['Booking Status'] == 'Completed', 0, -1)
    )

    filtered_df = df_copy[df_copy['is_cancelled'].isin([0, 1])].copy()
    return filtered_df


def feature_engineering(df):
    """
    Creates new features from existing columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the new 'hour_of_day' feature.
    """
    df_copy = df.copy()
    df_copy['hour_of_day'] = pd.to_datetime(df_copy['Time']).dt.hour
    return df_copy


def drop_unnecessary_columns(df):
    """
    Drops columns that are not needed for modeling.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with unnecessary columns removed.
    """
    cols_to_drop = [
        # Leakage Columns
        'Booking Status',
        'Cancelled Rides by Customer',
        'Reason for cancelling by Customer',
        'Cancelled Rides by Driver',
        'Driver Cancellation Reason',
        'Incomplete Rides',
        'Incomplete Rides Reason',
        # Post-Ride Info
        'Ride Distance',
        'Booking Value',
        'Driver Ratings',
        'Customer Rating',
        # High Cardinality/ID Columns
        'Booking ID',
        'Customer ID',
        # Redundant Columns
        'Date',
        'Time'
    ]

    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=existing_cols_to_drop)


def clean_and_save_data(input_filepath, output_filepath):
    """
    Orchestrates the entire data cleaning pipeline and saves the result.

    Args:
        input_filepath (str): The path to the raw CSV file.
        output_filepath (str): The path where the cleaned CSV will be saved.
    """
    print("Starting data cleaning process...")

    # Load the raw data
    raw_df = pd.read_csv(input_filepath)
    print(f"Loaded {len(raw_df)} rows from {input_filepath}")

    # Replace all occurrences of the string 'null' with np.nan
    raw_df = raw_df.replace('null', np.nan)

    # Step 1: Create the target variable
    df_with_target = create_target_variable(raw_df)
    print(f"Step 1: Created target variable. Rows remaining: {len(df_with_target)}")

    # Step 2: Engineer new features
    df_featured = feature_engineering(df_with_target)
    print("Step 2: Engineered 'hour_of_day' feature.")

    # Step 3: Handle missing data by imputing
    print("Step 3: Imputing missing values...")
    df_imputed = df_featured.copy()  # Work on a copy

    # Fill missing values in 'Avg VTAT' and 'Avg CTAT' with 0
    df_imputed['Avg VTAT'] = df_imputed['Avg VTAT'].fillna(0)
    df_imputed['Avg CTAT'] = df_imputed['Avg CTAT'].fillna(0)

    # After filling, convert 'Avg VTAT' and 'Avg CTAT' to a numeric type
    df_imputed['Avg VTAT'] = pd.to_numeric(df_imputed['Avg VTAT'])
    df_imputed['Avg CTAT'] = pd.to_numeric(df_imputed['Avg CTAT'])

    print(f"Imputation complete. Rows remaining: {len(df_imputed)}")

    # Step 4: Drop unnecessary columns
    df_cleaned = drop_unnecessary_columns(df_imputed)  # Use the imputed DataFrame
    print("Step 4: Dropped unnecessary columns.")
    print(f"Final columns: {df_cleaned.columns.tolist()}")

    # Final Step: Save the cleaned data
    df_cleaned.to_csv(output_filepath, index=False)

    print(f"Data cleaning complete. Cleaned data saved to {output_filepath}")


if __name__ == "__main__":
    RAW_DATA_FILE = 'uber_rides_2024.csv'
    CLEANED_DATA_FILE = 'clean_uber_rides_2024.csv'
    clean_and_save_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
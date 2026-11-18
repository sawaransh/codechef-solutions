import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def prepare_data(df):
    """
    Prepares the data for modeling by separating features and target,
    and applying one-hot encoding to categorical features.
    """
    # Define the target 'y' by selecting the 'is_cancelled' column.
    y = df['is_cancelled']
    
    # Define the features 'X' by dropping the 'is_cancelled' column.
    X = df.drop(columns=['is_cancelled'])
    
    # Use pd.get_dummies() on X to perform one-hot encoding. Set drop_first=True.
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Return the encoded features and the target.
    return X_encoded, y


def split_data(X, y):
    """
    Splits the data into training and testing sets using stratification.
    """
    # Use train_test_split with test_size=0.2, random_state=42, and stratify=y.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Return the four resulting datasets.
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a Decision Tree Classifier model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.

    Returns:
        DecisionTreeClassifier: The trained model object.
    """
    # Instantiate a DecisionTreeClassifier. Set random_state=42.
    model = DecisionTreeClassifier(random_state=42)
    
    # Fit the model to the training data.
    model.fit(X_train, y_train)
    
    # Return the trained model.
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set.
    """
    # Use the model to make predictions on X_test.
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy score by comparing predictions to y_test.
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the accuracy.
    return accuracy


def create_prediction_pipeline(filepath):
    """
    Orchestrates the full pipeline from loading data to returning a trained model.
    """
    print("Starting model training pipeline...")

    # Load the cleaned data from the given filepath.
    df = pd.read_csv(filepath)

    # Call prepare_data() to get features and target.
    X, y = prepare_data(df)

    # Call split_data() to get training and testing sets.
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Call train_model() to get a trained model.
    model = train_model(X_train, y_train)

    # Call evaluate_model() to get the model's accuracy.
    accuracy = evaluate_model(model, X_test, y_test)

    # Update this print statement with the calculated accuracy.
    print(f"Model evaluation complete. Accuracy: {accuracy:.4f}")

    # Return the trained model and the list of columns from the prepared feature set.
    return model, list(X.columns)


if __name__ == "__main__":
    CLEANED_DATA_FILE = 'clean_uber_rides_2024.csv'

    trained_model, model_columns = create_prediction_pipeline(CLEANED_DATA_FILE)

    print("\n--- Prediction Example ---")

    if trained_model and model_columns:
        sample_ride = {
            'Vehicle Type': 'Go Sedan',
            'Pickup Location': 'Khandsa',
            'Drop Location': 'Malviya Nagar',
            'Avg VTAT': 13.4,
            'Avg CTAT': 25.8,
            'Payment Method': 'UPI',
            'hour_of_day': 8
        }

        sample_df = pd.DataFrame([sample_ride])
        sample_encoded = pd.get_dummies(sample_df)

        sample_aligned = sample_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = trained_model.predict(sample_aligned)
        prediction_proba = trained_model.predict_proba(sample_aligned)

        result = "Cancelled" if prediction[0] == 1 else "Completed"
        print(f"Sample ride data: {sample_ride}")
        print(f"Predicted outcome: {result}")
        print(f"Prediction probability (0=Completed, 1=Cancelled): {prediction_proba[0]}")
    else:
        print("Pipeline did not return a trained model.")
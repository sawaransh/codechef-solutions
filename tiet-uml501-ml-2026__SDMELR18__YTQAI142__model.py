import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# --- Ensemble Learning & Model Optimization Functions ---

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier.
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate the model using accuracy, precision, and recall.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall

# --- Feature Selection, Handling Missing Values, Encoding Categorical Features, and Data Splitting from the previous part ---

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features for the analysis.
    """
    features = ['Age at enrollment', 'Gender', 'Debtor', 'Tuition fees up to date', 'Curricular units 1st sem (approved)']
    X = df[features]
    return X

def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        X (pd.DataFrame): The input feature matrix.

    Returns:
        pd.DataFrame: The feature matrix with missing values handled.
    """
    X = X.copy()
    
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X.loc[:, categorical_cols] = X[categorical_cols].fillna('Unknown')
    
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    X.loc[:, numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
    
    return X

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into numeric format using one-hot encoding.
    """
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# --- Main Execution Block ---
if __name__ == '__main__':
    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)
    
    features = select_features(df)
    
    features = features.copy()
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    features.loc[:, categorical_cols] = features[categorical_cols].fillna('Unknown')
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    features.loc[:, numerical_cols] = features[numerical_cols].fillna(features[numerical_cols].median())
    
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
    
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
    
    rf_model = train_random_forest(X_train, y_train)
    
    rf_accuracy, rf_precision, rf_recall = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Model Accuracy:", rf_accuracy)
    print("Random Forest Model Precision:", rf_precision)
    print("Random Forest Model Recall:", rf_recall)
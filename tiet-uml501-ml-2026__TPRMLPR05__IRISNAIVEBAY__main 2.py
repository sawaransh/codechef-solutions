import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
def load_dataset(path):
    # Must return a pandas DataFrame
    # Dataset must contain at least 150 rows
    # Dataset must contain a column named "variety"
    # Dataset must be read using pandas
    df = pd.read_csv(path)
    return df

# --------------------------------------------------
# 2. Feature Selection & Target Encoding
# --------------------------------------------------
def select_features(df):
    # Must select only feature columns (exclude target)
    # Target column must be "variety"
    # X must be a NumPy array
    # y must be a NumPy array
    # X must have exactly 4 columns
    # y must contain more than 2 unique classes
    # No NaN values allowed in X or y
    # X must be numeric
    # Target must be encoded using LabelEncoder
    # Encoded labels must be integers
    df_clean = df.dropna()
    X = df_clean.drop(columns=["variety"]).to_numpy(dtype=float)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_clean["variety"]).astype(int)
    
    return X, y_encoded

# --------------------------------------------------
# 3. Split Data
# --------------------------------------------------
def split_data(X, y, test_size=0.2):
    # Must use train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test

# --------------------------------------------------
# 4. Train Model
# --------------------------------------------------
def train_model(X_train, y_train):
    # Must use GaussianNB
    # Model must be fitted
    # Model must have learned parameters
    # model.theta_ must exist
    # model.sigma_ must exist
    # model.sigma_ must not be all zeros
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Alias var_ to sigma_ to satisfy test expectations
    # This is required because newer sklearn uses var_
    model.sigma_ = model.var_

    return model

# --------------------------------------------------
# 5. Predict
# --------------------------------------------------
def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

# --------------------------------------------------
# 6. Evaluate Model
# --------------------------------------------------
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# --------------------------------------------------
# 7. Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    df = load_dataset("dataset.csv")
    X, y = select_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)
    accuracy = evaluate_model(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
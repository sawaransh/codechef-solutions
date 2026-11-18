import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Step 1: Load dataset
def load_data(path):
  # Read the CSV file using pandas
  df = pd.read_csv(path)

  # Select features (Hours_Studied should be a DataFrame, not Series)
  X = df[["Hours_Studied"]]

  # Select target column (Exam_Pass)
  y = df["Exam_Pass"]

  return X, y


# Step 2: Train logistic regression
def train_model(X, y):
  # Split the dataset into training and testing (30% test, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42
  )

  # Create logistic regression model
  model = LogisticRegression()

  # Train the model
  model.fit(X_train, y_train)

  return model, X_test, y_test


# Step 3: Evaluate predictions
def evaluate_model(model, X_test, y_test):
  # Generate predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate Accuracy
  accuracy = accuracy_score(y_test, y_pred)

  # Calculate Precision
  precision = precision_score(y_test, y_pred)

  # Calculate Recall
  recall = recall_score(y_test, y_pred)

  # Calculate F1 Score
  f1 = f1_score(y_test, y_pred)

  return accuracy, precision, recall, f1


# Step 4: Display results
def display_results(model, accuracy, precision, recall, f1):
  print("Model used:", type(model).__name__)

  print("Accuracy:", accuracy)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1-score:", f1)


# Step 5: Main workflow
if __name__ == "__main__":
  # Define dataset path
  path = "students.csv"

  # Load dataset
  X, y = load_data(path)

  # Train logistic regression
  model, X_test, y_test = train_model(X, y)

  # Evaluate metrics
  accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

  # Print results
  display_results(model, accuracy, precision, recall, f1)
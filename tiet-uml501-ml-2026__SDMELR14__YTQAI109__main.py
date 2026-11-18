# TODO: Import pandas for handling CSV/dataframes
import pandas as pd

# TODO: Import DecisionTreeClassifier from sklearn
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load dataset from CSV
# TODO: Fill in the CSV filename (e.g., "student_pass.csv")
df = pd.read_csv("student_pass.csv")

# Step 2: Separate features and label
# TODO: Select the feature columns "Study_Hours" and "Attendance"
X = df[["Study_Hours", "Attendance"]]  # Features

# TODO: Select the target column "Pass"
y = df["Pass"]  # Target label

# Step 3: Create and train the Decision Tree model
# TODO: Initialize DecisionTreeClassifier with random_state=42
model = DecisionTreeClassifier(random_state=42)

# TODO: Fit the model on the features and label
model.fit(X, y)

# Step 4: Make predictions on the training data
# TODO: Generate predictions
predictions = model.predict(X)

# Step 5: Print the predictions
print("Predictions:", predictions.tolist())
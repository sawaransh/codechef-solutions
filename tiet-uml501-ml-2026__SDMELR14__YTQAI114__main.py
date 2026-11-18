# TODO: Import pandas, numpy and RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load dataset
# TODO: Load the CSV file "student_performance.csv"
df = pd.read_csv("student_performance.csv")

# Step 2: Select feature columns
# TODO: Replace with the appropriate feature column names
X = df[["Study_Hours", "Attendance"]]

# Step 3: Select target column
# TODO: Replace with the target column name
y = df["Pass"]

# Step 4: Train a RandomForestClassifier
# TODO: Set n_estimators and random_state
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X, y)

# Step 5: Make predictions
predictions = model.predict(X)

# Step 6: Predict probabilities
probabilities = model.predict_proba(X)

# Step 7: Round probabilities
probabilities = np.round(probabilities, 1)

# Step 8: Print predicted classes
print("Predictions:", predictions.tolist())

# Step 9: Print probabilities with comments
for row in probabilities:
    fail, pass_ = row
    print(f"({fail:.1f}) {int(fail*100)}% Fail, {int(pass_*100)}% Pass")
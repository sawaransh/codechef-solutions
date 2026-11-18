# Import pandas for handling CSV/dataframes
import pandas as pd

# Import DecisionTreeClassifier from sklearn
from sklearn.tree import DecisionTreeClassifier

# Step 1: Create the dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6],       # Hours studied
    "Attendance": [60, 70, 80, 85, 90],   # Attendance percentages
    "Pass": [0, 0, 1, 1, 1]                # Target labels (0 = Fail, 1 = Pass)
}

# TODO: Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Separate features and label
# TODO: Select Study_Hours and Attendance as features (X)
X = df[["Study_Hours", "Attendance"]]

# TODO: Select Pass as the target label (y)
y = df["Pass"]

# Step 3: Train the Decision Tree model
# TODO: Create a DecisionTreeClassifier with random_state=42
model = DecisionTreeClassifier(random_state=42)

# TODO: Fit the model on the training data
model.fit(X, y)

# Step 4: Make predictions on the training data
# TODO: Generate predictions
predictions = model.predict(X)

# TODO: Print the predictions as a list
print("Predictions:", predictions.tolist())
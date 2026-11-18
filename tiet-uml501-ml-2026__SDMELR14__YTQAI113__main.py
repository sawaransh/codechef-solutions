import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# TODO: You may change feature names and values if you want
data = {
    "Hours_Studied": [2, 4, 7, 8, 10],
    "Absences": [8, 6, 2, 1, 0],
    "Result": [0, 0, 1, 1, 1]
}

# TODO: Convert your dataset into a DataFrame
df = pd.DataFrame(data)

# TODO: Select the feature columns
X = df[["Hours_Studied", "Absences"]]

# TODO: Select the target column
y = df["Result"]

# TODO: Train a RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,  # Add number of decision trees (e.g 100)
    max_depth=3,
    min_samples_leaf=2,
    random_state=42     # fixes randomness, so you get the same results every time (e.g 42).
)
model.fit(X, y)

# TODO: Make predictions
predictions = model.predict(X)

# Predict probabilities
probabilities = model.predict_proba(X)

# Round to 1 decimal
probabilities = np.round(probabilities, 1)

# Print the Predicted Classes
print("Predictions:", predictions.tolist())
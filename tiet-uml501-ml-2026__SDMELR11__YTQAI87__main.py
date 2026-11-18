import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Sleep_Hours": [3, 4, 5, 6, 7, 8],
    "Alert": [0, 0, 0, 1, 1, 1]
}

# TODO: Create DataFrame from dictionary
df = pd.DataFrame(data)

# TODO: Select features (X) and labels (y)
X = df[["Sleep_Hours"]]   # Feature
y = df["Alert"]           # Label

# TODO: Create Logistic Regression model
model = LogisticRegression()
model.fit(X, y)           # Train the model

# TODO: Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)

# TODO: Get probability estimates
probabilities = model.predict_proba(X)
print("Probabilities:\n", probabilities)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Create a dataset
data = {
    "Sleep_Hours": [4, 5, 6, 7, 8],
    "Productivity_Score": [50, 60, 65, 75, 85]
}
df = pd.DataFrame(data)

# Features (X) and Labels (y)
X = df[["Sleep_Hours"]]        # Fill the feature column name
y = df["Productivity_Score"]   # Fill the label column name

# Step 2: Train Linear Regression model
model = LinearRegression()     # Fill the model class
model.fit(X, y)                # Train the model

# Step 3: Make predictions
y_pred = model.predict(X)      # Make predictions

# Step 4: Calculate R² score
r2 = r2_score(y, y_pred)       # Fill the evaluation function

print("Actual Scores:", list(y))
print("Predicted Scores:", list(y_pred.round(2)))
print("R² Score:", round(r2, 2))
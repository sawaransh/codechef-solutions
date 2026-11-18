import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the dataset
df = pd.read_csv("students.csv")

# Step 2: Select numeric feature(s) to scale (Age and Marks)
X = df[["Age", "Marks"]]

# Step 3: Apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Step 5: Print the scaled values
print("Scaled Data:")
print(X_scaled_df)

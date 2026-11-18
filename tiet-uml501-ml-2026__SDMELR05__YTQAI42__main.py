import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# TODO: Read the CSV file "employees.csv"
df = ___._______________(_____________)

# Step 2: Select numeric feature(s) to scale (Salary and Experience)
# TODO: Select Salary and Experience columns
X = df[___________________]

# Step 3: Apply Standard Scaling
# TODO: Create a StandardScaler object
scaler = ______________

# TODO: Fit and transform the data using scaler
X_scaled = scaler._____________(X)

# Step 4: Convert back to DataFrame
# TODO: Convert the scaled NumPy array back to a DataFrame
X_scaled_df = ___.___________(X_scaled, columns=["Salary", "Experience"])

# Step 5: Print the scaled values
print("Standardized Data:")
print(X_scaled_df)

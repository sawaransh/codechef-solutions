
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("employees.csv")

# 2. Display first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# 3. Separate features (X) and label (y)
X = df[['Age', 'Department', 'Salary']]
y = df['Left_Company']

# Display features
print("\nFeatures (X):")
print(X)

# Display label
print("\nLabel (y):")
print(y)

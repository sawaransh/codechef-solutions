import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
# TODO: Load the dataset "students.csv"
df = _________
print("Original Data:")
print(df)

# Step 2: One-Hot Encoding (City)
# TODO: Apply One-Hot Encoding on the "City" column
df_onehot = ______________
print("\nAfter One-Hot Encoding City:")
print(df_onehot)

# Step 3: Label Encoding (Grade)
# TODO: Use LabelEncoder to convert ordered categories in "Grade"
encoder = __________
df["Grade_Label"] = ________________
print("\nAfter Label Encoding Grade:")
print(df[["Name", "Grade", "Grade_Label"]])

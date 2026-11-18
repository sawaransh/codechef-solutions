import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = _________
print("Original Data:")
print(df)

# Step 2: Label Encoding (Subscribed)
label_encoder = _________
# Convert the "Subscribed" column into numeric values using LabelEncoder
df["Subscribed_Label"] = label_encoder._________
print("\nAfter Label Encoding Subscribed:")
print(df[["Name", "Subscribed", "Subscribed_Label"]])

# Step 3: One-Hot Encoding (Region)
# Convert the "Region" column into multiple binary columns
df_onehot = ______________

# Step 4: Display results
print("\nAfter One-Hot Encoding Region:")
print(df_onehot)

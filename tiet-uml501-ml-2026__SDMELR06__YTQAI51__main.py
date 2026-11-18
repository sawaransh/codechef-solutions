import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load Dataset
df = ________________   # <-- Fill: load employees.csv
print("First 5 rows:\n", ________________)

# Step 2: Features & Label
X = df.drop("Promotion_Eligibility", axis=1)  # Features
y = df["Promotion_Eligibility"]               # Label

print("\nFeatures:\n", ________________)       # <-- Fill
print("\nLabel:\n", ________________)          # <-- Fill

# Step 3: Encoding Categorical Data
# Label Encoding: Gender
df["Gender_Label"] = df["Gender"].map({_____: _____, _____: _____})  # <-- Fill: Male/0, Female/1
print("\nAfter Label Encoding Gender:\n", df[["Name","Gender","Gender_Label"]].head())

# One-Hot Encoding: Department
df_onehot = pd.get_dummies(df, columns=[_____])  # <-- Fill: column to one-hot encode
print("\nAfter One-Hot Encoding Department:\n", df_onehot.head())

# Step 4: Feature Scaling
# Normalization
minmax = MinMaxScaler()
X_normalized = ________________                 # <-- Fill: transform X numeric columns
print("\nAfter Normalization:\n", X_normalized[:5])

# Standardization
scaler = StandardScaler()
X_standardized = ________________               # <-- Fill: transform X numeric columns
print("\nAfter Standardization:\n", X_standardized[:5])

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = ________________   # <-- Fill: use train_test_split
print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

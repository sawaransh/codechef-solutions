import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = _________________  # <-- Fill: load CSV file 
print("First 5 rows:\n", df.head())

# -----------------------------
# Step 2: Identify Features & Label and Print features columns and label name
# -----------------------------
features = df.drop("Promotion_Eligibility", axis=1)  
label = df["Promotion_Eligibility"]  

print("\nFeatures:\n", ________________)  # <-- Fill: print feature columns 
print("Label:\n", _____________________)   # <-- Fill: print label name 

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
print("\nMissing Values Before:\n", df.________________)  # <-- Fill: check missing values 

# Fill numerical missing values with median
df["Age"] = df["Age"]._____  # <-- Fill: handle missing Age 
df["Years_Experience"] = df["Years_Experience"]._____  # <-- Fill: handle missing Years_Experience 

# Fill categorical missing values with mode
df["Department"] = df["Department"]._____  # <-- Fill: handle missing Department 
df["Promotion_Eligibility"] = df["Promotion_Eligibility"]._____  # <-- Fill: handle missing Promotion_Eligibility 

print("\nMissing Values After:\n", df.isnull().sum())

# -----------------------------
# Step 4: Detect & Handle Outliers (Simple Method)
# -----------------------------
# Identify Salary outliers (e.g., Salary < 30000 or Salary > 150000)
salary_outliers = df[(df["Salary"] < _____) | (df["Salary"] > _____)]  # <-- Fill: set low and high threshold values
print("\nSalary Outliers:\n", salary_outliers)

# Remove Salary outliers 
df = df[__________ & ___________]

# Handle Years_Experience outliers (remove > 30 years)
df = df[___________ <= ___]


# -----------------------------
# Step 5: Handling Outliers
# -----------------------------
print("\nDescriptive Statistics:\n", __________________)  # <-- Fill: get descriptive stats 
print("\nEmployees per Department:\n", df["Department"].________________)  # <-- Fill: count employees 

plt.figure(figsize=(6,4))
plt.boxplot(_______________)  # <-- Fill: column to visualize "Salary"
plt.title("Salary Distribution After Handling Outliers")
plt.ylabel(_______________)   # <-- Fill: y-axis label ("Salary")
plt.savefig("salary_boxplot.png")  # Save the image
plt.show()


from sklearn.metrics import mean_absolute_error

# Actual vs Predicted values
y_true = [30, 45, 50, 65, 80]
y_pred = [28, 50, 47, 60, 85]

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)

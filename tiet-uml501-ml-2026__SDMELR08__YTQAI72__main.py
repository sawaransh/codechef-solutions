import numpy as np
from sklearn.metrics import mean_squared_error

# Step 1: Actual and Predicted Scores
y_true = [72, 88, 95, 60]    # Actual values
y_pred = [70, 90, 92, 65]    # Predicted values

# Step 2: Calculate MSE
mse = mean_squared_error(y_true, y_pred)    # <-- Calculate MSE here using sklearn
rmse = np.sqrt(mse)    # <-- Calculate RMSE

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
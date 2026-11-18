# TODO: Import the required function
from sklearn.metrics import f1_score # TODO

# Actual labels (ground truth)
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1] # TODO: Fill the actual outcomes

# Predictions from our model
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1] # TODO: Fill the predicted outcomes

# Calculate F1 score
f1 = f1_score(y_true, y_pred) # TODO: Use the correct function

# Print result
print("F1-score:", f1) # TODO: Ensure correct print format
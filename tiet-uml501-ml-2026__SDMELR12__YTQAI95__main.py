from sklearn.metrics import accuracy_score

# Step 1: Define the actual labels (ground truth)
# TODO: Fill in the actual labels
y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]

# Step 2: Define the predictions from the model
# TODO: Fill in the predicted labels
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]

# Step 3: Calculate accuracy
# TODO: Call accuracy_score with y_true and y_pred
acc = accuracy_score(y_true, y_pred)

# Step 4: Print the accuracy
# TODO: Complete the print statement
print("Accuracy:", acc)
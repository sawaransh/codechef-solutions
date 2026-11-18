import numpy as np

# Step 1: Define the sigmoid function
def sigmoid(x):
    # TODO: Complete the formula for sigmoid
    return 1 / (1 + np.exp(-x))

# Step 2: Example raw scores (like the linear step inside Logistic Regression)
scores = np.array([-3.0, -0.5, 0.5, 2.5, 4.0])

# Step 3: Convert scores to probabilities using sigmoid
# TODO: Call the sigmoid function on scores
probabilities = sigmoid(scores)

# Step 4: Apply threshold (0.5) to get final predictions
# TODO: Convert probabilities into 0 or 1 based on threshold
predictions = (probabilities >= 0.5).astype(int)

# Display results
for s, p, pred in zip(scores, probabilities, predictions):
    # TODO: Complete the print statement
    print(f"Score: {s:>4} -> Probability: {p:.2f} -> Prediction: {pred}")
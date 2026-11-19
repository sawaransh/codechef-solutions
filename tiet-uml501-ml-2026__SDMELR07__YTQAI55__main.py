from sklearn.linear_model import LinearRegression

# Step 1: Input data
hours = [[1], [2], [3], [4]]    # Features
scores = [42, 48, 60, 68]       # Labels

# Step 2: Create the model
# TODO: complete this line
model = LinearRegression()

# Step 3: Train the model
# TODO: complete this line
model.fit(hours, scores)

# Step 4: Make prediction
predicted_score = model.predict([[5]])
print(predicted_score) # What do you think it will print?
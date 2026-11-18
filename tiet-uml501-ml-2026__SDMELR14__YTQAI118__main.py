import pandas as pd

# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Dataset
data = {
    "Message": [
        "Exclusive offer just for you",
        "Let's meet tomorrow morning",
        "Congratulations you won a prize",
        "Project review meeting at 5pm",
        "Free entry in a lucky draw"
    ],
    "Spam": [1, 0, 1, 0, 1]
}

# TODO: Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Step 2: Convert Text into Features
# TODO: Initialize CountVectorizer
vectorizer = CountVectorizer()

# TODO: Convert text messages into numeric features
X = vectorizer.fit_transform(df["Message"])

# TODO: Extract labels (Spam column)
y = df["Spam"]

# TODO: Print the feature names
print("Feature Names:", vectorizer.get_feature_names_out())

# Step 3: Train a Naive Bayes Model
# TODO: Initialize MultinomialNB
model = MultinomialNB()

# TODO: Train the model with features and labels
model.fit(X, y)

# Step 4: Make Predictions
predictions = model.predict(X)

# Step 5: Print predictions as a list
print("Predictions:", predictions.tolist())
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def train_credit_card_fraud_model():
  """
    Loads the dataset, scales features, splits into train-test sets,
    trains MultinomialNB, Decision Tree, and Random Forest,
    and returns key variables for testing.
    Students need to fill in the TODOs.
    """

  # --- Step 1: Load Dataset ---
  df = pd.read_csv("creditcard.csv")

  # --- Step 2: Prepare Features and Target ---
  X = df.drop("Class", axis=1)
  y = df["Class"]

  # --- Step 3: Scale Features to [0, 1] for MultinomialNB ---
  scaler = MinMaxScaler()
  X_scaled = scaler.fit_transform(X)

  # --- Step 4: Train-Test Split ---
  X_train, X_test, y_train, y_test = train_test_split(
      X_scaled, y, test_size=0.2, random_state=42, stratify=y
  )

  # --- Step 5: Define Models ---
  models = {
      "Multinomial Naive Bayes": MultinomialNB(),
      "Decision Tree": DecisionTreeClassifier(random_state=42),
      "Random Forest": RandomForestClassifier(random_state=42),
  }

  # --- Step 6: Train and Evaluate ---
  for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc*100:.2f}%")

  # Return all variables for testing
  return X_train, X_test, y_train, y_test, X_scaled, models, scaler


# --- Run training only when executing script directly ---
if __name__ == "__main__":
  train_credit_card_fraud_model()
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


class BernoulliNBClassifier:
  """
    Train a Bernoulli Naive Bayes classifier.
    """

  def __init__(self):
    self.dataset = None
    self.model = None

  def load_dataset(self):
    # Load the CSV file
    self.dataset = pd.read_csv("smart_home_bernoulli_nb.csv")

  def prepare_data(self):
    # Separate input features and target class
    X = self.dataset.drop(columns=["AlarmTriggered"])
    y = self.dataset["AlarmTriggered"]

    # Return the training and testing datasets
    return train_test_split(X, y, test_size=0.2, random_state=42)

  def train_model(self, X_train, y_train):
    # Train the Bernoulli Naive Bayes classifier
    self.model = BernoulliNB()
    self.model.fit(X_train, y_train)

  def predict(self, X_test):
    # Return the predicted classes
    return self.model.predict(X_test)

  def evaluate_model(self, y_test, predictions):
    # Return the classification accuracy
    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
  classifier = BernoulliNBClassifier()

  classifier.load_dataset()

  X_train, X_test, y_train, y_test = classifier.prepare_data()

  classifier.train_model(X_train, y_train)

  predictions = classifier.predict(X_test)

  accuracy = classifier.evaluate_model(y_test, predictions)

  print("Classification Accuracy:")
  print(f"{accuracy:.4f}")
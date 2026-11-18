import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import (
    BernoulliNB,
    GaussianNB,
    MultinomialNB,
)


class NaiveBayesAnalyzer:
  """
    Compare different
    Naive Bayes variants.
    """

  def __init__(self):
    self.gaussian_dataset = None
    self.multinomial_dataset = None
    self.bernoulli_dataset = None

  def load_datasets(self):
    # Load all three CSV files
    self.gaussian_dataset = pd.read_csv("machine_gaussian_nb.csv")
    self.multinomial_dataset = pd.read_csv("ticket_multinomial_nb.csv")
    self.bernoulli_dataset = pd.read_csv("website_bernoulli_nb.csv")

  def prepare_gaussian_data(self):
    # Return the training and testing datasets for Gaussian Naive Bayes.
    X = self.gaussian_dataset.drop(columns=["MaintenanceRequired"])
    y = self.gaussian_dataset["MaintenanceRequired"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

  def prepare_multinomial_data(self):
    # Return the training and testing datasets for Multinomial Naive Bayes.
    X = self.multinomial_dataset.drop(columns=["CriticalTicket"])
    y = self.multinomial_dataset["CriticalTicket"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

  def prepare_bernoulli_data(self):
    # Return the training and testing datasets for Bernoulli Naive Bayes.
    X = self.bernoulli_dataset.drop(columns=["Purchased"])
    y = self.bernoulli_dataset["Purchased"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

  def evaluate_gaussian_nb(self, X_train, X_test, y_train, y_test):
    # Train Gaussian Naive Bayes and return the accuracy.
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

  def evaluate_multinomial_nb(self, X_train, X_test, y_train, y_test):
    # Train Multinomial Naive Bayes and return the accuracy.
    model = MultinomialNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

  def evaluate_bernoulli_nb(self, X_train, X_test, y_train, y_test):
    # Train Bernoulli Naive Bayes and return the accuracy.
    model = BernoulliNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


if __name__ == "__main__":
  analyzer = NaiveBayesAnalyzer()

  analyzer.load_datasets()

  X_train_g, X_test_g, y_train_g, y_test_g = analyzer.prepare_gaussian_data()

  X_train_m, X_test_m, y_train_m, y_test_m = analyzer.prepare_multinomial_data()

  X_train_b, X_test_b, y_train_b, y_test_b = analyzer.prepare_bernoulli_data()

  gaussian_accuracy = analyzer.evaluate_gaussian_nb(
      X_train_g, X_test_g, y_train_g, y_test_g
  )

  multinomial_accuracy = analyzer.evaluate_multinomial_nb(
      X_train_m, X_test_m, y_train_m, y_test_m
  )

  bernoulli_accuracy = analyzer.evaluate_bernoulli_nb(
      X_train_b, X_test_b, y_train_b, y_test_b
  )

  print("Gaussian Naive Bayes Accuracy:")
  print(f"{gaussian_accuracy:.4f}")
  print("Multinomial Naive Bayes Accuracy:")
  print(f"{multinomial_accuracy:.4f}")
  print("Bernoulli Naive Bayes Accuracy:")
  print(f"{bernoulli_accuracy:.4f}")
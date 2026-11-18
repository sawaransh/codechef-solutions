import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesClassifier:
    def __init__(self, filepath="dataset.csv"):
        """
        Load dataset.
        """
        print("Loading dataset...")
        self.df = pd.read_csv(filepath)

    def prepare_data(self):
        """
        Prepare features and target class.
        """
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, X_train=None, y_train=None):
        """
        Train Naive Bayes classifier.
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        print("\nTraining Naive Bayes classifier...")
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, *args):
        """
        Generate predictions for testing dataset.
        Supports predict(), predict(X_test), and predict(model, X_test).
        """
        if len(args) == 0:
            model = self.model
            X_test = self.X_test
        elif len(args) == 1:
            model = self.model
            X_test = args[0]
        else:
            model, X_test = args[0], args[1]
            
        self.y_pred = model.predict(X_test)
        return self.y_pred

    def evaluate(self, *args):
        """
        Compute accuracy and classification report.
        Supports evaluate(), evaluate(y_test, y_pred), and evaluate(model, X_test, y_test).
        """
        if len(args) == 0:
            y_test = self.y_test
            y_pred = self.y_pred
        elif len(args) == 2:
            y_test, y_pred = args[0], args[1]
        elif len(args) == 3:
            model, X_test, y_test = args[0], args[1], args[2]
            y_pred = model.predict(X_test)
            
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\nModel Accuracy:\n{accuracy}")
        print(f"\nClassification Report:\n{report}")
        return accuracy, report

    def train_and_evaluate(self):
        self.train_model()
        self.predict()
        return self.evaluate()

if __name__ == "__main__":
    classifier = NaiveBayesClassifier()
    classifier.prepare_data()
    classifier.train_and_evaluate()
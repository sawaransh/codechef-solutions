import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNModelEvaluator:
    """
    Evaluate KNN models using
    different values of K.
    """

    def __init__(self):

        self.dataset = None

    def load_dataset(self, filepath="patient_knn.csv"):

        # Load the CSV file
        self.dataset = pd.read_csv(filepath)

    def prepare_data(self):

        # Return the training
        # and testing data
        X = self.dataset[["Age", "BloodPressure", "Cholesterol", "HeartRate"]]
        y = self.dataset["HasDisease"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Return the accuracy
        # for each value of K
        accuracies = {}

        for k in [1, 3, 5, 7]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies[k] = accuracy_score(y_test, y_pred)

        return accuracies


if __name__ == "__main__":

    evaluator = KNNModelEvaluator()

    evaluator.load_dataset()

    X_train, X_test, y_train, y_test = (
        evaluator.prepare_data()
    )

    accuracies = evaluator.evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    for k, accuracy in accuracies.items():

        print(f"Accuracy for K = {k}:")
        print(f"{accuracy:.4f}\n")
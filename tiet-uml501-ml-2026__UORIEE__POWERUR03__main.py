
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class TitanicEnsembleModel:

    def __init__(self):
        self.train_data = pd.read_csv("Titanic_train.csv")
        self.test_data = pd.read_csv("Titanic_test.csv")

        self.X_train = None
        self.X_validation = None
        self.y_train = None
        self.y_validation = None

        self.models = {}
        self.validation_results = {}
        self.validation_results_comparison = {}

        self.best_weights = None
        self.best_threshold = 0.5
        self.final_strategy = None
        self.final_models = {}

        self.feature_columns = None
        self.explanation = ""

    # --------------------------------------------------
    # 1. FEATURE ENGINEERING
    # --------------------------------------------------

    def engineer_features(self, df):
        df = df.copy()

        df["FamilySize"] = (
            df["SibSp"].fillna(0)
            + df["Parch"].fillna(0)
            + 1
        )

        df["IsAlone"] = (
            df["FamilySize"] == 1
        ).astype(int)

        df["FarePerPerson"] = (
            df["Fare"].fillna(df["Fare"].median())
            / df["FamilySize"].replace(0, 1)
        )

        df["Title"] = (
            df["Name"]
            .fillna("")
            .str.extract(r",\s*([^.]*)\.", expand=False)
            .str.strip()
        )

        title_mapping = {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Rare",
            "Countess": "Rare",
            "Capt": "Rare",
            "Col": "Rare",
            "Don": "Rare",
            "Dr": "Rare",
            "Major": "Rare",
            "Rev": "Rare",
            "Sir": "Rare",
            "Jonkheer": "Rare",
            "Dona": "Rare"
        }

        df["Title"] = (
            df["Title"]
            .replace(title_mapping)
            .fillna("Unknown")
        )

        df["Deck"] = (
            df["Cabin"]
            .fillna("Unknown")
            .astype(str)
            .str[0]
        )

        df.drop(
            columns=[
                "Name",
                "Ticket",
                "Cabin",
                "PassengerId"
            ],
            errors="ignore",
            inplace=True
        )

        return df

    # --------------------------------------------------
    # 2. DATA SPLITTING
    # --------------------------------------------------

    def split_data(self):

        y = self.train_data["Survived"].astype(int)

        X = self.train_data.drop(
            columns=["Survived"]
        )

        X = self.engineer_features(X)

        self.feature_columns = X.columns.tolist()

        (
            self.X_train,
            self.X_validation,
            self.y_train,
            self.y_validation
        ) = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        return (
            self.X_train,
            self.X_validation,
            self.y_train,
            self.y_validation
        )

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------

    def create_preprocessor(self, X):

        numerical_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ])

        return preprocessor

    # --------------------------------------------------
    # 3. BUILD MODELS
    # --------------------------------------------------

    def build_models(self, X, y):

        classifiers = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            ),

            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        }

        self.models = {}

        for name, classifier in classifiers.items():

            pipeline = Pipeline([
                ("preprocessor", self.create_preprocessor(X)),
                ("classifier", classifier)
            ])

            pipeline.fit(X, y)
            self.models[name] = pipeline

        return self.models

    # --------------------------------------------------
    # 4. EVALUATE INDIVIDUAL MODELS
    # --------------------------------------------------

    def evaluate_models(self):

        results = {}

        for name, model in self.models.items():

            predictions = model.predict(self.X_validation)

            results[name] = {
                "accuracy": accuracy_score(
                    self.y_validation, predictions
                ),
                "precision": precision_score(
                    self.y_validation, predictions,
                    zero_division=0
                ),
                "recall": recall_score(
                    self.y_validation, predictions,
                    zero_division=0
                ),
                "f1_score": f1_score(
                    self.y_validation, predictions,
                    zero_division=0
                )
            }

        self.validation_results = results
        return results

    # --------------------------------------------------
    # 5. WEIGHTED ENSEMBLE PROBABILITIES
    # --------------------------------------------------

    def create_ensemble_probabilities(self, models, X, weights):

        probabilities = np.zeros(len(X), dtype=float)

        for name, model in models.items():

            model_probabilities = model.predict_proba(X)[:, 1]

            probabilities += (
                weights[name] * model_probabilities
            )

        return probabilities

    # --------------------------------------------------
    # 6. OPTIMIZE ENSEMBLE
    # --------------------------------------------------

    def optimize_ensemble(self):

        model_names = list(self.models.keys())

        best_f1 = -1
        best_accuracy = -1

        self.best_weights = None
        self.best_threshold = 0.5

        # Positive weights with a total of 1
        weight_combinations = []

        for a in range(1, 9):
            for b in range(1, 9):

                c = 10 - a - b

                if c > 0:

                    weights = {
                        model_names[0]: a / 10,
                        model_names[1]: b / 10,
                        model_names[2]: c / 10
                    }

                    weight_combinations.append(weights)

        for weights in weight_combinations:

            probabilities = self.create_ensemble_probabilities(
                self.models,
                self.X_validation,
                weights
            )

            for threshold in np.arange(0.30, 0.701, 0.02):

                predictions = (
                    probabilities >= threshold
                ).astype(int)

                current_f1 = f1_score(
                    self.y_validation,
                    predictions,
                    zero_division=0
                )

                current_accuracy = accuracy_score(
                    self.y_validation,
                    predictions
                )

                if (
                    current_f1 > best_f1
                    or (
                        np.isclose(current_f1, best_f1)
                        and current_accuracy > best_accuracy
                    )
                ):

                    best_f1 = current_f1
                    best_accuracy = current_accuracy

                    self.best_weights = weights.copy()
                    self.best_threshold = float(threshold)

        return {
            "weights": self.best_weights,
            "threshold": self.best_threshold,
            "accuracy": best_accuracy,
            "f1_score": best_f1
        }

    # --------------------------------------------------
    # 7. COMPARE STRATEGIES
    # --------------------------------------------------

    def compare_strategies(self):

        results = dict(self.validation_results)

        probabilities = self.create_ensemble_probabilities(
            self.models,
            self.X_validation,
            self.best_weights
        )

        predictions = (
            probabilities >= self.best_threshold
        ).astype(int)

        results["Optimized Ensemble"] = {
            "accuracy": accuracy_score(
                self.y_validation, predictions
            ),
            "precision": precision_score(
                self.y_validation, predictions,
                zero_division=0
            ),
            "recall": recall_score(
                self.y_validation, predictions,
                zero_division=0
            ),
            "f1_score": f1_score(
                self.y_validation, predictions,
                zero_division=0
            )
        }

        self.validation_results_comparison = results
        return results

    # --------------------------------------------------
    # 8. SELECT FINAL STRATEGY
    # --------------------------------------------------

    def select_final_strategy(self):

        if not self.validation_results_comparison:
            self.compare_strategies()

        results = self.validation_results_comparison

        self.final_strategy = max(
            results,
            key=lambda name: (
                results[name]["f1_score"],
                results[name]["accuracy"]
            )
        )

        return self.final_strategy

    # --------------------------------------------------
    # 9. EXPLAIN FINAL STRATEGY
    # --------------------------------------------------

    def explain_final_strategy(self):

        if not self.validation_results_comparison:
            self.compare_strategies()

        if self.final_strategy is None:
            self.select_final_strategy()

        results = self.validation_results_comparison

        selected = self.final_strategy
        selected_metrics = results[selected]

        alternatives = {
            name: metrics
            for name, metrics in results.items()
            if name != selected
        }

        if alternatives:

            strongest = max(
                alternatives,
                key=lambda name: (
                    alternatives[name]["f1_score"],
                    alternatives[name]["accuracy"]
                )
            )

            other_metrics = alternatives[strongest]

            comparison = (
                f"The strongest alternative was {strongest}, "
                f"with F1 score {other_metrics['f1_score']:.4f} "
                f"and accuracy {other_metrics['accuracy']:.4f}. "
            )

        else:
            comparison = "No alternative strategy was available. "

        explanation = (
            f"The selected strategy is {selected}. "
            f"It achieved a validation F1 score of "
            f"{selected_metrics['f1_score']:.4f} and accuracy of "
            f"{selected_metrics['accuracy']:.4f}. "
            f"{comparison}"
            f"Feature engineering captures family size, fare per "
            f"person, passenger title and cabin deck. Class weighting "
            f"helps handle class imbalance. Weighted probability "
            f"ensembling and validation-based threshold optimization "
            f"help select the strongest classification strategy."
        )

        self.explanation = explanation
        return explanation

    # --------------------------------------------------
    # 10. RETRAIN MODELS ON FULL TRAINING DATA
    # --------------------------------------------------

    def train_final_models(self):

        y = self.train_data["Survived"].astype(int)

        X = self.train_data.drop(
            columns=["Survived"]
        )

        X = self.engineer_features(X)

        classifiers = {
            "Logistic Regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            ),

            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        }

        self.final_models = {}

        for name, classifier in classifiers.items():

            pipeline = Pipeline([
                ("preprocessor", self.create_preprocessor(X)),
                ("classifier", classifier)
            ])

            pipeline.fit(X, y)
            self.final_models[name] = pipeline

        return self.final_models

    # --------------------------------------------------
    # 11. PREDICT TEST DATA
    # --------------------------------------------------

    def predict_test(self):

        X_test = self.engineer_features(self.test_data)

        X_test = X_test.reindex(
            columns=self.feature_columns
        )

        if self.final_strategy == "Optimized Ensemble":

            probabilities = self.create_ensemble_probabilities(
                self.final_models,
                X_test,
                self.best_weights
            )

            predictions = (
                probabilities >= self.best_threshold
            ).astype(int)

        else:

            predictions = self.final_models[
                self.final_strategy
            ].predict(X_test)

        return np.asarray(predictions, dtype=int)

    # --------------------------------------------------
    # 12. COMPLETE PIPELINE
    # --------------------------------------------------

    def train_and_evaluate(self):

        self.split_data()

        self.build_models(
            self.X_train,
            self.y_train
        )

        self.evaluate_models()

        self.optimize_ensemble()

        self.compare_strategies()

        self.select_final_strategy()

        self.explain_final_strategy()

        self.train_final_models()

        predictions = self.predict_test()

        return np.asarray(predictions, dtype=int)

    # --------------------------------------------------
    # RUN
    # --------------------------------------------------

    def run(self):

        predictions = self.train_and_evaluate()

        return np.asarray(predictions, dtype=int)


if __name__ == "__main__":

    model = TitanicEnsembleModel()

    predictions = model.run()

    print("\nFinal Predictions:")
    print(predictions)

    print("\nTotal predictions:", len(predictions))

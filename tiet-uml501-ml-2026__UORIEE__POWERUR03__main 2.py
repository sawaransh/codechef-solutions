import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class EnsembleResult(tuple):
    """Tuple of (weights, threshold) that also supports dict-style key lookup."""

    def __getitem__(self, key):
        if key == "weights":
            return super().__getitem__(0)
        if key == "threshold":
            return super().__getitem__(1)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "weights":
            return super().__getitem__(0)
        if key == "threshold":
            return super().__getitem__(1)
        return default


class TitanicEnsembleModel:
    def __init__(self):
        # Load Titanic_train.csv and Titanic_test.csv
        self.train_data = pd.read_csv("Titanic_train.csv")
        self.test_data = pd.read_csv("Titanic_test.csv")

        self.X_train = None
        self.X_validation = None
        self.y_train = None
        self.y_validation = None

        self.models = {}
        self.validation_results = None

        self.best_weights = None
        self.best_threshold = None
        self.final_strategy = None

        self.final_models = {}

    def engineer_features(self, df):
        """Create meaningful Titanic features."""
        data = df.copy()

        # Derive FamilySize from SibSp and Parch
        data["FamilySize"] = data["SibSp"].fillna(0) + data["Parch"].fillna(0) + 1

        # Derive IsAlone from FamilySize
        data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

        # Calculate fare paid per family member
        fare_filled = data["Fare"].fillna(data["Fare"].median())
        data["FarePerPerson"] = fare_filled / data["FamilySize"]

        # Extract passenger title from Name
        titles = data["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        rare_titles = {
            "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
            "Rev", "Sir", "Jonkheer", "Dona", "the Countess"
        }
        titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        data["Title"] = titles.apply(lambda t: "Rare" if t in rare_titles or pd.isna(t) else t)

        # Extract deck information from Cabin
        data["Deck"] = data["Cabin"].fillna("Unknown").astype(str).str[0]

        # Remove high-cardinality / non-informative raw fields and target if present
        drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
        data = data.drop(columns=[col for col in drop_cols if col in data.columns])

        return data

    def split_data(self):
        """Create feature-engineered training and validation sets."""
        X = self.engineer_features(self.train_data)
        y = self.train_data["Survived"].astype(int)

        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        return self.X_train, self.X_validation, self.y_train, self.y_validation

    def _create_preprocessor(self, X):
        """Create column transformer for numerical and categorical features."""
        numerical_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

    def build_models(self, X, y):
        """Build and train the three required model pipelines."""
        classifiers = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
            )
        }

        trained_models = {}
        for name, clf in classifiers.items():
            pipeline = Pipeline([
                ("preprocessor", self._create_preprocessor(X)),
                ("classifier", clf)
            ])
            pipeline.fit(X, y)
            trained_models[name] = pipeline

        self.models = trained_models
        return trained_models

    def _compute_metrics(self, y_true, y_pred):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    def evaluate_models(self):
        """Evaluate individual models on validation data."""
        results = {}
        items = self.models.items() if isinstance(self.models, dict) else enumerate(self.models)
        for name, model in items:
            preds = model.predict(self.X_validation)
            results[str(name)] = self._compute_metrics(self.y_validation, preds)

        self.validation_results = results
        return results

    def create_ensemble_probabilities(self, models, X, weights):
        """Combine positive-class probabilities using supplied weights."""
        if isinstance(models, dict):
            model_names = list(models.keys())
            model_list = list(models.values())
        else:
            model_names = [f"model_{i}" for i in range(len(models))]
            model_list = list(models)

        if isinstance(weights, dict):
            weight_list = [weights[name] for name in model_names]
        else:
            weight_list = list(weights)

        if len(model_list) < 2:
            raise ValueError("Ensemble requires more than one model.")
        if len(weight_list) != len(model_list):
            raise ValueError("Weights must match the number of models.")
        if any(w <= 0 for w in weight_list):
            raise ValueError("Every model must receive a positive weight.")
        if not np.isclose(sum(weight_list), 1.0):
            raise ValueError("Ensemble weights must sum to 1.")

        blended_probs = np.zeros(len(X), dtype=float)
        for model, weight in zip(model_list, weight_list):
            probs = model.predict_proba(X)[:, 1]
            blended_probs += weight * probs

        return blended_probs

    def optimize_ensemble(self):
        """Search ensemble weights and classification threshold."""
        if not self.models:
            if self.X_train is None or self.y_train is None:
                self.split_data()
            self.build_models(self.X_train, self.y_train)

        if isinstance(self.models, dict):
            model_names = list(self.models.keys())
            model_list = list(self.models.values())
        else:
            model_names = [f"model_{i}" for i in range(len(self.models))]
            model_list = list(self.models)

        n_models = len(model_list)
        val_probs = [m.predict_proba(self.X_validation)[:, 1] for m in model_list]

        # Generate positive weights summing to 1.0
        weight_candidates = []
        grid = [round(x * 0.1, 1) for x in range(1, 10)]
        if n_models == 3:
            for w1 in grid:
                for w2 in grid:
                    w3 = round(1.0 - w1 - w2, 1)
                    if w3 >= 0.1 and np.isclose(w1 + w2 + w3, 1.0):
                        weight_candidates.append([w1, w2, w3])
        else:
            weight_candidates.append([round(1.0 / n_models, 4)] * n_models)

        thresholds = np.round(np.arange(0.30, 0.71, 0.02), 2)

        best_score = (-1.0, -1.0)  # (f1_score, accuracy)
        best_weights = None
        best_threshold = 0.50

        for weights in weight_candidates:
            blended = np.zeros(len(self.X_validation), dtype=float)
            for w, probs in zip(weights, val_probs):
                blended += w * probs

            for thresh in thresholds:
                preds = (blended >= thresh).astype(int)
                f1 = f1_score(self.y_validation, preds, zero_division=0)
                acc = accuracy_score(self.y_validation, preds)
                if (round(f1, 6), round(acc, 6)) > best_score:
                    best_score = (round(f1, 6), round(acc, 6))
                    best_weights = list(weights)
                    best_threshold = float(thresh)

        self.best_weights = {
            name: w for name, w in zip(model_names, best_weights)
        }
        self.best_threshold = best_threshold

        return EnsembleResult((self.best_weights, self.best_threshold))

    def _normalize_comparison(self, comparison):
        """Normalize comparison output into a dict of {strategy_name: metrics}."""
        if isinstance(comparison, tuple):
            for item in comparison:
                if isinstance(item, dict):
                    return item
            return dict(comparison)
        if isinstance(comparison, list):
            return dict(comparison)
        return comparison

    def _extract_f1_acc(self, metrics):
        """Extract (f1_score, accuracy) from either a metric dict or tuple/list."""
        if isinstance(metrics, dict):
            f1 = metrics.get("f1_score", metrics.get("F1 Score", 0.0))
            acc = metrics.get("accuracy", metrics.get("Accuracy", 0.0))
            return float(f1), float(acc)
        if isinstance(metrics, (tuple, list)):
            if len(metrics) >= 4:
                return float(metrics[3]), float(metrics[0])
            if len(metrics) == 2:
                return float(metrics[0]), float(metrics[1])
        return float(metrics), 0.0

    def compare_strategies(self):
        """Compare individual models and optimized ensemble."""
        if not self.models:
            if self.X_train is None or self.y_train is None:
                self.split_data()
            self.build_models(self.X_train, self.y_train)
        if self.validation_results is None:
            self.evaluate_models()
        if self.best_weights is None or self.best_threshold is None:
            self.optimize_ensemble()

        val_res = self._normalize_comparison(self.validation_results)
        comparison = {
            k: (v.copy() if isinstance(v, dict) else v)
            for k, v in val_res.items()
        }

        ens_probs = self.create_ensemble_probabilities(
            self.models, self.X_validation, self.best_weights
        )
        ens_preds = (ens_probs >= self.best_threshold).astype(int)
        comparison["Optimized Ensemble"] = self._compute_metrics(self.y_validation, ens_preds)

        return comparison

    def select_final_strategy(self):
        """Select the strongest validation strategy."""
        raw_comparison = self.compare_strategies()
        comparison = self._normalize_comparison(raw_comparison)

        best_name = None
        best_key = (-1.0, -1.0)  # (f1_score, accuracy)

        for strategy_name, metrics in comparison.items():
            key = self._extract_f1_acc(metrics)
            if key > best_key:
                best_key = key
                best_name = strategy_name

        self.final_strategy = best_name
        return best_name

    def explain_final_strategy(self):
        """Explain the selected strategy using validation results."""
        comparison = self._normalize_comparison(self.compare_strategies())
        if self.final_strategy is None or self.final_strategy not in comparison:
            self.select_final_strategy()

        selected_raw = comparison[self.final_strategy]
        sel_f1, sel_acc = self._extract_f1_acc(selected_raw)
        sel_rec = selected_raw.get("recall", 0.0) if isinstance(selected_raw, dict) else 0.0
        sel_prec = selected_raw.get("precision", 0.0) if isinstance(selected_raw, dict) else 0.0

        alternatives = {
            k: v for k, v in comparison.items() if k != self.final_strategy
        }
        if alternatives:
            runner_up_name = max(
                alternatives.keys(),
                key=lambda k: self._extract_f1_acc(alternatives[k])
            )
            ru_f1, ru_acc = self._extract_f1_acc(alternatives[runner_up_name])
            f1_diff = round(sel_f1 - ru_f1, 4)
            acc_diff = round(sel_acc - ru_acc, 4)
            comparison_text = (
                f"outperforming the strongest alternative '{runner_up_name}' "
                f"(F1 Score: {ru_f1:.4f}, Accuracy: {ru_acc:.4f}) "
                f"by {f1_diff:+.4f} F1 and {acc_diff:+.4f} Accuracy. "
            )
        else:
            comparison_text = "ranking highest among all evaluated validation strategies. "

        thresh_val = self.best_threshold if self.best_threshold is not None else 0.50
        explanation = (
            f"Selected '{self.final_strategy}' because it achieved the highest validation F1 Score "
            f"({sel_f1:.4f}) and Accuracy ({sel_acc:.4f}), "
            f"{comparison_text}"
            f"Domain feature engineering (Title, Deck, FamilySize, IsAlone, FarePerPerson), balanced class weighting, "
            f"and validation-tuned probability thresholding ({thresh_val:.2f}) improved minority-class "
            f"recall ({sel_rec:.4f}) while maintaining strong precision ({sel_prec:.4f})."
        )
        return explanation

    def train_final_models(self):
        """Retrain required models using all labeled training data."""
        X_full = self.engineer_features(self.train_data)
        y_full = self.train_data["Survived"].astype(int)

        if not self.models:
            self.build_models(X_full, y_full)

        if self.final_strategy == "Optimized Ensemble" or self.final_strategy is None:
            models_to_retrain = list(self.models.keys())
        else:
            models_to_retrain = [self.final_strategy]

        self.final_models = {}
        for name in models_to_retrain:
            cloned_pipeline = clone(self.models[name])
            cloned_pipeline.fit(X_full, y_full)
            self.final_models[name] = cloned_pipeline

        return self.final_models

    def predict_test(self):
        """Generate final predictions for Titanic_test.csv."""
        X_test = self.engineer_features(self.test_data)

        if self.final_strategy == "Optimized Ensemble":
            probs = self.create_ensemble_probabilities(
                self.final_models, X_test, self.best_weights
            )
            threshold = self.best_threshold if self.best_threshold is not None else 0.5
            predictions = (probs >= threshold).astype(int)
        else:
            model = self.final_models[self.final_strategy]
            predictions = model.predict(X_test).astype(int)

        return predictions

    def run(self):
        """Run the complete modeling workflow."""
        self.split_data()
        self.build_models(self.X_train, self.y_train)
        self.evaluate_models()
        best_weights, best_threshold = self.optimize_ensemble()
        comparison = self.compare_strategies()
        strategy = self.select_final_strategy()
        reason = self.explain_final_strategy()
        self.train_final_models()
        predictions = self.predict_test()

        print("Validation Comparison:")
        print(comparison)
        print("\nOptimized Ensemble:")
        print({"weights": best_weights, "threshold": best_threshold})
        print("\nFinal Strategy:")
        print(strategy)
        print("\nReason:")
        print(reason)

        return predictions


if __name__ == "__main__":
    model = TitanicEnsembleModel()
    predictions = model.run()

    print("\nFinal Predictions:")
    print(predictions)
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


class IrisANNExperiment:

    def __init__(self, epochs=8, random_state=42):
        self.epochs = epochs
        self.random_state = random_state
        self.X = self.y = self.class_names = None
        self.X_train = self.X_validation = self.X_test = None
        self.y_train = self.y_validation = self.y_test = None
        self.feature_means = self.feature_stds = None
        self.experiment_results = []
        self.best_config = None
        self.best_weights = None
        self.best_history = None
        self.best_model = None
        self.final_evaluation = None

    def load_data(self):
        iris = load_iris()
        self.X = iris.data.astype(np.float64)
        self.y = iris.target
        self.class_names = iris.target_names
        return self.X, self.y

    def split_and_normalize(self):
        # 60% training, 40% temp (validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X,
            self.y,
            test_size=0.4,
            stratify=self.y,
            random_state=self.random_state,
        )
        # Split remaining 40% equally into 20% validation and 20% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

        # Compute normalization statistics strictly from training set
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0)

        self.X_train = (X_train - self.feature_means) / self.feature_stds
        self.X_validation = (X_val - self.feature_means) / self.feature_stds
        self.X_test = (X_test - self.feature_means) / self.feature_stds

        self.y_train = y_train
        self.y_validation = y_val
        self.y_test = y_test

        return (
            (self.X_train, self.X_validation, self.X_test),
            (self.y_train, self.y_validation, self.y_test),
        )

    def build_model(self, hidden_neurons, learning_rate):
        tf.keras.utils.set_random_seed(self.random_state)
        model = Sequential([
            Input(shape=(4,)),
            Dense(hidden_neurons, activation="relu"),
            Dense(3, activation=tf.keras.activations.softmax),
        ])
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_model(self, model, batch_size):
        start_time = time.perf_counter()
        history_obj = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_validation, self.y_validation),
            epochs=self.epochs,
            batch_size=batch_size,
            verbose=0,
        )
        training_time = time.perf_counter() - start_time
        return model, history_obj, training_time

    def run_hyperparameter_experiments(self):
        configurations = [
            {"hidden_neurons": 8, "learning_rate": 0.001, "batch_size": 16},
            {"hidden_neurons": 8, "learning_rate": 0.010, "batch_size": 16},
            {"hidden_neurons": 16, "learning_rate": 0.001, "batch_size": 16},
            {"hidden_neurons": 16, "learning_rate": 0.010, "batch_size": 16},
        ]

        self.experiment_results = []
        best_val_acc = -1.0
        best_val_loss = float("inf")

        for cfg in configurations:
            tf.keras.backend.clear_session()
            model = self.build_model(cfg["hidden_neurons"], cfg["learning_rate"])
            model, history_obj, train_time = self.train_model(model, cfg["batch_size"])
            hist = history_obj.history if hasattr(history_obj, "history") else history_obj

            train_acc = float(hist["accuracy"][-1])
            val_acc = float(hist["val_accuracy"][-1])
            val_loss = float(hist["val_loss"][-1])

            test_preds = np.argmax(model.predict(self.X_test, verbose=0), axis=1)
            test_acc = float(accuracy_score(self.y_test, test_preds))

            record = {
                "hidden_neurons": cfg["hidden_neurons"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "training_accuracy": train_acc,
                "validation_accuracy": val_acc,
                "validation_loss": val_loss,
                "test_accuracy": test_acc,
                "training_time": float(train_time),
            }
            self.experiment_results.append(record)

            if (val_acc > best_val_acc) or (
                np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
            ):
                best_val_acc = val_acc
                best_val_loss = val_loss
                self.best_config = dict(record)
                self.best_weights = model.get_weights()
                self.best_history = dict(hist)

            del model

        return self.experiment_results

    def compare_models(self):
        columns = [
            "hidden_neurons",
            "learning_rate",
            "batch_size",
            "training_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "training_time",
        ]
        return pd.DataFrame(self.experiment_results)[columns]

    def select_best_model(self):
        sorted_results = sorted(
            self.experiment_results,
            key=lambda r: (-r["validation_accuracy"], r["validation_loss"]),
        )
        self.best_config = dict(sorted_results[0])
        return self.best_config

    def restore_best_model(self):
        tf.keras.backend.clear_session()
        model = self.build_model(
            self.best_config["hidden_neurons"],
            self.best_config["learning_rate"],
        )
        model.set_weights(self.best_weights)
        self.best_model = model
        return self.best_model

    def evaluate_final_model(self):
        y_probs = self.best_model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)

        self.final_evaluation = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            "precision": float(precision_score(self.y_test, y_pred, average="macro")),
            "recall": float(recall_score(self.y_test, y_pred, average="macro")),
            "f1_score": float(f1_score(self.y_test, y_pred, average="macro")),
        }
        return self.final_evaluation

    def plot_training_history(self):
        hist = self.best_history.history if hasattr(self.best_history, "history") else self.best_history
        epochs_range = range(1, len(hist["accuracy"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(epochs_range, hist["accuracy"], label="Training")
        axes[0].plot(epochs_range, hist["val_accuracy"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(epochs_range, hist["loss"], label="Training")
        axes[1].plot(epochs_range, hist["val_loss"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()
        plot_filename = "ann_training_history.png"
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close(fig)
        return plot_filename

    def get_fitting_evidence(self):
        hist = self.best_history.history if hasattr(self.best_history, "history") else self.best_history
        final_train_acc = float(hist["accuracy"][-1])
        final_val_acc = float(hist["val_accuracy"][-1])
        final_train_loss = float(hist["loss"][-1])
        final_val_loss = float(hist["val_loss"][-1])
        accuracy_gap = float(final_train_acc - final_val_acc)

        overfitting_indicator = bool(accuracy_gap > 0.10)
        underfitting_indicator = bool(final_train_acc < 0.80 and final_val_acc < 0.80)

        return {
            "final_training_accuracy": final_train_acc,
            "final_validation_accuracy": final_val_acc,
            "final_training_loss": final_train_loss,
            "final_validation_loss": final_val_loss,
            "accuracy_gap": accuracy_gap,
            "overfitting_indicator": overfitting_indicator,
            "underfitting_indicator": underfitting_indicator,
        }

    def run(self):
        self.load_data()
        self.split_and_normalize()
        self.run_hyperparameter_experiments()
        comparison = self.compare_models()
        best_config = self.select_best_model()
        self.restore_best_model()
        final_evaluation = self.evaluate_final_model()
        plot_file = self.plot_training_history()
        fitting_evidence = self.get_fitting_evidence()

        return {
            "comparison": comparison,
            "best_config": best_config,
            "final_evaluation": final_evaluation,
            "plot_file": plot_file,
            "fitting_evidence": fitting_evidence,
        }


if __name__ == "__main__":
    experiment = IrisANNExperiment()
    results = experiment.run()

    print("Hyperparameter Comparison:")
    print(results["comparison"].to_string(index=False))

    print("\nBest Configuration:")
    print(results["best_config"])

    print("\nFinal Evaluation:")
    print(results["final_evaluation"])

    print("\nFitting Evidence:")
    print(results["fitting_evidence"])

    print("\nTraining Plot:")
    print(results["plot_file"])
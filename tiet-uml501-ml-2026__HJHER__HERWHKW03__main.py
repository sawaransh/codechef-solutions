import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam


class ANNDesignExperiment:

    def __init__(self, epochs=10, random_state=42):
        self.epochs = epochs
        self.random_state = random_state
        self.X_train = self.X_validation = self.X_test = None
        self.y_train = self.y_validation = self.y_test = None
        self.feature_means = self.feature_stds = None
        self.results = []
        self.histories = {}

    def load_and_prepare_data(self):
        iris = load_iris()
        X = iris.data.astype(np.float64)
        y = iris.target

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.4,
            stratify=y,
            random_state=self.random_state,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

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

    def create_optimizer(self, name, learning_rate):
        key = name.strip().lower()
        if key == "sgd":
            return SGD(learning_rate=learning_rate)
        elif key in ("momentum", "sgd with momentum", "sgd_momentum"):
            return SGD(learning_rate=learning_rate, momentum=0.9)
        elif key == "rmsprop":
            return RMSprop(learning_rate=learning_rate)
        elif key == "adam":
            return Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def build_model(
        self,
        activation="relu",
        optimizer="adam",
        learning_rate=0.001,
        dropout_rate=0.0,
        l2_strength=0.0,
        large_network=False,
    ):
        tf.keras.utils.set_random_seed(self.random_state)
        units_1, units_2 = (64, 32) if large_network else (16, 8)
        kernel_reg = l2(l2_strength) if l2_strength > 0 else None

        layers = [
            Input(shape=(4,)),
            Dense(units_1, activation=activation, kernel_regularizer=kernel_reg),
        ]
        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))

        layers.append(
            Dense(units_2, activation=activation, kernel_regularizer=kernel_reg)
        )
        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))

        layers.append(Dense(3, activation=tf.keras.activations.softmax))

        model = Sequential(layers)
        opt = self.create_optimizer(optimizer, learning_rate)
        model.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train_model(self, model, X_train=None, y_train=None):
        if self.X_train is None:
            self.load_and_prepare_data()

        X_tr = self.X_train if X_train is None else X_train
        y_tr = self.y_train if y_train is None else y_train

        start_time = time.perf_counter()
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(self.X_validation, self.y_validation),
            epochs=self.epochs,
            batch_size=16,
            verbose=0,
        )
        training_time = time.perf_counter() - start_time
        return model, history, training_time

    def epochs_to_converge(self, history):
        hist = history.history if hasattr(history, "history") else history
        val_losses = hist["val_loss"]
        consecutive = 0

        for idx in range(1, len(val_losses)):
            if abs(val_losses[idx] - val_losses[idx - 1]) <= 0.005:
                consecutive += 1
                if consecutive >= 2:
                    return idx + 1
            else:
                consecutive = 0

        return len(val_losses)

    def _evaluate_and_record(
        self,
        model,
        history,
        training_time,
        experiment_name,
        configuration_label,
        activation_label,
        regularization_label,
        optimizer_label,
        history_key,
    ):
        hist = history.history if hasattr(history, "history") else history
        _, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)

        record = {
            "experiment": experiment_name,
            "configuration": configuration_label,
            "activation": activation_label,
            "regularization": regularization_label,
            "optimizer": optimizer_label,
            "training_accuracy": float(hist["accuracy"][-1]),
            "validation_accuracy": float(hist["val_accuracy"][-1]),
            "test_accuracy": float(test_acc),
            "training_time": float(training_time),
            "epochs_to_converge": int(self.epochs_to_converge(hist)),
        }
        self.results.append(record)
        self.histories[history_key] = dict(hist)
        return record

    def run_activation_experiments(self):
        if self.X_train is None:
            self.load_and_prepare_data()

        activations = ["sigmoid", "tanh", "relu"]
        group_records = []

        for act in activations:
            tf.keras.backend.clear_session()
            model = self.build_model(
                activation=act,
                optimizer="adam",
                learning_rate=0.001,
                dropout_rate=0.0,
                l2_strength=0.0,
                large_network=False,
            )
            model, history, train_time = self.train_model(model)
            rec = self._evaluate_and_record(
                model=model,
                history=history,
                training_time=train_time,
                experiment_name="Activation",
                configuration_label=act,
                activation_label=act,
                regularization_label="None",
                optimizer_label="Adam",
                history_key=f"activation_{act}",
            )
            group_records.append(rec)
            del model

        return group_records

    def run_regularization_experiments(self):
        if self.X_train is None:
            self.load_and_prepare_data()

        X_sub = self.X_train[:30]
        y_sub = self.y_train[:30]

        configs = [
            {"label": "None", "dropout_rate": 0.0, "l2_strength": 0.0},
            {"label": "Dropout", "dropout_rate": 0.30, "l2_strength": 0.0},
            {"label": "L2", "dropout_rate": 0.0, "l2_strength": 0.001},
        ]
        group_records = []

        for cfg in configs:
            tf.keras.backend.clear_session()
            model = self.build_model(
                activation="relu",
                optimizer="adam",
                learning_rate=0.001,
                dropout_rate=cfg["dropout_rate"],
                l2_strength=cfg["l2_strength"],
                large_network=True,
            )
            model, history, train_time = self.train_model(model, X_train=X_sub, y_train=y_sub)
            rec = self._evaluate_and_record(
                model=model,
                history=history,
                training_time=train_time,
                experiment_name="Regularization",
                configuration_label=cfg["label"],
                activation_label="relu",
                regularization_label=cfg["label"],
                optimizer_label="Adam",
                history_key=f"regularization_{cfg['label']}",
            )
            group_records.append(rec)
            del model

        return group_records

    def run_optimizer_experiments(self):
        if self.X_train is None:
            self.load_and_prepare_data()

        configs = [
            {"opt_arg": "sgd", "label": "SGD", "lr": 0.01},
            {"opt_arg": "momentum", "label": "SGD Momentum", "lr": 0.01},
            {"opt_arg": "rmsprop", "label": "RMSprop", "lr": 0.001},
            {"opt_arg": "adam", "label": "Adam", "lr": 0.001},
        ]
        group_records = []

        for cfg in configs:
            tf.keras.backend.clear_session()
            model = self.build_model(
                activation="relu",
                optimizer=cfg["opt_arg"],
                learning_rate=cfg["lr"],
                dropout_rate=0.0,
                l2_strength=0.0,
                large_network=False,
            )
            model, history, train_time = self.train_model(model)
            rec = self._evaluate_and_record(
                model=model,
                history=history,
                training_time=train_time,
                experiment_name="Optimizer",
                configuration_label=cfg["label"],
                activation_label="relu",
                regularization_label="None",
                optimizer_label=cfg["label"],
                history_key=f"optimizer_{cfg['label']}",
            )
            group_records.append(rec)
            del model

        return group_records

    def plot_activation_curves(self):
        if "activation_sigmoid" not in self.histories:
            self.run_activation_experiments()

        fig, ax = plt.subplots(figsize=(7, 5))
        for act in ["sigmoid", "tanh", "relu"]:
            hist = self.histories[f"activation_{act}"]
            epochs = range(1, len(hist["val_accuracy"]) + 1)
            ax.plot(epochs, hist["val_accuracy"], label=act)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.legend()
        plt.tight_layout()
        filename = "activation_comparison.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return filename

    def plot_regularization_curves(self):
        if "regularization_None" not in self.histories:
            self.run_regularization_experiments()

        fig, ax = plt.subplots(figsize=(7, 5))
        for label in ["None", "Dropout", "L2"]:
            hist = self.histories[f"regularization_{label}"]
            epochs = range(1, len(hist["val_loss"]) + 1)
            ax.plot(epochs, hist["val_loss"], label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend()
        plt.tight_layout()
        filename = "regularization_comparison.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return filename

    def plot_optimizer_curves(self):
        if "optimizer_SGD" not in self.histories:
            self.run_optimizer_experiments()

        fig, ax = plt.subplots(figsize=(7, 5))
        for label in ["SGD", "SGD Momentum", "RMSprop", "Adam"]:
            hist = self.histories[f"optimizer_{label}"]
            epochs = range(1, len(hist["loss"]) + 1)
            ax.plot(epochs, hist["loss"], label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.legend()
        plt.tight_layout()
        filename = "optimizer_comparison.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return filename

    def create_results_table(self):
        columns = [
            "experiment",
            "configuration",
            "activation",
            "regularization",
            "optimizer",
            "training_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "training_time",
            "epochs_to_converge",
        ]
        return pd.DataFrame(self.results)[columns]

    def recommend_configuration(self):
        def _best_in_group(exp_name):
            group = [r for r in self.results if r["experiment"] == exp_name]
            best = sorted(
                group,
                key=lambda r: (-r["validation_accuracy"], r["epochs_to_converge"]),
            )[0]
            return {
                "configuration": best["configuration"],
                "validation_accuracy": best["validation_accuracy"],
                "test_accuracy": best["test_accuracy"],
                "epochs_to_converge": best["epochs_to_converge"],
            }

        return {
            "best_activation": _best_in_group("Activation"),
            "best_regularization": _best_in_group("Regularization"),
            "best_optimizer": _best_in_group("Optimizer"),
        }

    def run(self):
        self.results = []
        self.histories = {}
        self.load_and_prepare_data()
        self.run_activation_experiments()
        self.run_regularization_experiments()
        self.run_optimizer_experiments()

        results_table = self.create_results_table()
        p1 = self.plot_activation_curves()
        p2 = self.plot_regularization_curves()
        p3 = self.plot_optimizer_curves()
        recommendation = self.recommend_configuration()

        return {
            "results_table": results_table,
            "recommendation": recommendation,
            "plots": [p1, p2, p3],
        }


if __name__ == "__main__":
    experiment = ANNDesignExperiment()
    output = experiment.run()

    print("Results Table:")
    print(output["results_table"].to_string(index=False))

    print("\nRecommendation:")
    print(output["recommendation"])

    print("\nPlots:")
    for filename in output["plots"]:
        print(filename)
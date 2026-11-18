import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LogicGatePerceptron:
    """Train a manual perceptron on AND, OR, and XOR gates."""

    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.X = None
        self.targets = {}
        self.results = {}

    def prepare_data(self):
        """Create logical-gate training data."""
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=float)

        self.targets = {
            "AND": np.array([0, 0, 0, 1], dtype=int),
            "OR": np.array([0, 1, 1, 1], dtype=int),
            "XOR": np.array([0, 1, 1, 0], dtype=int)
        }
        return self.X, self.targets

    def step_activation(self, value):
        """Apply threshold activation."""
        return np.where(value >= 0, 1, 0)

    def predict(self, X, weights, bias):
        """Predict binary outputs."""
        weighted_sum = np.dot(X, weights) + bias
        return self.step_activation(weighted_sum)

    def train_gate(self, X, y):
        """Train one perceptron."""
        weights = np.zeros(X.shape[1], dtype=float)
        bias = 0.0
        error_history = []
        converged = False
        epochs_run = 0

        for epoch in range(1, self.max_epochs + 1):
            epochs_run = epoch
            misclassifications = 0

            for x_i, target in zip(X, y):
                z = np.dot(x_i, weights) + bias
                prediction = int(self.step_activation(z))
                error = target - prediction

                if error != 0:
                    misclassifications += 1
                    weights = weights + self.learning_rate * error * x_i
                    bias = bias + self.learning_rate * error

            error_history.append(misclassifications)

            if misclassifications == 0:
                converged = True
                break

        return {
            "weights": weights,
            "bias": float(bias),
            "error_history": error_history,
            "epochs": epochs_run,
            "converged": converged
        }

    def train_all_gates(self):
        """Train independent perceptrons for AND, OR and XOR."""
        if self.X is None or not self.targets:
            self.prepare_data()

        self.results = {}
        for gate_name, y in self.targets.items():
            self.results[gate_name] = self.train_gate(self.X, y)

        return self.results

    def plot_decision_boundary(self, gate_name):
        """Plot the learned decision boundary for AND or OR."""
        if gate_name not in self.results:
            raise ValueError(f"Gate '{gate_name}' has not been trained.")

        gate_result = self.results[gate_name]
        weights = gate_result["weights"]
        bias = gate_result["bias"]
        y = self.targets[gate_name]

        fig, ax = plt.subplots(figsize=(6, 5))

        class_0 = self.X[y == 0]
        class_1 = self.X[y == 1]

        ax.scatter(class_0[:, 0], class_0[:, 1], label="Class 0", s=60, zorder=3)
        ax.scatter(class_1[:, 0], class_1[:, 1], label="Class 1", s=60, zorder=3)

        x_vals = np.linspace(-0.5, 1.5, 200)
        if not np.isclose(weights[1], 0.0):
            y_vals = -(weights[0] * x_vals + bias) / weights[1]
            ax.plot(x_vals, y_vals, label="Decision Boundary", zorder=2)
        elif not np.isclose(weights[0], 0.0):
            x_boundary = -bias / weights[0]
            ax.axvline(x=x_boundary, label="Decision Boundary", zorder=2)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("Input 1")
        ax.set_ylabel("Input 2")
        ax.set_title(f"{gate_name} Gate Decision Boundary")
        ax.grid(True)
        ax.legend()

        filename = f"{gate_name.lower()}_decision_boundary.png"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return filename

    def plot_error_curves(self):
        """Plot epoch errors for all three gates."""
        fig, ax = plt.subplots(figsize=(8, 5))

        for gate_name in ["AND", "OR", "XOR"]:
            if gate_name in self.results:
                errors = self.results[gate_name]["error_history"]
                epochs = range(1, len(errors) + 1)
                ax.plot(epochs, errors, marker="o", markersize=4, label=gate_name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of Misclassifications")
        ax.set_title("Perceptron Training Error")
        ax.grid(True)
        ax.legend()

        filename = "perceptron_errors.png"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return filename

    def analyze_xor(self):
        """Explain the XOR convergence result."""
        xor_res = self.results.get("XOR", {})
        converged = xor_res.get("converged", False)
        final_error = xor_res.get("error_history", [-1])[-1]

        if not converged:
            return (
                f"XOR failed to converge after {xor_res.get('epochs', self.max_epochs)} epochs "
                f"(ending with {final_error} misclassifications): XOR does not converge to zero "
                "training error using a single perceptron because XOR is not linearly separable. "
                "A single straight decision boundary cannot separate its two classes; solving XOR "
                "requires a model capable of learning multiple decision boundaries, such as a "
                "neural network with a hidden layer."
            )
        return "XOR converged unexpectedly."

    def run(self):
        """Run the complete perceptron experiment."""
        self.prepare_data()
        self.train_all_gates()
        self.plot_decision_boundary("AND")
        self.plot_decision_boundary("OR")
        self.plot_error_curves()

        output = dict(self.results)
        output["xor_analysis"] = self.analyze_xor()
        return output


if __name__ == "__main__":
    model = LogicGatePerceptron()
    results = model.run()

    for gate in ["AND", "OR", "XOR"]:
        print(f"\n{gate}:")
        print("Weights:", results[gate]["weights"])
        print("Bias:", results[gate]["bias"])
        print("Converged:", results[gate]["converged"])

    print("\nXOR Analysis:")
    print(results["xor_analysis"])

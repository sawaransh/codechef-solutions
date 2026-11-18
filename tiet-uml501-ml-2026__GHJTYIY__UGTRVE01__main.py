import pandas as pd
import numpy as np


class HousingKMeans:
    """Implement K-Means manually using SSE convergence."""

    def __init__(
        self,
        k_min=2,
        k_max=10,
        max_iterations=100,
        tolerance=1e-4,
        n_init=5,
        random_state=42
    ):
        self.df = pd.read_csv('USA_Housing.csv')

        self.k_min = k_min
        self.k_max = k_max
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_init = n_init
        self.random_state = random_state

        self.X = None
        self.X_scaled = None

        self.feature_names = None
        self.feature_means = None
        self.feature_stds = None

        self.elbow_results = {}
        self.best_k = None

        self.best_centroids = None
        self.best_labels = None
        self.final_sse = None

    def get_dataset(self):
        """Return the loaded dataset."""
        return self.df

    def prepare_data(self):
        """Use and standardize all six numerical features."""
        self.feature_names = list(self.df.columns)
        self.X = self.df.to_numpy(dtype=float)

        self.feature_means = np.mean(self.X, axis=0)
        self.feature_stds = np.std(self.X, axis=0)

        safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self.X_scaled = (self.X - self.feature_means) / safe_stds
        return self.X_scaled

    def initialize_centroids(self, X, k, rng=None):
        """Initialize K centroids using K-Means++ initialization."""
        if rng is None:
            rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        centroids = np.empty((k, n_features), dtype=float)

        first_idx = rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        for c in range(1, k):
            dist_sq = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :c, :]) ** 2, axis=2),
                axis=1
            )
            total_dist_sq = np.sum(dist_sq)
            if total_dist_sq == 0:
                next_idx = rng.integers(0, n_samples)
            else:
                probs = dist_sq / total_dist_sq
                next_idx = rng.choice(n_samples, p=probs)
            centroids[c] = X[next_idx]

        return centroids

    def assign_clusters(self, X, centroids):
        """Assign each sample to the nearest centroid using squared Euclidean distance."""
        dist_sq = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(dist_sq, axis=1)

    def update_centroids(self, X, labels, k, rng=None):
        """Recalculate each centroid as the mean of its assigned points."""
        if rng is None:
            rng = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        new_centroids = np.empty((k, n_features), dtype=float)

        for cluster_idx in range(k):
            cluster_points = X[labels == cluster_idx]
            if len(cluster_points) == 0:
                random_idx = rng.integers(0, n_samples)
                new_centroids[cluster_idx] = X[random_idx]
            else:
                new_centroids[cluster_idx] = np.mean(cluster_points, axis=0)

        return new_centroids

    def calculate_sse(self, X, labels, centroids):
        """Calculate Sum of Squared Errors (SSE) for the current clustering."""
        assigned_centroids = centroids[labels]
        return float(np.sum((X - assigned_centroids) ** 2))

    def fit_kmeans(self, arg1, arg2=None):
        """Run K-Means with multiple initializations and return the best run."""
        if isinstance(arg1, np.ndarray):
            X_data = arg1
            k = int(arg2)
        else:
            k = int(arg1)
            if arg2 is not None and isinstance(arg2, np.ndarray):
                X_data = arg2
            else:
                if self.X_scaled is None:
                    self.prepare_data()
                X_data = self.X_scaled

        rng = np.random.default_rng(self.random_state)

        best_centroids = None
        best_labels = None
        best_sse = np.inf
        best_iterations = 0
        best_history = []

        for _ in range(self.n_init):
            centroids = self.initialize_centroids(X_data, k, rng)
            prev_sse = np.inf
            sse_history = []
            labels = None
            iterations = 0

            for iteration in range(1, self.max_iterations + 1):
                iterations = iteration
                labels = self.assign_clusters(X_data, centroids)
                centroids = self.update_centroids(X_data, labels, k, rng)
                labels = self.assign_clusters(X_data, centroids)
                current_sse = self.calculate_sse(X_data, labels, centroids)
                sse_history.append(current_sse)

                if abs(prev_sse - current_sse) <= self.tolerance:
                    break
                prev_sse = current_sse

            final_run_sse = sse_history[-1]
            if final_run_sse < best_sse:
                best_sse = final_run_sse
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_iterations = iterations
                best_history = sse_history

        return best_centroids, best_labels, best_sse, best_iterations, best_history

    def run_elbow_method(self):
        """Evaluate SSE across K from k_min to k_max."""
        if self.X_scaled is None:
            self.prepare_data()

        self.elbow_results = {}
        for k in range(self.k_min, self.k_max + 1):
            _, _, sse, _, _ = self.fit_kmeans(k)
            self.elbow_results[k] = sse

        return self.elbow_results

    def find_elbow(self):
        """Determine optimal K using maximum perpendicular distance from the endpoint line."""
        if not self.elbow_results:
            self.run_elbow_method()

        k_values = np.array(sorted(self.elbow_results.keys()), dtype=float)
        sse_values = np.array([self.elbow_results[int(k)] for k in k_values], dtype=float)

        if len(k_values) <= 2:
            self.best_k = int(k_values[0])
            return self.best_k

        p1 = np.array([k_values[0], sse_values[0]])
        p2 = np.array([k_values[-1], sse_values[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        max_dist = -1.0
        selected_k = int(k_values[1])

        for i in range(1, len(k_values) - 1):
            p = np.array([k_values[i], sse_values[i]])
            if line_len == 0:
                dist = np.linalg.norm(p - p1)
            else:
                dist = abs(line_vec[0] * (p1[1] - p[1]) - (p1[0] - p[0]) * line_vec[1]) / line_len

            if dist > max_dist:
                max_dist = dist
                selected_k = int(k_values[i])

        self.best_k = selected_k
        return self.best_k

    def fit_best_model(self):
        """Fit the final K-Means model using self.best_k."""
        if self.best_k is None:
            self.find_elbow()

        centroids, labels, sse, _, _ = self.fit_kmeans(self.best_k)
        self.best_centroids = centroids
        self.best_labels = labels
        self.final_sse = sse
        return self.best_centroids, self.best_labels, self.final_sse

    def get_cluster_summary(self):
        """Return summary statistics for each cluster with centroids in original feature scale."""
        if self.best_centroids is None or self.best_labels is None:
            self.fit_best_model()

        safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        original_centroids = self.best_centroids * safe_stds + self.feature_means

        summary = []
        for cluster_idx in range(self.best_k):
            count = int(np.sum(self.best_labels == cluster_idx))
            centroid_dict = {
                feature: float(original_centroids[cluster_idx, j])
                for j, feature in enumerate(self.feature_names)
            }
            summary.append({
                "cluster": cluster_idx,
                "num_samples": count,
                "centroid": centroid_dict
            })

        return summary

    def run(self):
        """Execute the full clustering and elbow selection pipeline."""
        self.prepare_data()
        elbow_results = self.run_elbow_method()
        best_k = self.find_elbow()
        self.fit_best_model()
        cluster_summary = self.get_cluster_summary()
        return {
            "elbow_results": elbow_results,
            "best_k": best_k,
            "final_sse": self.final_sse,
            "cluster_summary": cluster_summary
        }

if __name__ == "__main__":
    model = HousingKMeans()

    results = model.run()

    print("Elbow Results:")
    print(results["elbow_results"])

    print("\nBest K:")
    print(results["best_k"])

    print("\nFinal SSE:")
    print(results["final_sse"])

    print("\nCluster Summary:")
    print(results["cluster_summary"])
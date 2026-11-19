import pandas as pd
import numpy as np


class FlexibleResult(dict):
    """Dict result that also supports tuple-style unpacking and integer indexing."""

    def __init__(self, ordered_keys, data):
        super().__init__(data)
        self._ordered_keys = list(ordered_keys)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return dict.__getitem__(self, self._ordered_keys[key])
        if isinstance(key, slice):
            return tuple(dict.__getitem__(self, k) for k in self._ordered_keys[key])
        return dict.__getitem__(self, key)

    def __iter__(self):
        return iter([dict.__getitem__(self, k) for k in self._ordered_keys])


class HousingKMedoids:
    """Implement K-Medoids manually using SSE convergence."""

    def __init__(self, k_min=2, k_max=10, max_iterations=100,
                 tolerance=1e-4, n_init=5, random_state=42):
        self.df = pd.read_csv("USA_Housing.csv")
        self.data = self.df

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

        self.best_medoid_indices = None
        self.best_medoids = None
        self.best_labels = None
        self.final_sse = None

    def prepare_data(self):
        """Use and standardize all six numerical features."""
        numeric = self.df.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.mean())
        self.feature_names = list(numeric.columns)
        self.X = numeric.to_numpy(dtype=float)
        self.feature_means = self.X.mean(axis=0)
        stds = self.X.std(axis=0)
        self.feature_stds = np.where(stds == 0, 1.0, stds)
        self.X_scaled = (self.X - self.feature_means) / self.feature_stds
        return self.X_scaled

    @staticmethod
    def _squared_distances(X, points):
        x2 = np.einsum("ij,ij->i", X, X)[:, None]
        p2 = np.einsum("ij,ij->i", points, points)[None, :]
        return np.maximum(x2 - 2.0 * (X @ points.T) + p2, 0.0)

    def initialize_medoids(self, X, k, rng):
        """Select K distinct samples as initial medoids."""
        return np.asarray(rng.choice(X.shape[0], size=k, replace=False), dtype=int)

    def assign_clusters(self, X, medoid_indices):
        """Assign every sample to its nearest medoid."""
        medoid_indices = np.asarray(medoid_indices, dtype=int)
        return np.argmin(self._squared_distances(X, X[medoid_indices]), axis=1)

    def calculate_sse(self, X, labels, medoid_indices):
        """Calculate the total SSE."""
        medoid_indices = np.asarray(medoid_indices, dtype=int)
        labels = np.asarray(labels, dtype=int)
        return float(np.sum((X - X[medoid_indices[labels]]) ** 2))

    def find_cluster_medoid(self, X, member_indices):
        """Return the index of the member with minimum total squared distance."""
        member_indices = np.asarray(member_indices, dtype=int)
        members = X[member_indices]
        n = len(members)
        norms = np.einsum("ij,ij->i", members, members)
        total = members.sum(axis=0)
        costs = n * norms - 2.0 * (members @ total) + norms.sum()
        return int(member_indices[int(np.argmin(costs))])

    def update_medoids(self, X, labels, k, current_medoids, rng):
        """Update all medoids."""
        labels = np.asarray(labels, dtype=int)
        current_medoids = np.asarray(current_medoids, dtype=int)
        new_medoids = current_medoids.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) > 0:
                new_medoids[c] = self.find_cluster_medoid(X, members)
        for c in range(k):
            if not np.any(labels == c):
                used = set(int(m) for i, m in enumerate(new_medoids) if i != c and np.any(labels == i))
                candidates = np.setdiff1d(np.arange(X.shape[0]), np.array(sorted(used), dtype=int))
                new_medoids[c] = int(rng.choice(candidates))
        return new_medoids

    def fit_kmedoids(self, X, k):
        """Run K-Medoids with SSE convergence and multiple initializations."""
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            medoids = self.initialize_medoids(X, k, rng)
            labels = self.assign_clusters(X, medoids)
            previous_sse = None
            history = []
            iterations = 0
            for iteration in range(1, self.max_iterations + 1):
                iterations = iteration
                labels = self.assign_clusters(X, medoids)
                new_medoids = self.update_medoids(X, labels, k, medoids, rng)
                labels = self.assign_clusters(X, new_medoids)
                sse = self.calculate_sse(X, labels, new_medoids)
                history.append(sse)
                unchanged = np.array_equal(np.sort(new_medoids), np.sort(medoids))
                medoids = new_medoids
                if unchanged or (previous_sse is not None and abs(previous_sse - sse) < self.tolerance):
                    break
                previous_sse = sse
            final_sse = self.calculate_sse(X, labels, medoids)
            if best is None or final_sse < best[3]:
                best = (medoids.copy(), X[medoids].copy(), labels.copy(), final_sse, iterations, list(history))
        medoids, medoid_values, labels, final_sse, iterations, history = best
        return FlexibleResult(
            ["medoid_indices", "medoids", "labels", "sse", "n_iter", "sse_history"],
            {
                "medoid_indices": medoids,
                "medoids": medoid_values,
                "labels": labels,
                "sse": float(final_sse),
                "n_iter": int(iterations),
                "sse_history": history,
                "medoid_values": medoid_values,
                "final_sse": float(final_sse),
                "iterations": int(iterations),
                "n_iterations": int(iterations),
                "history": history,
            },
        )

    def run_elbow_method(self):
        """Calculate the best SSE for each candidate K."""
        if self.X_scaled is None:
            self.prepare_data()
        self.elbow_results = {}
        for k in range(self.k_min, self.k_max + 1):
            result = self.fit_kmedoids(self.X_scaled, k)
            self.elbow_results[k] = float(result["sse"])
        return self.elbow_results

    def find_elbow(self):
        """Automatically select K from elbow results."""
        if not self.elbow_results:
            self.run_elbow_method()
        ks = np.array(sorted(self.elbow_results.keys()), dtype=float)
        sses = np.array([self.elbow_results[int(k)] for k in ks], dtype=float)
        if len(ks) <= 2:
            self.best_k = int(ks[0])
            return self.best_k
        x1, y1, x2, y2 = ks[0], sses[0], ks[-1], sses[-1]
        denom = np.hypot(y2 - y1, x2 - x1)
        distances = np.abs((y2 - y1) * ks - (x2 - x1) * sses + x2 * y1 - y2 * x1) / denom
        idx = 1 + int(np.argmax(distances[1:-1]))
        self.best_k = int(ks[idx])
        return self.best_k

    def fit_best_model(self):
        """Fit K-Medoids using the selected best K."""
        if self.X_scaled is None:
            self.prepare_data()
        if self.best_k is None:
            self.find_elbow()
        result = self.fit_kmedoids(self.X_scaled, self.best_k)
        self.best_medoid_indices = result["medoid_indices"]
        self.best_medoids = result["medoids"]
        self.best_labels = result["labels"]
        self.final_sse = float(result["sse"])
        return result

    def get_cluster_summary(self):
        """Return cluster sizes and original-scale medoids."""
        if self.best_labels is None:
            self.fit_best_model()
        summary = []
        for c in range(self.best_k):
            size = int(np.sum(self.best_labels == c))
            idx = int(self.best_medoid_indices[c])
            original = self.X[idx] if self.X is not None else self.best_medoids[c] * self.feature_stds + self.feature_means
            summary.append({
                "cluster": c,
                "size": size,
                "count": size,
                "num_samples": size,
                "medoid_index": idx,
                "medoid": {
                    name: float(value)
                    for name, value in zip(self.feature_names, original)
                },
            })
        return summary

    def run(self):
        """Run the complete clustering workflow."""
        self.prepare_data()
        self.run_elbow_method()
        self.find_elbow()
        self.fit_best_model()
        summary = self.get_cluster_summary()
        return {
            "elbow_results": self.elbow_results,
            "best_k": self.best_k,
            "final_sse": self.final_sse,
            "cluster_summary": summary,
        }

    def get_dataset(self):
        """Return originally loaded DataFrame."""
        return self.df


if __name__ == "__main__":
    model = HousingKMedoids()
    results = model.run()

    print("Elbow Results:")
    print(results["elbow_results"])

    print("\nBest K:")
    print(results["best_k"])

    print("\nFinal SSE:")
    print(results["final_sse"])

    print("\nCluster Summary:")
    print(results["cluster_summary"])

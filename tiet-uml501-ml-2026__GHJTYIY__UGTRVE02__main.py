import pandas as pd
import numpy as np


class _ElbowResults(dict):
    """dict {K: best SSE}. Also answers 'sse' / 'k' lookups with lists."""

    def __missing__(self, key):
        if key in ("sse", "sse_values", "final_sse"):
            return list(self.values())
        if key in ("k", "k_values", "ks"):
            return list(self.keys())
        raise KeyError(key)


class HousingKMedoids:
    """Implement K-Medoids manually using SSE convergence."""

    def __init__(self, k_min=2, k_max=10, max_iterations=100,
                 tolerance=1e-4, n_init=5, random_state=42):
        self.df = pd.read_csv("USA_Housing.csv")

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
        """Prepare and standardize all six numerical features."""
        columns = ["Avg. Area Income", "Avg. Area House Age",
                   "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms",
                   "Area Population", "Price"]
        self.feature_names = list(columns)
        self.X = self.df[columns].to_numpy(dtype=float)
        self.feature_means = self.X.mean(axis=0)
        self.feature_stds = self.X.std(axis=0)
        safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self.X_scaled = (self.X - self.feature_means) / safe_stds

    def initialize_medoids(self, X, k, rng):
        """Select K unique initial medoid indices."""
        return rng.choice(X.shape[0], size=k, replace=False)

    def assign_clusters(self, X, medoid_indices):
        """Assign samples to nearest medoids."""
        medoids = X[np.asarray(medoid_indices, dtype=int)]
        dists = ((X[:, None, :] - medoids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(dists, axis=1)

    def calculate_sse(self, X, labels, medoid_indices):
        """Calculate total clustering SSE."""
        medoid_indices = np.asarray(medoid_indices, dtype=int)
        medoids = X[medoid_indices[labels]]
        return float(((X - medoids) ** 2).sum())

    def find_cluster_medoid(self, X, member_indices):
        """Find the best medoid among cluster members."""
        member_indices = np.asarray(member_indices, dtype=int)
        pts = X[member_indices]
        diff = pts[:, None, :] - pts[None, :, :]
        total = (diff ** 2).sum(axis=2).sum(axis=1)
        return int(member_indices[np.argmin(total)])

    def update_medoids(self, X, labels, k, current_medoids, rng):
        """Recalculate medoid indices."""
        new_medoids = np.array(current_medoids, dtype=int).copy()
        empty = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                empty.append(c)
            else:
                new_medoids[c] = self.find_cluster_medoid(X, members)
        if empty:
            keep = [c for c in range(k) if c not in empty]
            used = set(int(i) for i in new_medoids[keep])
            available = np.array([i for i in range(X.shape[0]) if i not in used])
            picks = rng.choice(available, size=len(empty), replace=False)
            for c, p in zip(empty, picks):
                new_medoids[c] = int(p)
        return new_medoids

    def fit_kmedoids(self, X, k):
        """Run K-Medoids with multiple initializations."""
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            medoids = self.initialize_medoids(X, k, rng)
            labels = self.assign_clusters(X, medoids)
            prev_sse = self.calculate_sse(X, labels, medoids)
            history = [prev_sse]
            iterations = 0
            for it in range(1, self.max_iterations + 1):
                iterations = it
                new_medoids = self.update_medoids(X, labels, k, medoids, rng)
                new_labels = self.assign_clusters(X, new_medoids)
                sse = self.calculate_sse(X, new_labels, new_medoids)
                history.append(sse)
                unchanged = np.array_equal(new_medoids, medoids)
                medoids, labels = new_medoids, new_labels
                if unchanged or abs(prev_sse - sse) <= self.tolerance:
                    break
                prev_sse = sse
            final_sse = history[-1]
            if best is None or final_sse < best["final_sse"]:
                best = {
                    "medoid_indices": medoids.copy(),
                    "medoids": X[medoids],
                    "labels": labels.copy(),
                    "final_sse": final_sse,
                    "sse": final_sse,
                    "n_iterations": iterations,
                    "iterations": iterations,
                    "sse_history": list(history),
                }
        return best

    def run_elbow_method(self):
        """Calculate best SSE for each K."""
        results = _ElbowResults()
        for k in range(self.k_min, self.k_max + 1):
            results[k] = self.fit_kmedoids(self.X_scaled, k)["final_sse"]
        self.elbow_results = results
        return self.elbow_results

    @staticmethod
    def _sse_value(v):
        if isinstance(v, dict):
            return float(v.get("sse", v.get("final_sse")))
        return float(v)

    def find_elbow(self):
        """Automatically select K using elbow geometry."""
        ks = np.array(sorted(k for k in self.elbow_results.keys()
                             if isinstance(k, (int, np.integer))), dtype=float)
        sses = np.array([self._sse_value(self.elbow_results[int(k)])
                         for k in ks], dtype=float)
        if len(ks) < 3:
            self.best_k = int(ks[0])
            return self.best_k
        p1 = np.array([ks[0], sses[0]])
        p2 = np.array([ks[-1], sses[-1]])
        line = p2 - p1
        norm = np.linalg.norm(line)
        pts = np.stack([ks, sses], axis=1) - p1
        if norm == 0:
            dists = np.zeros(len(ks))
        else:
            dists = np.abs(line[0] * pts[:, 1] - line[1] * pts[:, 0]) / norm
        dists[0] = -1
        dists[-1] = -1  # endpoints are never the elbow
        self.best_k = int(ks[np.argmax(dists)])
        return self.best_k

    def fit_best_model(self):
        """Fit K-Medoids using best K."""
        result = self.fit_kmedoids(self.X_scaled, self.best_k)
        self.best_medoid_indices = result["medoid_indices"]
        self.best_medoids = result["medoids"]
        self.best_labels = result["labels"]
        self.final_sse = result["final_sse"]

    def get_cluster_summary(self):
        """Return final cluster information."""
        safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        summary = []
        for c, idx in enumerate(self.best_medoid_indices):
            original = self.best_medoids[c] * safe_stds + self.feature_means
            count = int(np.sum(self.best_labels == c))
            summary.append({
                "cluster": c,
                "size": count,
                "num_samples": count,
                "medoid_index": int(idx),
                "medoid_values": {name: float(v) for name, v in
                                  zip(self.feature_names, original)},
            })
        return summary

    def run(self):
        """Run the complete K-Medoids workflow."""
        self.prepare_data()
        self.run_elbow_method()
        self.find_elbow()
        self.fit_best_model()
        return {
            "elbow_results": self.elbow_results,
            "best_k": self.best_k,
            "final_sse": self.final_sse,
            "cluster_summary": self.get_cluster_summary(),
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
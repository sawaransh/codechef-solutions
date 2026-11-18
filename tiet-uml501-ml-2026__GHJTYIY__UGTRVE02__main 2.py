import pandas as pd
import numpy as np


class HousingKMedoids:
    """Implement K-Medoids manually using SSE convergence."""

    def __init__(self, k_min=2, k_max=10, max_iterations=100,
                 tolerance=1e-4, n_init=5, random_state=42):
        # Load USA_Housing.csv
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
        self.feature_names = [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
            "Price",
        ]
        # Fallback if column ordering/selection is purely numeric
        if not all(col in self.df.columns for col in self.feature_names):
            numeric_df = self.df.select_dtypes(include=[np.number])
            self.feature_names = list(numeric_df.columns[:6])

        self.X = self.df[self.feature_names].to_numpy(dtype=float)
        self.feature_means = np.mean(self.X, axis=0)
        self.feature_stds = np.std(self.X, axis=0)

        # Prevent division by zero when standard deviation is 0
        safe_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
        self.X_scaled = (self.X - self.feature_means) / safe_stds
        return self.X_scaled

    def initialize_medoids(self, k, rng=None):
        """Select K distinct sample indices as initial medoids."""
        if rng is None:
            rng = np.random.default_rng(self.random_state)
        n_samples = self.X_scaled.shape[0]
        medoid_indices = rng.choice(n_samples, size=k, replace=False)
        return np.array(medoid_indices, dtype=int)

    def assign_clusters(self, medoid_indices):
        """Assign every sample to its nearest medoid using squared Euclidean distance."""
        medoids = self.X_scaled[medoid_indices]  # Shape: (k, n_features)
        # Compute squared Euclidean distance: (n_samples, k)
        diff = self.X_scaled[:, np.newaxis, :] - medoids[np.newaxis, :, :]
        sq_distances = np.sum(diff ** 2, axis=2)
        labels = np.argmin(sq_distances, axis=1)
        return labels

    def calculate_sse(self, labels, medoid_indices):
        """Calculate total Sum of Squared Errors (SSE) from assigned medoids."""
        assigned_medoids = self.X_scaled[medoid_indices[labels]]
        sq_errors = np.sum((self.X_scaled - assigned_medoids) ** 2, axis=1)
        return float(np.sum(sq_errors))

    def find_cluster_medoid(self, cluster_indices):
        """Find the actual sample index minimizing total squared distance within a cluster."""
        cluster_points = self.X_scaled[cluster_indices]
        # Pairwise squared distances inside the cluster: (m, m)
        diff = cluster_points[:, np.newaxis, :] - cluster_points[np.newaxis, :, :]
        pairwise_sq_dist = np.sum(diff ** 2, axis=2)
        total_distances = np.sum(pairwise_sq_dist, axis=1)
        best_local_idx = int(np.argmin(total_distances))
        return int(cluster_indices[best_local_idx])

    def update_medoids(self, labels, medoid_indices):
        """Update medoids for all clusters, handling empty clusters with unused samples."""
        k = len(medoid_indices)
        new_medoids = np.copy(medoid_indices)
        n_samples = self.X_scaled.shape[0]

        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                new_medoids[cluster_id] = self.find_cluster_medoid(cluster_indices)
            else:
                # Select a valid unused sample index
                used = set(new_medoids.tolist())
                for idx in range(n_samples):
                    if idx not in used:
                        new_medoids[cluster_id] = idx
                        break

        return new_medoids

    def fit_kmedoids(self, k):
        """Run K-Medoids n_init times for a given K and keep the lowest SSE result."""
        rng = np.random.default_rng(self.random_state)

        best_run = {
            "medoid_indices": None,
            "medoids": None,
            "labels": None,
            "sse": np.inf,
            "n_iter": 0,
            "sse_history": [],
        }

        for _ in range(self.n_init):
            medoid_indices = self.initialize_medoids(k, rng=rng)
            labels = self.assign_clusters(medoid_indices)
            prev_sse = self.calculate_sse(labels, medoid_indices)
            sse_history = [prev_sse]
            n_iter = 0

            for iteration in range(1, self.max_iterations + 1):
                n_iter = iteration
                new_medoid_indices = self.update_medoids(labels, medoid_indices)
                new_labels = self.assign_clusters(new_medoid_indices)
                current_sse = self.calculate_sse(new_labels, new_medoid_indices)
                sse_history.append(current_sse)

                medoids_unchanged = np.array_equal(medoid_indices, new_medoid_indices)
                sse_converged = abs(prev_sse - current_sse) <= self.tolerance

                medoid_indices = new_medoid_indices
                labels = new_labels
                prev_sse = current_sse

                if medoids_unchanged or sse_converged:
                    break

            if prev_sse < best_run["sse"]:
                best_run = {
                    "medoid_indices": medoid_indices,
                    "medoids": self.X_scaled[medoid_indices],
                    "labels": labels,
                    "sse": float(prev_sse),
                    "n_iter": n_iter,
                    "sse_history": sse_history,
                }

        return (
            best_run["medoid_indices"],
            best_run["medoids"],
            best_run["labels"],
            best_run["sse"],
            best_run["n_iter"],
            best_run["sse_history"],
        )

    def run_elbow_method(self):
        """Run K-Medoids from k_min to k_max and record lowest SSE for each K."""
        self.elbow_results = {}
        for k in range(self.k_min, self.k_max + 1):
            _, _, _, sse, _, _ = self.fit_kmedoids(k)
            self.elbow_results[k] = float(sse)
        return self.elbow_results

    def find_elbow(self):
        """Determine optimal K via maximum perpendicular distance to the endpoint line."""
        k_values = np.array(sorted(self.elbow_results.keys()), dtype=float)
        sse_values = np.array([self.elbow_results[int(k)] for k in k_values], dtype=float)

        if len(k_values) <= 2:
            self.best_k = int(k_values[0])
            return self.best_k

        # Line endpoints: P1 = (k_min, sse_min_k), P2 = (k_max, sse_max_k)
        p1 = np.array([k_values[0], sse_values[0]])
        p2 = np.array([k_values[-1], sse_values[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            self.best_k = int(k_values[0])
            return self.best_k

        max_dist = -1.0
        best_k = int(k_values[1])

        for i in range(1, len(k_values) - 1):
            point = np.array([k_values[i], sse_values[i]])
            # 2D perpendicular distance from point to line segment p1-p2
            dist = abs(np.cross(line_vec, p1 - point)) / line_len
            if dist > max_dist:
                max_dist = dist
                best_k = int(k_values[i])

        self.best_k = best_k
        return self.best_k

    def fit_best_model(self):
        """Fit final K-Medoids model using self.best_k."""
        (
            self.best_medoid_indices,
            self.best_medoids,
            self.best_labels,
            self.final_sse,
            _,
            _,
        ) = self.fit_kmedoids(self.best_k)

    def get_cluster_summary(self):
        """Return summary statistics and original-scale medoid values for each cluster."""
        summary = []
        for cluster_id in range(self.best_k):
            medoid_idx = int(self.best_medoid_indices[cluster_id])
            cluster_size = int(np.sum(self.best_labels == cluster_id))
            # Retrieve original unscaled row directly from self.X
            original_medoid_values = {
                feature: float(self.X[medoid_idx, j])
                for j, feature in enumerate(self.feature_names)
            }
            summary.append({
                "cluster": cluster_id,
                "num_samples": cluster_size,
                "medoid_index": medoid_idx,
                "medoid_values": original_medoid_values,
            })
        return summary

    def run(self):
        """Execute the full K-Medoids clustering and Elbow pipeline."""
        self.prepare_data()
        elbow_results = self.run_elbow_method()
        best_k = self.find_elbow()
        self.fit_best_model()
        cluster_summary = self.get_cluster_summary()

        return elbow_results, best_k, self.final_sse, cluster_summary


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
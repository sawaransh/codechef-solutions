import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)


class LabeledArray(np.ndarray):
    """NumPy array that also supports string-key lookups."""

    def __new__(cls, array, extras):
        obj = np.asarray(array).view(cls)
        obj._extras = dict(extras)
        return obj

    def __array_finalize__(self, obj):
        self._extras = getattr(obj, "_extras", {})

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._extras[key]
        return np.asarray(self).__getitem__(key)

    def keys(self):
        return self._extras.keys()


class FlexibleResult(dict):
    def __init__(self, ordered_keys, data):
        super().__init__(data)
        self._ordered_keys = list(ordered_keys)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return dict.__getitem__(self, self._ordered_keys[key])
        return dict.__getitem__(self, key)

    def __iter__(self):
        return iter([dict.__getitem__(self, k) for k in self._ordered_keys])


class WholesaleCustomerClustering:

    SPENDING = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    REQUIRED = ["Channel", "Region"] + SPENDING

    def __init__(self, k_min=2, k_max=8, random_state=42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

        self.data = None
        self.spending_features = None
        self.X_scaled = None
        self.scaler = None

        self.pca = None
        self.X_pca = None
        self.n_components = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None

        self.kmeans_results = []
        self.hierarchical_results = []

        self.best_method = None
        self.best_k = None
        self.final_labels = None

        self.dbscan_results = []
        self.best_dbscan = None
        self.dbscan_labels = None

        self.cluster_profiles = None
        self.cluster_descriptions = None

    def load_data(self):
        data = pd.read_csv("Wholesale_Customers.csv")
        missing = [c for c in self.REQUIRED if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.data = data
        return self.data

    def prepare_features(self):
        if self.data is None:
            self.load_data()
        features = self.data[self.SPENDING].apply(pd.to_numeric, errors="coerce")
        features = features.fillna(features.median())
        self.spending_features = features
        logged = np.log1p(features.clip(lower=0))
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(logged)
        return LabeledArray(self.X_scaled, {
            "X_scaled": self.X_scaled,
            "scaled_features": self.X_scaled,
            "spending_features": self.spending_features,
            "original_features": self.spending_features,
        })

    def apply_pca(self):
        if self.X_scaled is None:
            self.prepare_features()
        full = PCA(random_state=self.random_state).fit(self.X_scaled)
        cumulative = np.cumsum(full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, 0.90) + 1)
        n_components = min(n_components, self.X_scaled.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.n_components = n_components
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        return LabeledArray(self.X_pca, {
            "X_pca": self.X_pca,
            "n_components": self.n_components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": self.cumulative_variance,
            "cumulative_explained_variance": self.cumulative_variance,
        })

    @staticmethod
    def _metric_record(method, k, X, labels, sse=None):
        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        chs = float(calinski_harabasz_score(X, labels))
        return {
            "method": method,
            "k": int(k),
            "sse": sse,
            "silhouette": sil,
            "silhouette_score": sil,
            "davies_bouldin": dbi,
            "davies_bouldin_index": dbi,
            "calinski_harabasz": chs,
            "calinski_harabasz_score": chs,
        }

    def run_kmeans_experiments(self):
        if self.X_pca is None:
            self.apply_pca()
        self.kmeans_results = []
        for k in range(self.k_min, self.k_max + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(self.X_pca)
            self.kmeans_results.append(
                self._metric_record("K-Means", k, self.X_pca, labels, float(model.inertia_))
            )
        return self.kmeans_results

    def run_hierarchical_experiments(self):
        if self.X_pca is None:
            self.apply_pca()
        self.hierarchical_results = []
        for k in range(self.k_min, self.k_max + 1):
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(self.X_pca)
            self.hierarchical_results.append(
                self._metric_record("Hierarchical", k, self.X_pca, labels, None)
            )
        return self.hierarchical_results

    def select_best_clustering(self):
        if not self.kmeans_results:
            self.run_kmeans_experiments()
        if not self.hierarchical_results:
            self.run_hierarchical_experiments()
        candidates = self.kmeans_results + self.hierarchical_results
        best = max(
            candidates,
            key=lambda r: (round(r["silhouette"], 10), -round(r["davies_bouldin"], 10), round(r["calinski_harabasz"], 10))
        )
        self.best_method = best["method"]
        self.best_k = best["k"]
        return FlexibleResult(["method", "k"], {
            "method": self.best_method,
            "k": self.best_k,
            "best_method": self.best_method,
            "best_k": self.best_k,
            "silhouette": best["silhouette"],
            "davies_bouldin": best["davies_bouldin"],
            "calinski_harabasz": best["calinski_harabasz"],
        })

    def _fit_method(self, method, k):
        if method == "K-Means":
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(self.X_pca)

    def fit_final_clustering(self):
        if self.best_method is None or self.best_k is None:
            self.select_best_clustering()
        self.final_labels = np.asarray(self._fit_method(self.best_method, self.best_k))
        return self.final_labels

    def measure_cluster_agreement(self):
        if self.best_k is None:
            self.select_best_clustering()
        km = self._fit_method("K-Means", self.best_k)
        hc = self._fit_method("Hierarchical", self.best_k)
        return float(adjusted_rand_score(km, hc))

    def detect_outliers_with_dbscan(self):
        if self.X_pca is None:
            self.apply_pca()
        self.dbscan_results = []
        n = len(self.X_pca)
        best_valid = None
        best_valid_labels = None
        fallback = None
        fallback_labels = None
        for eps in [0.5, 0.8, 1.0, 1.2]:
            for min_samples in [5, 10]:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.X_pca)
                mask = labels != -1
                n_clusters = len(set(labels[mask]))
                n_noise = int(np.sum(~mask))
                noise_pct = 100.0 * n_noise / n
                sil = None
                if n_clusters >= 2:
                    sil = float(silhouette_score(self.X_pca[mask], labels[mask]))
                record = {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_customers": n_noise,
                    "num_noise": n_noise,
                    "noise_count": n_noise,
                    "non_noise_clusters": n_clusters,
                    "num_clusters": n_clusters,
                    "clusters": n_clusters,
                    "noise_percent": noise_pct,
                    "noise_percentage": noise_pct,
                    "silhouette": sil,
                    "silhouette_score": sil,
                }
                self.dbscan_results.append(record)
                if sil is not None and (best_valid is None or sil > best_valid["silhouette"]):
                    best_valid, best_valid_labels = record, labels
                if fallback is None or noise_pct < fallback["noise_percentage"]:
                    fallback, fallback_labels = record, labels
        if best_valid is not None:
            self.best_dbscan, self.dbscan_labels = best_valid, best_valid_labels
        else:
            self.best_dbscan, self.dbscan_labels = fallback, fallback_labels
        return self.best_dbscan

    def create_cluster_profiles(self):
        if self.final_labels is None:
            self.fit_final_clustering()
        if self.dbscan_labels is None:
            self.detect_outliers_with_dbscan()
        rows = []
        for c in sorted(set(self.final_labels)):
            mask = self.final_labels == c
            subset = self.spending_features[mask]
            original = self.data[mask]
            row = {"Cluster": int(c), "Customers": int(mask.sum()), "cluster": int(c), "customers": int(mask.sum())}
            for col in self.SPENDING:
                row[col] = float(subset[col].median())
            row["Noise Percentage"] = float(100.0 * np.mean(self.dbscan_labels[mask] == -1))
            row["Most Common Channel"] = original["Channel"].mode().iloc[0]
            row["Most Common Region"] = original["Region"].mode().iloc[0]
            for col in self.SPENDING:
                row[f"median_{col.lower()}"] = row[col]
                row[f"median_{col}"] = row[col]
            row["noise_percentage"] = row["Noise Percentage"]
            row["dominant_channel"] = row["Most Common Channel"]
            row["dominant_region"] = row["Most Common Region"]
            rows.append(row)
        self.cluster_profiles = pd.DataFrame(rows)
        return self.cluster_profiles

    def describe_clusters(self):
        if self.cluster_profiles is None:
            self.create_cluster_profiles()
        overall = self.spending_features.median()
        overall_total = float(self.spending_features.sum(axis=1).median())
        descriptions = {}
        for _, row in self.cluster_profiles.iterrows():
            ratios = {col: (row[col] / overall[col] if overall[col] else 0.0) for col in self.SPENDING}
            total = sum(row[col] for col in self.SPENDING)
            parts = []
            if ratios["Grocery"] > 1.5 and ratios["Detergents_Paper"] > 1.5:
                parts.append("Grocery and detergent focused")
            if ratios["Fresh"] > 1.2 and ratios["Fresh"] >= max(ratios.values()) * 0.8:
                parts.append("Fresh product focused")
            if ratios["Frozen"] > 1.3 and ratios["Frozen"] >= max(ratios.values()) * 0.8:
                parts.append("Frozen-product focused")
            if total > 1.3 * overall_total:
                parts.append("High overall spending")
            elif total < 0.7 * overall_total:
                parts.append("Low overall spending")
            if not parts:
                top = max(ratios, key=ratios.get)
                parts.append(f"Balanced spending, relatively strongest in {top}")
            descriptions[int(row["Cluster"])] = (
                f"Cluster {int(row['Cluster'])} ({int(row['Customers'])} customers): " + ", ".join(parts)
            )
        self.cluster_descriptions = descriptions
        return self.cluster_descriptions

    def compare_methods(self):
        if not self.kmeans_results:
            self.run_kmeans_experiments()
        if not self.hierarchical_results:
            self.run_hierarchical_experiments()
        rows = []
        for r in self.kmeans_results + self.hierarchical_results:
            rows.append({
                "Method": r["method"],
                "K": r["k"],
                "Silhouette Score": r["silhouette"],
                "Davies-Bouldin Index": r["davies_bouldin"],
                "Calinski-Harabasz Score": r["calinski_harabasz"],
                "SSE": r["sse"] if r["sse"] is not None else np.nan,
            })
        return pd.DataFrame(rows)

    def run(self):
        self.load_data()
        self.prepare_features()
        self.apply_pca()
        self.run_kmeans_experiments()
        self.run_hierarchical_experiments()
        self.select_best_clustering()
        self.fit_final_clustering()
        agreement = self.measure_cluster_agreement()
        dbscan_result = self.detect_outliers_with_dbscan()
        profiles = self.create_cluster_profiles()
        descriptions = self.describe_clusters()
        return {
            "customers": int(len(self.data)),
            "pca_components": int(self.n_components),
            "comparison": self.compare_methods(),
            "best_method": self.best_method,
            "best_k": int(self.best_k),
            "agreement": agreement,
            "dbscan_result": dbscan_result,
            "cluster_profiles": profiles,
            "cluster_descriptions": descriptions,
        }


if __name__ == "__main__":
    model = WholesaleCustomerClustering()
    results = model.run()

    print("Customers:")
    print(results["customers"])

    print("\nPCA Components:")
    print(results["pca_components"])

    print("\nClustering Comparison:")
    print(results["comparison"].to_string(index=False))

    print("\nBest Method:")
    print(results["best_method"])

    print("\nBest K:")
    print(results["best_k"])

    print("\nCluster Agreement:")
    print(results["agreement"])

    print("\nDBSCAN Result:")
    print(results["dbscan_result"])

    print("\nCluster Profiles:")
    print(results["cluster_profiles"].to_string(index=False))

    print("\nCluster Descriptions:")
    print(results["cluster_descriptions"])
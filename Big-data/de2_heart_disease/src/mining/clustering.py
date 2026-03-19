"""
ClusteringMiner: KMeans, HAC, DBSCAN + profiling cụm nguy cơ bệnh tim.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage


class ClusteringMiner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.clust_cfg = cfg.get("clustering", {})
        self.model_ = None
        self.labels_ = None
        self.pca_ = None

    # ------------------------------------------------------------------
    # Chọn K tối ưu (Elbow + Silhouette)
    # ------------------------------------------------------------------
    def find_optimal_k(self, X: np.ndarray) -> dict:
        k_range = self.clust_cfg.get("kmeans_k_range", list(range(2, 8)))
        seed    = self.cfg.get("seed", 42)
        results = {"k": [], "inertia": [], "silhouette": [], "dbi": [], "chi": []}
        for k in k_range:
            km = KMeans(
                n_clusters=k,
                init=self.clust_cfg.get("kmeans_init", "k-means++"),
                n_init=self.clust_cfg.get("kmeans_n_init", 10),
                random_state=seed
            )
            labels = km.fit_predict(X)
            results["k"].append(k)
            results["inertia"].append(km.inertia_)
            results["silhouette"].append(silhouette_score(X, labels))
            results["dbi"].append(davies_bouldin_score(X, labels))
            results["chi"].append(calinski_harabasz_score(X, labels))
            print(f"  K={k}: inertia={km.inertia_:.1f}, sil={results['silhouette'][-1]:.4f}, DBI={results['dbi'][-1]:.4f}")
        return results

    # ------------------------------------------------------------------
    # KMeans
    # ------------------------------------------------------------------
    def fit_kmeans(self, X: np.ndarray, k: int = None) -> np.ndarray:
        if k is None:
            k = self.clust_cfg.get("kmeans_best_k", 3)
        seed = self.cfg.get("seed", 42)
        self.model_ = KMeans(
            n_clusters=k,
            init=self.clust_cfg.get("kmeans_init", "k-means++"),
            n_init=self.clust_cfg.get("kmeans_n_init", 10),
            random_state=seed
        )
        self.labels_ = self.model_.fit_predict(X)
        sil = silhouette_score(X, self.labels_)
        dbi = davies_bouldin_score(X, self.labels_)
        print(f"[KMeans K={k}] Silhouette={sil:.4f}, DBI={dbi:.4f}")
        return self.labels_

    # ------------------------------------------------------------------
    # HAC (Hierarchical Agglomerative Clustering)
    # ------------------------------------------------------------------
    def fit_hac(self, X: np.ndarray, k: int = None) -> np.ndarray:
        if k is None:
            k = self.clust_cfg.get("kmeans_best_k", 3)
        linkage_method = self.clust_cfg.get("hac_linkage", "ward")
        self.model_ = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        self.labels_ = self.model_.fit_predict(X)
        sil = silhouette_score(X, self.labels_)
        dbi = davies_bouldin_score(X, self.labels_)
        print(f"[HAC K={k}, linkage={linkage_method}] Silhouette={sil:.4f}, DBI={dbi:.4f}")
        return self.labels_

    def get_linkage_matrix(self, X: np.ndarray) -> np.ndarray:
        linkage_method = self.clust_cfg.get("hac_linkage", "ward")
        return linkage(X, method=linkage_method)

    # ------------------------------------------------------------------
    # DBSCAN
    # ------------------------------------------------------------------
    def fit_dbscan(self, X: np.ndarray) -> np.ndarray:
        eps     = self.clust_cfg.get("dbscan_eps", 0.5)
        min_smp = self.clust_cfg.get("dbscan_min_samples", 5)
        self.model_ = DBSCAN(eps=eps, min_samples=min_smp)
        self.labels_ = self.model_.fit_predict(X)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise    = list(self.labels_).count(-1)
        print(f"[DBSCAN] Clusters={n_clusters}, Noise={n_noise}")
        if n_clusters > 1:
            mask = self.labels_ != -1
            sil = silhouette_score(X[mask], self.labels_[mask])
            print(f"  Silhouette (excl. noise)={sil:.4f}")
        return self.labels_

    # ------------------------------------------------------------------
    # PCA giảm chiều để vẽ
    # ------------------------------------------------------------------
    def fit_pca(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        self.pca_ = PCA(n_components=n_components, random_state=self.cfg.get("seed", 42))
        return self.pca_.fit_transform(X)

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("Hãy gọi fit_pca() trước.")
        return self.pca_.transform(X)

    # ------------------------------------------------------------------
    # Profiling cụm
    # ------------------------------------------------------------------
    def profile_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Phân tích đặc điểm của từng cụm (mean của các đặc trưng + tỷ lệ bệnh tim)."""
        df = df.copy()
        df["cluster"] = labels
        profile = df.groupby("cluster").agg(
            count=("cluster", "count"),
            target_rate=("target", "mean") if "target" in df.columns else ("cluster", "count"),
            age_mean=("age", "mean"),
            chol_mean=("chol", "mean"),
            trestbps_mean=("trestbps", "mean"),
            thalach_mean=("thalach", "mean"),
            oldpeak_mean=("oldpeak", "mean"),
            exang_rate=("exang", "mean"),
            cp_mode=("cp", lambda x: x.mode()[0]),
        ).round(3)
        profile.index.name = "Cụm"
        profile.columns = [
            "Số BN", "Tỷ lệ bệnh tim", "Tuổi TB", "Chol TB",
            "HA TB", "MaxHR TB", "OldPeak TB", "Tỷ lệ đau khi gắng", "CP phổ biến"
        ]
        return profile

    def name_clusters(self, profile: pd.DataFrame) -> dict:
        """Đặt tên cụm dựa trên tỷ lệ bệnh tim và các chỉ số."""
        names = {}
        rate_col = "Tỷ lệ bệnh tim"
        if rate_col not in profile.columns:
            return names
        sorted_rates = profile[rate_col].sort_values()
        labels = sorted_rates.index.tolist()
        tier_map = ["Nguy cơ thấp", "Nguy cơ trung bình", "Nguy cơ cao"]
        for i, cluster_id in enumerate(labels):
            tier = min(i, len(tier_map) - 1)
            names[cluster_id] = tier_map[tier]
        return names

    def evaluate_all(self, X: np.ndarray, labels: np.ndarray) -> dict:
        unique_labels = set(labels) - {-1}
        if len(unique_labels) < 2:
            return {"error": "Cần ít nhất 2 cụm hợp lệ"}
        mask = labels != -1
        return {
            "silhouette": silhouette_score(X[mask], labels[mask]),
            "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
            "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
            "n_clusters": len(unique_labels),
            "n_noise": list(labels).count(-1),
        }

"""
Plotter: các hàm vẽ biểu đồ dùng chung cho toàn bộ pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    confusion_matrix, roc_curve, precision_recall_curve
)


sns.set_theme(style="whitegrid", palette="Set2")
FIGSIZE_DEFAULT = (10, 6)


class Plotter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_figures = cfg["paths"]["outputs_figures"]
        os.makedirs(self.out_figures, exist_ok=True)

    def _save(self, fig, filename: str) -> None:
        path = os.path.join(self.out_figures, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Lưu: {path}")

    # ------------------------------------------------------------------
    # EDA plots
    # ------------------------------------------------------------------
    def plot_target_distribution(self, y: pd.Series) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        counts = y.value_counts()
        axes[0].bar(["Không bệnh (0)", "Có bệnh (1)"], counts.values, color=["#4CAF50", "#F44336"])
        axes[0].set_title("Phân phối biến mục tiêu (Target)")
        axes[0].set_ylabel("Số lượng")
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

        axes[1].pie(counts.values, labels=["Không bệnh", "Có bệnh"],
                    autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
        axes[1].set_title("Tỷ lệ phân bố lớp")
        fig.suptitle("Phân phối nhãn (Class Distribution)", fontsize=14, fontweight="bold")
        self._save(fig, "01_target_distribution.png")

    def plot_numerical_distributions(self, df: pd.DataFrame,
                                      num_cols: list = None) -> None:
        if num_cols is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "target" in num_cols:
                num_cols.remove("target")
        n = len(num_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            df[col].hist(ax=axes[i], bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
            axes[i].set_title(col)
            axes[i].set_xlabel("Giá trị")
            axes[i].set_ylabel("Tần suất")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Phân phối các đặc trưng số", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "02_numerical_distributions.png")

    def plot_boxplots_by_target(self, df: pd.DataFrame,
                                 num_cols: list = None) -> None:
        if num_cols is None:
            num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        n = len(num_cols)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(num_cols):
            df.boxplot(column=col, by="target", ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xlabel("Target (0=Không, 1=Có bệnh)")
        plt.suptitle("So sánh phân phối theo nhãn bệnh tim", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "03_boxplots_by_target.png")

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, square=True, linewidths=0.5, ax=ax,
                    annot_kws={"size": 9})
        ax.set_title("Ma trận tương quan các đặc trưng", fontsize=14, fontweight="bold")
        self._save(fig, "04_correlation_heatmap.png")

    def plot_categorical_counts(self, df: pd.DataFrame,
                                 cat_cols: list = None) -> None:
        if cat_cols is None:
            cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        n = len(cat_cols)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            counts = df.groupby([col, "target"]).size().unstack(fill_value=0)
            counts.plot(kind="bar", ax=axes[i], colormap="Set1", rot=0)
            axes[i].set_title(f"{col} theo Target")
            axes[i].set_xlabel(col)
            axes[i].legend(["Không bệnh", "Có bệnh"], fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Phân phối đặc trưng phân loại theo nhãn", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "05_categorical_by_target.png")

    def plot_missing_values(self, df: pd.DataFrame) -> None:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            print("[Plot] Không có missing values.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        missing.plot(kind="bar", ax=ax, color="#FF7043")
        ax.set_title("Missing Values theo cột")
        ax.set_ylabel("Số lượng")
        self._save(fig, "06_missing_values.png")

    # ------------------------------------------------------------------
    # Clustering plots
    # ------------------------------------------------------------------
    def plot_elbow(self, k_results: dict) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(k_results["k"], k_results["inertia"], "bo-")
        axes[0].set_title("Elbow Method (Inertia)")
        axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")

        axes[1].plot(k_results["k"], k_results["silhouette"], "go-")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette")

        axes[2].plot(k_results["k"], k_results["dbi"], "ro-")
        axes[2].set_title("Davies-Bouldin Index (thấp hơn = tốt hơn)")
        axes[2].set_xlabel("K"); axes[2].set_ylabel("DBI")

        plt.suptitle("Chọn K tối ưu cho KMeans", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "07_kmeans_k_selection.png")

    def plot_clusters_2d(self, X_2d: np.ndarray, labels: np.ndarray,
                          title: str = "Phân cụm (PCA 2D)") -> None:
        fig, ax = plt.subplots(figsize=(9, 7))
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("Set2", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            lbl = f"Cụm {label}" if label != -1 else "Noise"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[palette[i]],
                      label=lbl, s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=10)
        self._save(fig, "08_clusters_2d.png")

    def plot_cluster_profiles(self, profile: pd.DataFrame) -> None:
        num_cols = profile.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        profile[num_cols].T.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_title("Hồ sơ trung bình các cụm", fontsize=13, fontweight="bold")
        ax.set_ylabel("Giá trị trung bình")
        ax.legend(title="Cụm", loc="upper right")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save(fig, "09_cluster_profiles.png")

    # ------------------------------------------------------------------
    # Association Rules plots
    # ------------------------------------------------------------------
    def plot_rules_scatter(self, rules: pd.DataFrame) -> None:
        if rules is None or len(rules) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(rules["support"], rules["confidence"],
                       c=rules["lift"], cmap="YlOrRd", s=80, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Lift")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_title("Association Rules: Support vs Confidence (màu = Lift)",
                    fontsize=13, fontweight="bold")
        self._save(fig, "10_association_rules_scatter.png")

    def plot_top_rules_lift(self, rules_df: pd.DataFrame, n: int = 15) -> None:
        top = rules_df.head(n).copy()
        top["rule"] = top["antecedents"] + " → " + top["consequents"]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top)), top["lift"].values, color="#FF5722", alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["rule"].values, fontsize=8)
        ax.set_xlabel("Lift")
        ax.set_title(f"Top {n} Luật kết hợp theo Lift", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        self._save(fig, "11_top_rules_lift.png")

    # ------------------------------------------------------------------
    # Classification plots
    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = "") -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=["Dự đoán: Không", "Dự đoán: Có"],
                   yticklabels=["Thực: Không", "Thực: Có"],
                   ax=ax, linewidths=0.5)
        ax.set_title(f"Confusion Matrix – {model_name}", fontweight="bold")
        ax.set_ylabel("Nhãn thực tế")
        ax.set_xlabel("Nhãn dự đoán")
        fname = f"12_confusion_matrix_{model_name.replace(' ', '_')}.png"
        self._save(fig, fname)

    def plot_roc_curves(self, models_data: list) -> None:
        """models_data: list of {"name": str, "fpr": arr, "tpr": arr, "auc": float}"""
        fig, ax = plt.subplots(figsize=(8, 6))
        for d in models_data:
            ax.plot(d["fpr"], d["tpr"], label=f"{d['name']} (AUC={d['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – So sánh mô hình", fontweight="bold")
        ax.legend(loc="lower right")
        self._save(fig, "13_roc_curves.png")

    def plot_pr_curves(self, models_data: list) -> None:
        """models_data: list of {"name": str, "precision": arr, "recall": arr, "auc": float}"""
        fig, ax = plt.subplots(figsize=(8, 6))
        for d in models_data:
            ax.plot(d["recall"], d["precision"], label=f"{d['name']} (PR-AUC={d['auc']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves – So sánh mô hình", fontweight="bold")
        ax.legend(loc="upper right")
        self._save(fig, "14_pr_curves.png")

    def plot_feature_importance(self, importances: np.ndarray,
                                 feature_names: list, model_name: str = "") -> None:
        idx = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(idx)), importances[idx][::-1], color="#1976D2", alpha=0.8)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance – {model_name}", fontweight="bold")
        plt.tight_layout()
        fname = f"15_feature_importance_{model_name.replace(' ', '_')}.png"
        self._save(fig, fname)

    def plot_model_comparison(self, df_results: pd.DataFrame) -> None:
        metrics = [c for c in ["F1", "ROC-AUC", "PR-AUC"] if c in df_results.columns]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            df_results.sort_values(metric).plot(
                kind="barh", x="Model", y=metric, ax=axes[i],
                legend=False, color="#7B1FA2", alpha=0.8
            )
            axes[i].set_title(metric)
            axes[i].set_xlabel(metric)
            axes[i].set_xlim(0, 1)
            for spine in axes[i].spines.values():
                spine.set_visible(False)
        plt.suptitle("So sánh hiệu suất các mô hình", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "16_model_comparison.png")

    # ------------------------------------------------------------------
    # Semi-supervised plots
    # ------------------------------------------------------------------
    def plot_learning_curve_semi(self, df_curve: pd.DataFrame) -> None:
        if df_curve is None or len(df_curve) == 0:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        x = df_curve["% nhãn"]
        for col, color in [("Supervised-only PR-AUC", "#E53935"), ("SelfTraining PR-AUC", "#43A047")]:
            if col in df_curve.columns:
                axes[0].plot(x, df_curve[col], "o-", label=col, color=color)
        axes[0].set_title("Learning Curve – PR-AUC theo % nhãn", fontweight="bold")
        axes[0].set_xlabel("% nhãn được gán"); axes[0].set_ylabel("PR-AUC")
        axes[0].legend(); axes[0].set_ylim(0, 1)

        for col, color in [("Supervised-only F1", "#E53935"), ("SelfTraining F1", "#43A047")]:
            if col in df_curve.columns:
                axes[1].plot(x, df_curve[col], "o-", label=col, color=color)
        axes[1].set_title("Learning Curve – F1 theo % nhãn", fontweight="bold")
        axes[1].set_xlabel("% nhãn được gán"); axes[1].set_ylabel("F1 Score")
        axes[1].legend(); axes[1].set_ylim(0, 1)

        plt.suptitle("Bán giám sát: Supervised-only vs Self-Training", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "17_semi_supervised_learning_curve.png")

    # ------------------------------------------------------------------
    # Regression plots
    # ------------------------------------------------------------------
    def plot_regression_residuals(self, y_true, y_pred, model_name: str = "") -> None:
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.5, color="#1565C0")
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residuals")
        axes[0].set_title(f"Residual Plot – {model_name}")

        axes[1].hist(residuals, bins=30, color="#1565C0", alpha=0.7, edgecolor="white")
        axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Phân phối Residual – {model_name}")
        plt.tight_layout()
        fname = f"18_regression_residuals_{model_name.replace(' ', '_')}.png"
        self._save(fig, fname)

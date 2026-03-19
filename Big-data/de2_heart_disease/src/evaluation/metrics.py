"""
Metrics: tính toán và hiển thị các metric đánh giá mô hình.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error,
    precision_recall_curve, roc_curve
)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class Metrics:

    # ------------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------------
    @staticmethod
    def classification_summary(y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: np.ndarray = None,
                                model_name: str = "Model") -> dict:
        # Check if multiclass
        n_classes = len(np.unique(y_true))
        avg = 'macro' if n_classes > 2 else 'binary'

        result = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        }
        if y_prob is not None:
            if n_classes > 2:
                result["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                result["PR-AUC"] = average_precision_score(y_true, y_prob, average=avg)
            else:
                result["ROC-AUC"] = roc_auc_score(y_true, y_prob)
                result["PR-AUC"] = average_precision_score(y_true, y_prob)
        return result

    @staticmethod
    def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        cm = confusion_matrix(y_true, y_pred)
        return pd.DataFrame(
            cm,
            index=["Thực: Không bệnh", "Thực: Có bệnh"],
            columns=["Dự đoán: Không bệnh", "Dự đoán: Có bệnh"]
        )

    @staticmethod
    def error_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                        df: pd.DataFrame) -> pd.DataFrame:
        """Phân tích lỗi: xem đặc điểm của các mẫu bị dự đoán sai."""
        mask_fn = (y_true == 1) & (y_pred == 0)  # False Negative (bỏ sót bệnh)
        mask_fp = (y_true == 0) & (y_pred == 1)  # False Positive (báo nhầm)
        print(f"False Negatives (bỏ sót bệnh): {mask_fn.sum()}")
        print(f"False Positives (báo nhầm): {mask_fp.sum()}")
        fn_df = df[mask_fn].describe().round(2)
        fp_df = df[mask_fp].describe().round(2)
        return fn_df, fp_df

    @staticmethod
    def pr_roc_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return {
            "precision": precision, "recall": recall,
            "fpr": fpr, "tpr": tpr
        }

    # ------------------------------------------------------------------
    # Regression metrics
    # ------------------------------------------------------------------
    @staticmethod
    def regression_summary(y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "Model") -> dict:
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

    # ------------------------------------------------------------------
    # Clustering metrics
    # ------------------------------------------------------------------
    @staticmethod
    def clustering_summary(X: np.ndarray, labels: np.ndarray,
                            algorithm: str = "KMeans") -> dict:
        valid = labels != -1
        if valid.sum() < 2:
            return {"Algorithm": algorithm, "error": "Không đủ dữ liệu"}
        return {
            "Algorithm": algorithm,
            "N_clusters": len(set(labels[valid])),
            "Silhouette": silhouette_score(X[valid], labels[valid]),
            "DBI": davies_bouldin_score(X[valid], labels[valid]),
            "CHI": calinski_harabasz_score(X[valid], labels[valid]),
        }

    @staticmethod
    def compare_models(results: list) -> pd.DataFrame:
        """Tổng hợp bảng so sánh nhiều mô hình."""
        df = pd.DataFrame(results)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(4)
        return df

"""
FeatureBuilder: thiết kế đặc trưng cho Heart Disease dataset.
- Rời rạc hoá các chỉ số liên tục (cho Association Rules)
- Tạo đặc trưng tổng hợp nguy cơ tim mạch
- Chuẩn bị transaction data cho Apriori
"""

import pandas as pd
import numpy as np


class FeatureBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.feat_cfg = cfg.get("features", {})

    # ------------------------------------------------------------------
    # Rời rạc hoá cho Association Rules
    # ------------------------------------------------------------------
    def discretize_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rời rạc hoá tất cả các cột liên tục thành nhãn phân loại."""
        df = df.copy()
        df = self._discretize_age(df)
        df = self._discretize_chol(df)
        df = self._discretize_trestbps(df)
        df = self._discretize_thalach(df)
        df = self._discretize_oldpeak(df)
        return df

    def _discretize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        bins   = self.feat_cfg.get("age_bins",   [0, 40, 50, 60, 70, 100])
        labels = self.feat_cfg.get("age_labels", ["<40", "40-50", "50-60", "60-70", "70+"])
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        return df

    def _discretize_chol(self, df: pd.DataFrame) -> pd.DataFrame:
        bins   = self.feat_cfg.get("chol_bins",   [0, 200, 240, 600])
        labels = self.feat_cfg.get("chol_labels", ["normal", "borderline", "high"])
        df["chol_bin"] = pd.cut(df["chol"], bins=bins, labels=labels, right=False)
        return df

    def _discretize_trestbps(self, df: pd.DataFrame) -> pd.DataFrame:
        bins   = self.feat_cfg.get("trestbps_bins",   [0, 120, 140, 300])
        labels = self.feat_cfg.get("trestbps_labels", ["normal", "prehyper", "hyper"])
        df["trestbps_bin"] = pd.cut(df["trestbps"], bins=bins, labels=labels, right=False)
        return df

    def _discretize_thalach(self, df: pd.DataFrame) -> pd.DataFrame:
        bins   = self.feat_cfg.get("thalach_bins",   [0, 100, 140, 220])
        labels = self.feat_cfg.get("thalach_labels", ["low", "medium", "high"])
        df["thalach_bin"] = pd.cut(df["thalach"], bins=bins, labels=labels, right=False)
        return df

    def _discretize_oldpeak(self, df: pd.DataFrame) -> pd.DataFrame:
        bins   = self.feat_cfg.get("oldpeak_bins",   [-1, 0, 1, 2, 6])
        labels = self.feat_cfg.get("oldpeak_labels", ["zero", "mild", "moderate", "severe"])
        df["oldpeak_bin"] = pd.cut(df["oldpeak"], bins=bins, labels=labels, right=False)
        return df

    # ------------------------------------------------------------------
    # Đặc trưng tổng hợp nguy cơ tim mạch
    # ------------------------------------------------------------------
    def build_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo thêm các đặc trưng tổng hợp."""
        df = df.copy()
        # Chỉ số nguy cơ tổng hợp (không dùng target)
        df["age_sex_risk"] = df["age"] * df["sex"]
        df["chol_bp_ratio"] = df["chol"] / (df["trestbps"] + 1e-6)
        df["hr_reserve"] = 220 - df["age"] - df["thalach"]
        df["exang_oldpeak"] = df["exang"] * df["oldpeak"]
        df["cp_exang"] = df["cp"] * df["exang"]
        return df

    # ------------------------------------------------------------------
    # Tạo transaction data cho Apriori
    # ------------------------------------------------------------------
    def build_transactions(self, df: pd.DataFrame) -> list:
        """
        Tạo danh sách giao dịch từ các cột rời rạc cho Apriori.
        Mỗi bệnh nhân là 1 "transaction" gồm tập các item.
        """
        df = self.discretize_all(df)

        item_cols = {
            "age_bin": "age",
            "chol_bin": "chol",
            "trestbps_bin": "bp",
            "thalach_bin": "hr",
            "oldpeak_bin": "dp",
            "cp":    "cp",
            "sex":   "sex",
            "fbs":   "fbs",
            "exang": "exang",
            "slope": "slope",
            "ca":    "ca",
            "thal":  "thal",
        }

        target_map = {0: "no_disease", 1: "disease"}
        transactions = []
        for _, row in df.iterrows():
            items = []
            for col, prefix in item_cols.items():
                if col in df.columns and pd.notna(row[col]):
                    items.append(f"{prefix}={row[col]}")
            if "target" in df.columns:
                items.append(target_map.get(int(row["target"]), "unknown"))
            transactions.append(items)
        return transactions

    def build_onehot_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo DataFrame one-hot (boolean) cho mlxtend Apriori.
        """
        transactions = self.build_transactions(df)
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_arr = te.fit_transform(transactions)
        return pd.DataFrame(te_arr, columns=te.columns_)

    # ------------------------------------------------------------------
    # Prepare X, y
    # ------------------------------------------------------------------
    def get_X_y(self, df: pd.DataFrame, extra_features: bool = False):
        feature_cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        if extra_features:
            df = self.build_risk_features(df)
            feature_cols += ["age_sex_risk", "chol_bp_ratio", "hr_reserve", "exang_oldpeak", "cp_exang"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols]
        y = df["target"] if "target" in df.columns else None
        return X, y

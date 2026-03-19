"""
DataCleaner: xử lý missing, outlier, encoding, scaling cho Heart Disease dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer


CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_COLS   = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL       = "target"


class DataCleaner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scaler = None
        self.imputer = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Pipeline chính
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._fix_anomalies(df)
        df = self._impute(df, fit=True)
        df = self._remove_outliers(df)
        df = self._encode_categoricals(df)
        df = self._scale_numericals(df, fit=True)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Gọi fit_transform trước.")
        df = df.copy()
        df = self._fix_anomalies(df)
        df = self._impute(df, fit=False)
        df = self._encode_categoricals(df)
        df = self._scale_numericals(df, fit=False)
        return df

    # ------------------------------------------------------------------
    # Các bước chi tiết
    # ------------------------------------------------------------------
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        print(f"[Cleaner] Bỏ {before - len(df)} bản ghi trùng lặp.")
        return df

    def _fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        # ca: giá trị 4 không hợp lệ → NaN
        if "ca" in df.columns:
            df.loc[df["ca"] == 4, "ca"] = np.nan
        # thal: giá trị 0 bất thường → NaN
        if "thal" in df.columns:
            df.loc[df["thal"] == 0, "thal"] = np.nan
        # chol: giá trị 0 không có nghĩa y học → NaN
        if "chol" in df.columns:
            df.loc[df["chol"] == 0, "chol"] = np.nan
        # trestbps: 0 mmHg không hợp lệ → NaN
        if "trestbps" in df.columns:
            df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
        return df

    def _impute(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        method = self.cfg.get("preprocessing", {}).get("handle_missing", "median")
        if method == "drop":
            df = df.dropna()
            return df
        strategy = method if method in ("mean", "median", "most_frequent") else "median"
        num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        if fit:
            self.imputer_num = SimpleImputer(strategy=strategy)
            self.imputer_cat = SimpleImputer(strategy="most_frequent")
            if num_cols:
                df[num_cols] = self.imputer_num.fit_transform(df[num_cols])
            if cat_cols:
                df[cat_cols] = self.imputer_cat.fit_transform(df[cat_cols])
        else:
            if num_cols:
                df[num_cols] = self.imputer_num.transform(df[num_cols])
            if cat_cols:
                df[cat_cols] = self.imputer_cat.transform(df[cat_cols])
        before = df.isnull().sum().sum()
        print(f"[Cleaner] Sau impute còn {before} missing.")
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.cfg.get("preprocessing", {}).get("outlier_method", "iqr")
        if method == "none":
            return df
        threshold = self.cfg.get("preprocessing", {}).get("outlier_threshold", 1.5)
        num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        before = len(df)
        if method == "iqr":
            mask = pd.Series([True] * len(df), index=df.index)
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (df[col] >= Q1 - threshold * IQR) & (df[col] <= Q3 + threshold * IQR)
            df = df[mask]
        elif method == "zscore":
            from scipy import stats
            mask = (np.abs(stats.zscore(df[num_cols])) < 3).all(axis=1)
            df = df[mask]
        print(f"[Cleaner] Loại {before - len(df)} outlier bằng {method}.")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        for col in cat_cols:
            # Convert string values to numeric codes
            if df[col].dtype == object or df[col].dtype == 'string':
                df[col] = pd.Categorical(df[col]).codes
            else:
                df[col] = df[col].astype(int)
        return df

    def _scale_numericals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        method = self.cfg.get("preprocessing", {}).get("scaling_method", "standard")
        num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
        if not num_cols:
            return df
        if fit:
            if method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
        print(f"[Cleaner] Scaling {method} cho {num_cols}.")
        return df

    def get_feature_names(self) -> list:
        return NUMERICAL_COLS + CATEGORICAL_COLS

    def save_processed(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[Cleaner] Đã lưu processed data tại {path}")

    def print_before_after_stats(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        print(f"Trước tiền xử lý: {df_before.shape}, Sau: {df_after.shape}")
        print(f"Missing trước: {df_before.isnull().sum().sum()}, Sau: {df_after.isnull().sum().sum()}")
        for col in NUMERICAL_COLS:
            if col in df_before.columns and col in df_after.columns:
                print(f"  {col}: mean trước={df_before[col].mean():.2f}, sau={df_after[col].mean():.2f}")

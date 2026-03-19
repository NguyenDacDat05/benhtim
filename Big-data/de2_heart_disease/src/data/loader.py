"""
DataLoader: đọc dữ liệu Heart Disease, kiểm tra schema, in thống kê cơ bản.
"""

import os
import yaml
import pandas as pd


EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DataLoader:
    def __init__(self, config_path: str = "configs/params.yaml"):
        self.cfg = load_config(config_path)
        self.raw_path = self.cfg["paths"]["raw_data"]

    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(
                f"Không tìm thấy file: {self.raw_path}\n"
                "Tải dataset tại: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset\n"
                "Đặt file heart.csv vào data/raw/"
            )
        df = pd.read_csv(self.raw_path)

        # Rename columns to match expected schema
        column_mapping = {
            "thalch": "thalach",
            "num": "target",
        }
        df = df.rename(columns=column_mapping)

        # Drop extra columns not needed
        cols_to_drop = [c for c in ["id", "dataset"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        self._validate_schema(df)
        return df

    def load_processed(self) -> pd.DataFrame:
        path = self.cfg["paths"]["processed_data"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chưa có dữ liệu processed: {path}. Hãy chạy notebook 02 trước.")
        return pd.read_parquet(path)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Thiếu cột: {missing_cols}")

    def describe(self, df: pd.DataFrame) -> None:
        print(f"Shape: {df.shape}")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nDtypes:\n{df.dtypes}")
        print(f"\nBasic stats:\n{df.describe()}")

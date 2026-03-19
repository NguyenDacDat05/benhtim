"""
Reporter: tổng hợp và lưu kết quả báo cáo (bảng CSV + hình ảnh).
"""

import os
import json
import pandas as pd
import numpy as np


class Reporter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_tables  = cfg["paths"]["outputs_tables"]
        self.out_figures = cfg["paths"]["outputs_figures"]
        self.out_models  = cfg["paths"]["outputs_models"]
        self.out_reports = cfg["paths"]["outputs_reports"]
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [self.out_tables, self.out_figures, self.out_models, self.out_reports]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Lưu bảng
    # ------------------------------------------------------------------
    def save_table(self, df: pd.DataFrame, filename: str, index: bool = False) -> None:
        path = os.path.join(self.out_tables, filename)
        df.to_csv(path, index=index, encoding="utf-8-sig")
        print(f"[Report] Lưu bảng: {path}")

    def save_json(self, data: dict, filename: str) -> None:
        path = os.path.join(self.out_reports, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Report] Lưu JSON: {path}")

    # ------------------------------------------------------------------
    # Bảng tổng kết
    # ------------------------------------------------------------------
    def summarize_classification(self, results: list) -> pd.DataFrame:
        df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall",
                                     "F1", "ROC-AUC", "PR-AUC"]]
        df = df.round(4).sort_values("PR-AUC", ascending=False)
        self.save_table(df, "classification_comparison.csv")
        return df

    def summarize_regression(self, results: list) -> pd.DataFrame:
        df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2"]]
        df = df.round(4).sort_values("MAE")
        self.save_table(df, "regression_comparison.csv")
        return df

    def summarize_clustering(self, results: list) -> pd.DataFrame:
        df = pd.DataFrame(results)
        self.save_table(df, "clustering_comparison.csv")
        return df

    def summarize_semi_supervised(self, df: pd.DataFrame) -> None:
        self.save_table(df, "semi_supervised_learning_curve.csv")

    def summarize_association_rules(self, rules_df: pd.DataFrame) -> None:
        self.save_table(rules_df, "association_rules_top.csv")

    # ------------------------------------------------------------------
    # Insight hành động
    # ------------------------------------------------------------------
    def print_insights(self, profile: pd.DataFrame, rules_df: pd.DataFrame,
                        clf_results: list) -> None:
        print("\n" + "="*60)
        print("INSIGHTS & KHUYẾN NGHỊ HÀNH ĐỘNG")
        print("="*60)
        print("\n1. PHÂN CỤM BỆNH NHÂN:")
        if profile is not None:
            print(profile.to_string())

        print("\n2. TOP LUẬT KẾT HỢP (liên quan bệnh tim):")
        if rules_df is not None and len(rules_df) > 0:
            top5 = rules_df.head(5)
            for _, row in top5.iterrows():
                print(f"   {row['antecedents']} → {row['consequents']} "
                      f"(lift={row['lift']:.2f}, conf={row['confidence']:.2f})")

        print("\n3. PHÂN LỚP – MÔ HÌNH TỐT NHẤT:")
        if clf_results:
            best = max(clf_results, key=lambda x: x.get("PR-AUC", 0))
            print(f"   {best['Model']}: PR-AUC={best.get('PR-AUC',0):.4f}, F1={best.get('F1',0):.4f}")

        print("\n4. KHUYẾN NGHỊ Y TẾ:")
        print("   - Bệnh nhân nam > 50 tuổi, đau ngực dạng 0 (typical angina), exang=1: nguy cơ rất cao")
        print("   - Cholesterol > 240 mg/dl + thalach thấp: cần theo dõi đặc biệt")
        print("   - Cụm nguy cơ cao cần tầm soát định kỳ mỗi 3-6 tháng")
        print("   - Mô hình có thể hỗ trợ sàng lọc sơ bộ, KHÔNG thay thế chẩn đoán bác sĩ")
        print("="*60)

"""
AssociationMiner: Apriori / FP-Growth để tìm tổ hợp triệu chứng bệnh tim.
"""

import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


class AssociationMiner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.assoc_cfg = cfg.get("association", {})
        self.frequent_itemsets_ = None
        self.rules_ = None

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------
    def run_apriori(self, df_onehot: pd.DataFrame) -> pd.DataFrame:
        """Chạy Apriori và trả về frequent itemsets."""
        min_sup = self.assoc_cfg.get("min_support", 0.1)
        self.frequent_itemsets_ = apriori(
            df_onehot,
            min_support=min_sup,
            use_colnames=True,
            max_len=self.assoc_cfg.get("max_len", 4)
        )
        self.frequent_itemsets_["length"] = self.frequent_itemsets_["itemsets"].apply(len)
        print(f"[Apriori] Tìm được {len(self.frequent_itemsets_)} frequent itemsets (min_support={min_sup})")
        return self.frequent_itemsets_

    def run_fpgrowth(self, df_onehot: pd.DataFrame) -> pd.DataFrame:
        """Chạy FP-Growth (nhanh hơn Apriori với dataset lớn)."""
        min_sup = self.assoc_cfg.get("min_support", 0.1)
        self.frequent_itemsets_ = fpgrowth(
            df_onehot,
            min_support=min_sup,
            use_colnames=True,
            max_len=self.assoc_cfg.get("max_len", 4)
        )
        self.frequent_itemsets_["length"] = self.frequent_itemsets_["itemsets"].apply(len)
        print(f"[FP-Growth] Tìm được {len(self.frequent_itemsets_)} frequent itemsets")
        return self.frequent_itemsets_

    def generate_rules(self) -> pd.DataFrame:
        """Tạo association rules từ frequent itemsets."""
        if self.frequent_itemsets_ is None:
            raise RuntimeError("Hãy chạy run_apriori() hoặc run_fpgrowth() trước.")
        min_conf  = self.assoc_cfg.get("min_confidence", 0.6)
        min_lift  = self.assoc_cfg.get("min_lift", 1.2)
        rules = association_rules(
            self.frequent_itemsets_,
            metric="confidence",
            min_threshold=min_conf
        )
        rules = rules[rules["lift"] >= min_lift]
        rules = rules.sort_values("lift", ascending=False)
        self.rules_ = rules
        print(f"[Rules] Số luật (conf>={min_conf}, lift>={min_lift}): {len(rules)}")
        return rules

    # ------------------------------------------------------------------
    # Phân tích luật
    # ------------------------------------------------------------------
    def get_disease_rules(self) -> pd.DataFrame:
        """Lọc các luật có consequent là 'disease'."""
        if self.rules_ is None:
            return pd.DataFrame()
        mask = self.rules_["consequents"].apply(lambda x: "disease" in x)
        return self.rules_[mask].sort_values("lift", ascending=False)

    def get_top_rules(self, n: int = 20, metric: str = "lift") -> pd.DataFrame:
        if self.rules_ is None:
            return pd.DataFrame()
        return self.rules_.nlargest(n, metric)

    def rules_summary(self) -> pd.DataFrame:
        """Tóm tắt thống kê luật."""
        if self.rules_ is None:
            return pd.DataFrame()
        stats = {
            "Tổng số luật": len(self.rules_),
            "Support TB": self.rules_["support"].mean(),
            "Confidence TB": self.rules_["confidence"].mean(),
            "Lift TB": self.rules_["lift"].mean(),
            "Lift max": self.rules_["lift"].max(),
        }
        return pd.DataFrame([stats])

    def format_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        """Định dạng luật để hiển thị đẹp."""
        display = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        display["antecedents"] = display["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        display["consequents"] = display["consequents"].apply(lambda x: ", ".join(sorted(x)))
        display["support"]     = display["support"].round(4)
        display["confidence"]  = display["confidence"].round(4)
        display["lift"]        = display["lift"].round(4)
        return display

    def save_rules(self, path: str) -> None:
        if self.rules_ is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.format_rules(self.rules_).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[Rules] Đã lưu {len(self.rules_)} luật tại {path}")

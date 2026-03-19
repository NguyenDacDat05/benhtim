"""
SemiSupervisedTrainer: thực nghiệm học bán giám sát cho Heart Disease.
- Self-Training với ngưỡng tin cậy cao
- Label Spreading / Label Propagation
- So sánh PR-AUC theo % nhãn
- Phân tích rủi ro pseudo-label
"""

import numpy as np
import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading, LabelPropagation
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, classification_report
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


UNLABELED = -1


class SemiSupervisedTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ss_cfg = cfg.get("semi_supervised", {})
        self.seed   = cfg.get("seed", 42)
        self.results_ = []

    # ------------------------------------------------------------------
    # Giả lập thiếu nhãn
    # ------------------------------------------------------------------
    def simulate_partial_labels(self, X: np.ndarray, y: np.ndarray,
                                 label_ratio: float) -> tuple:
        """
        Giữ lại label_ratio% nhãn, phần còn lại → -1 (unlabeled).
        Trả về y_partial (với -1 cho unlabeled).
        """
        rng = np.random.RandomState(self.seed)
        n_labeled = max(2, int(len(y) * label_ratio))
        labeled_idx = rng.choice(len(y), size=n_labeled, replace=False)
        y_partial = np.full_like(y, UNLABELED)
        y_partial[labeled_idx] = y[labeled_idx]
        return y_partial, labeled_idx

    # ------------------------------------------------------------------
    # Self-Training
    # ------------------------------------------------------------------
    def train_self_training(self, X_train, y_partial, X_test, y_test,
                            base_name: str = "SVM") -> dict:
        threshold = self.ss_cfg.get("self_training_threshold", 0.85)
        max_iter  = self.ss_cfg.get("self_training_max_iter", 10)

        if base_name == "SVM":
            base = SVC(kernel="rbf", probability=True, random_state=self.seed,
                       class_weight="balanced")
        elif base_name == "XGBoost":
            base = XGBClassifier(n_estimators=100, random_state=self.seed,
                                 use_label_encoder=False, eval_metric="logloss", verbosity=0)
        elif base_name == "RF":
            base = RandomForestClassifier(n_estimators=100, random_state=self.seed,
                                          class_weight="balanced")
        else:
            base = SVC(kernel="rbf", probability=True, random_state=self.seed,
                       class_weight="balanced")

        clf = SelfTrainingClassifier(
            estimator=base,
            threshold=threshold,
            max_iter=max_iter,
            verbose=False
        )
        clf.fit(X_train, y_partial)
        return self._evaluate(clf, X_test, y_test, f"SelfTraining-{base_name}")

    # ------------------------------------------------------------------
    # Label Spreading
    # ------------------------------------------------------------------
    def train_label_spreading(self, X_train, y_partial, X_test, y_test) -> dict:
        kernel = self.ss_cfg.get("label_spreading_kernel", "rbf")
        alpha  = self.ss_cfg.get("label_spreading_alpha", 0.2)
        clf = LabelSpreading(kernel=kernel, alpha=alpha, max_iter=1000)
        clf.fit(X_train, y_partial)
        return self._evaluate(clf, X_test, y_test, "LabelSpreading")

    # ------------------------------------------------------------------
    # Label Propagation
    # ------------------------------------------------------------------
    def train_label_propagation(self, X_train, y_partial, X_test, y_test) -> dict:
        clf = LabelPropagation(kernel="rbf", max_iter=1000)
        clf.fit(X_train, y_partial)
        return self._evaluate(clf, X_test, y_test, "LabelPropagation")

    # ------------------------------------------------------------------
    # Supervised-only (baseline ít nhãn)
    # ------------------------------------------------------------------
    def train_supervised_only(self, X_train, y_partial, X_test, y_test) -> dict:
        labeled_mask = y_partial != UNLABELED
        X_labeled = X_train[labeled_mask]
        y_labeled  = y_partial[labeled_mask]
        if len(np.unique(y_labeled)) < 2:
            return {"Model": "Supervised-only", "F1": 0, "PR-AUC": 0, "ROC-AUC": 0}
        clf = RandomForestClassifier(n_estimators=100, random_state=self.seed,
                                     class_weight="balanced")
        clf.fit(X_labeled, y_labeled)
        return self._evaluate(clf, X_test, y_test, "Supervised-only")

    # ------------------------------------------------------------------
    # Learning curve theo % nhãn
    # ------------------------------------------------------------------
    def learning_curve_by_label_ratio(self, X_train, y_train,
                                       X_test, y_test) -> pd.DataFrame:
        """
        So sánh Supervised-only vs Self-Training theo % nhãn.
        Trả về DataFrame với cột: ratio, supervised_prauc, semi_prauc.
        """
        ratios = self.ss_cfg.get("label_ratios", [0.05, 0.10, 0.20, 0.30])
        rows = []
        for ratio in ratios:
            y_partial, _ = self.simulate_partial_labels(X_train, y_train, ratio)
            n_labeled = (y_partial != UNLABELED).sum()
            if len(np.unique(y_partial[y_partial != UNLABELED])) < 2:
                continue
            sup_res  = self.train_supervised_only(X_train, y_partial, X_test, y_test)
            semi_res = self.train_self_training(X_train, y_partial, X_test, y_test)
            rows.append({
                "% nhãn": f"{int(ratio*100)}%",
                "n_labeled": n_labeled,
                "Supervised-only PR-AUC": sup_res.get("PR-AUC", 0),
                "SelfTraining PR-AUC":    semi_res.get("PR-AUC", 0),
                "Supervised-only F1":     sup_res.get("F1", 0),
                "SelfTraining F1":        semi_res.get("F1", 0),
            })
            print(f"  {int(ratio*100)}% nhãn: Sup PR-AUC={sup_res.get('PR-AUC',0):.4f}, "
                  f"Semi PR-AUC={semi_res.get('PR-AUC',0):.4f}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Phân tích rủi ro pseudo-label
    # ------------------------------------------------------------------
    def analyze_pseudo_label_risk(self, X_train, y_partial,
                                   true_labels: np.ndarray) -> dict:
        """
        Phân tích pseudo-label sai: tỷ lệ gán nhãn sai theo các nhóm.
        """
        threshold = self.ss_cfg.get("self_training_threshold", 0.85)
        base = SVC(kernel="rbf", probability=True, random_state=self.seed,
                   class_weight="balanced")
        clf = SelfTrainingClassifier(estimator=base, threshold=threshold,
                                      max_iter=10)
        clf.fit(X_train, y_partial)
        pseudo_labels   = clf.transduction_
        unlabeled_mask  = y_partial == UNLABELED
        pseudo_unlabeled = pseudo_labels[unlabeled_mask]
        true_unlabeled   = true_labels[unlabeled_mask]
        n_wrong = (pseudo_unlabeled != true_unlabeled).sum()
        n_total = unlabeled_mask.sum()
        risk = {
            "n_unlabeled": int(n_total),
            "n_pseudo_labeled": int((pseudo_unlabeled != UNLABELED).sum()),
            "n_wrong_pseudo": int(n_wrong),
            "error_rate": float(n_wrong / max(n_total, 1)),
        }
        print(f"[PseudoLabel Risk] {n_wrong}/{n_total} sai = {risk['error_rate']:.2%}")
        return risk

    # ------------------------------------------------------------------
    # Helper đánh giá
    # ------------------------------------------------------------------
    def _evaluate(self, clf, X_test: np.ndarray, y_test: np.ndarray,
                  name: str) -> dict:
        y_pred = clf.predict(X_test)
        y_prob = None
        n_classes = len(np.unique(y_test))
        avg = 'macro' if n_classes > 2 else 'binary'

        if hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(X_test)
                if n_classes == 2:
                    y_prob = y_prob[:, 1]
            except Exception:
                pass

        result = {
            "Model": name,
            "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None and n_classes > 2 else (roc_auc_score(y_test, y_prob) if y_prob is not None else None),
            "PR-AUC": average_precision_score(y_test, y_prob, average=avg) if y_prob is not None else None,
        }
        self.results_.append(result)
        print(f"  [{name}] F1={result['F1']:.4f}, "
              f"PR-AUC={result.get('PR-AUC', 'N/A')}")
        return result

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results_)

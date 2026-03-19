"""
SupervisedTrainer: huấn luyện mô hình phân lớp và hồi quy cho Heart Disease.
Hỗ trợ: LogisticRegression, SVM, RandomForest, XGBoost
Imbalance: class_weight, SMOTE
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class SupervisedTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.clf_cfg  = cfg.get("classification", {})
        self.reg_cfg  = cfg.get("regression", {})
        self.seed     = cfg.get("seed", 42)
        self.results_ = {}
        self.models_  = {}

    # ------------------------------------------------------------------
    # Phân lớp (Classification)
    # ------------------------------------------------------------------
    def build_classifiers(self) -> dict:
        """Tạo dict các mô hình phân lớp cần thử."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.seed, class_weight="balanced"
            ),
            "SVM": SVC(
                kernel="rbf", probability=True, random_state=self.seed, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=self.seed, class_weight="balanced"
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=self.seed, verbosity=0,
                scale_pos_weight=1
            ),
        }

    def train_with_cv(self, X: np.ndarray, y: np.ndarray,
                      use_smote: bool = True) -> pd.DataFrame:
        """Train tất cả mô hình với cross-validation, trả về bảng so sánh."""
        classifiers = self.build_classifiers()
        cv = StratifiedKFold(n_splits=self.clf_cfg.get("cv_folds", 5),
                             shuffle=True, random_state=self.seed)
        rows = []
        for name, clf in classifiers.items():
            start = time.time()
            if use_smote:
                pipeline = ImbPipeline([
                    ("smote", SMOTE(random_state=self.seed)),
                    ("clf", clf)
                ])
            else:
                pipeline = Pipeline([("clf", clf)])

            # Check if multiclass - use macro average for scoring
            n_classes = len(np.unique(y))
            scoring = {
                "f1": "f1_macro" if n_classes > 2 else "f1",
                "roc_auc": "roc_auc_ovr_weighted" if n_classes > 2 else "roc_auc",
                "average_precision": "average_precision"
            }

            scores = cross_validate(
                pipeline, X, y, cv=cv,
                scoring=scoring,
                n_jobs=-1
            )
            elapsed = time.time() - start
            row = {
                "Mô hình": name,
                "F1 (CV)": scores["test_f1"].mean(),
                "F1 std": scores["test_f1"].std(),
                "ROC-AUC (CV)": scores["test_roc_auc"].mean(),
                "PR-AUC (CV)": scores["test_average_precision"].mean(),
                "Thời gian (s)": round(elapsed, 2),
            }
            rows.append(row)
            print(f"  {name}: F1={row['F1 (CV)']:.4f}, ROC-AUC={row['ROC-AUC (CV)']:.4f}, "
                  f"PR-AUC={row['PR-AUC (CV)']:.4f} ({elapsed:.1f}s)")
        return pd.DataFrame(rows).sort_values("PR-AUC (CV)", ascending=False)

    def train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          model_name: str = "XGBoost",
                          use_smote: bool = True) -> object:
        """Train mô hình tốt nhất trên toàn bộ tập train."""
        classifiers = self.build_classifiers()
        clf = classifiers[model_name]
        if use_smote:
            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=self.seed)),
                ("clf", clf)
            ])
        else:
            pipeline = Pipeline([("clf", clf)])
        pipeline.fit(X_train, y_train)
        self.models_[model_name] = pipeline
        print(f"[Train] {model_name} đã được train xong.")
        return pipeline

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                 model_name: str = "Model") -> dict:
        """Đánh giá mô hình trên tập test."""
        y_pred = model.predict(X_test)

        # Check if multiclass - use macro average for F1
        n_classes = len(np.unique(y_test))
        avg = 'macro' if n_classes > 2 else 'binary'

        # Get probabilities - use full matrix for multiclass
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if n_classes > 2:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                pr_auc = average_precision_score(y_test, y_prob, average=avg)
            else:
                y_prob = y_prob[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob) if n_classes == 2 else None
                pr_auc = average_precision_score(y_test, y_prob) if n_classes == 2 else None
        else:
            y_prob = None
            roc_auc = None
            pr_auc = None

        result = {
            "Model": model_name,
            "F1": f1_score(y_test, y_pred, average=avg),
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
        self.results_[model_name] = result
        print(f"\n[Eval] {model_name}")
        print(f"  F1={result['F1']:.4f}, ROC-AUC={result['ROC-AUC']:.4f}, PR-AUC={result['PR-AUC']:.4f}")
        print(classification_report(y_test, y_pred))
        return result

    def tune_best_model(self, X: np.ndarray, y: np.ndarray,
                        model_name: str = "XGBoost") -> object:
        """GridSearchCV để tối ưu hyperparameter mô hình tốt nhất."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        if model_name == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                  random_state=self.seed, verbosity=0)
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            }
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=self.seed, class_weight="balanced")
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            }
        else:
            raise ValueError(f"Chưa hỗ trợ tune: {model_name}")

        gs = GridSearchCV(model, param_grid, cv=cv, scoring="average_precision",
                         n_jobs=-1, verbose=0)
        gs.fit(X, y)
        print(f"[Tune] Best params: {gs.best_params_}")
        print(f"[Tune] Best PR-AUC: {gs.best_score_:.4f}")
        return gs.best_estimator_

    # ------------------------------------------------------------------
    # Hồi quy (Regression) – dự đoán trestbps/chol
    # ------------------------------------------------------------------
    def build_regressors(self) -> dict:
        return {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "XGBRegressor": XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=self.seed, verbosity=0
            ),
        }

    def train_regression(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        regressors = self.build_regressors()
        rows = []
        for name, reg in regressors.items():
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rows.append({"Mô hình": name, "MAE": mae, "RMSE": rmse})
            print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            self.models_[name] = reg
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Lưu / tải mô hình
    # ------------------------------------------------------------------
    def save_model(self, model, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"[Save] Mô hình lưu tại {path}")

    def load_model(self, path: str):
        return joblib.load(path)

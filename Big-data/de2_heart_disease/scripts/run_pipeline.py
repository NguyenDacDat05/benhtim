"""
run_pipeline.py: Chay toan bo pipeline Heart Disease Prediction (khong can notebook).
Luong: Load -> Clean -> Features -> Mining -> Modeling -> Evaluation
"""

import sys
import os
import warnings
import argparse
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader, load_config
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationMiner
from src.mining.clustering import ClusteringMiner
from src.models.supervised import SupervisedTrainer
from src.models.semi_supervised import SemiSupervisedTrainer
from src.evaluation.metrics import Metrics
from src.evaluation.report import Reporter
from src.visualization.plots import Plotter


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run(config_path: str = "configs/params.yaml",
        skip_semi: bool = False,
        skip_regression: bool = False) -> None:

    t_start = time.time()
    cfg = load_config(config_path)
    seed = cfg["seed"]
    reporter = Reporter(cfg)
    plotter  = Plotter(cfg)

    # ------------------------------------------------------------------
    # STEP 1: Load & Clean
    # ------------------------------------------------------------------
    print_section("STEP 1: LOAD & PREPROCESS")
    loader  = DataLoader(config_path)
    df_raw  = loader.load_raw()
    print(f"Raw data: {df_raw.shape}")

    cleaner = DataCleaner(cfg)
    df_proc = cleaner.fit_transform(df_raw)
    cleaner.print_before_after_stats(df_raw, df_proc)
    cleaner.save_processed(df_proc, cfg["paths"]["processed_data"])

    # ------------------------------------------------------------------
    # STEP 2: Feature Engineering
    # ------------------------------------------------------------------
    print_section("STEP 2: FEATURE ENGINEERING")
    builder = FeatureBuilder(cfg)
    X, y = builder.get_X_y(df_proc)
    X_arr, y_arr = X.values, y.values
    feature_names = list(X.columns)
    print(f"Features: {feature_names}")
    print(f"X: {X_arr.shape}, y distribution: {dict(zip(*np.unique(y_arr, return_counts=True)))}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=cfg["test_size"], stratify=y_arr, random_state=seed
    )

    # ------------------------------------------------------------------
    # STEP 3: Association Rules
    # ------------------------------------------------------------------
    print_section("STEP 3: ASSOCIATION RULES (APRIORI)")
    miner = AssociationMiner(cfg)
    df_onehot = builder.build_onehot_transactions(df_raw)
    freq_items = miner.run_apriori(df_onehot)
    rules = miner.generate_rules()
    disease_rules = miner.get_disease_rules()
    print(f"\nTop 5 luat lien quan benh tim:")
    if len(disease_rules) > 0:
        print(miner.format_rules(disease_rules.head(5)).to_string())
    reporter.summarize_association_rules(miner.format_rules(disease_rules.head(20)))
    miner.save_rules("outputs/tables/all_association_rules.csv")
    plotter.plot_rules_scatter(rules)
    if len(disease_rules) > 0:
        plotter.plot_top_rules_lift(miner.format_rules(disease_rules.head(15)))

    # ------------------------------------------------------------------
    # STEP 4: Clustering
    # ------------------------------------------------------------------
    print_section("STEP 4: CLUSTERING")
    clusterer = ClusteringMiner(cfg)

    print("\n[4a] Chon K toi uu:")
    k_results = clusterer.find_optimal_k(X_arr)
    plotter.plot_elbow(k_results)

    k_best = cfg["clustering"]["kmeans_best_k"]
    print(f"\n[4b] KMeans K={k_best}:")
    labels_km = clusterer.fit_kmeans(X_arr, k_best)

    print("\n[4c] HAC:")
    labels_hac = clusterer.fit_hac(X_arr, k_best)

    print("\n[4d] DBSCAN:")
    labels_db = clusterer.fit_dbscan(X_arr)

    clust_results = [
        Metrics.clustering_summary(X_arr, labels_km,  "KMeans"),
        Metrics.clustering_summary(X_arr, labels_hac, "HAC"),
        Metrics.clustering_summary(X_arr, labels_db,  "DBSCAN"),
    ]
    clust_df = reporter.summarize_clustering(clust_results)
    print("\nBang so sanh phan cum:")
    print(clust_df.to_string())

    X_2d = clusterer.fit_pca(X_arr)
    plotter.plot_clusters_2d(X_2d, labels_km, f"KMeans K={k_best}")
    profile = clusterer.profile_clusters(df_proc, labels_km)
    print("\nHo so cum:")
    print(profile.to_string())
    reporter.save_table(profile, "cluster_profiles.csv", index=True)
    plotter.plot_cluster_profiles(profile)

    # ------------------------------------------------------------------
    # STEP 5: Classification
    # ------------------------------------------------------------------
    print_section("STEP 5: CLASSIFICATION")
    trainer = SupervisedTrainer(cfg)

    print("\n[5a] Cross-validation (5-fold):")
    cv_results = trainer.train_with_cv(X_train, y_train, use_smote=True)
    reporter.save_table(cv_results, "cv_results.csv")

    print("\n[5b] Train & evaluate tren test set:")
    from sklearn.metrics import roc_curve, precision_recall_curve
    all_clf_results = []
    models_roc, models_pr = [], []

    for model_name in ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]:
        model = trainer.train_final_model(X_train, y_train, model_name=model_name, use_smote=True)
        ev = trainer.evaluate(model, X_test, y_test, model_name=model_name)
        y_pred = model.predict(X_test)
        # Get full probability matrix for multiclass
        y_prob = model.predict_proba(X_test)
        summary = Metrics.classification_summary(y_test, y_pred, y_prob, model_name)
        all_clf_results.append(summary)

        # ROC and PR curves only work for binary
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            prec, rec, _ = precision_recall_curve(y_test, y_prob[:, 1])
            models_roc.append({"name": model_name, "fpr": fpr, "tpr": tpr, "auc": summary["ROC-AUC"]})
            models_pr.append({"name": model_name, "precision": prec, "recall": rec, "auc": summary["PR-AUC"]})

    clf_df = reporter.summarize_classification(all_clf_results)
    print("\nBang ket qua phan lop:")
    print(clf_df.to_string())
    plotter.plot_model_comparison(clf_df)
    plotter.plot_roc_curves(models_roc)
    plotter.plot_pr_curves(models_pr)

    # Best model: confusion matrix + feature importance
    best_model = trainer.models_.get("XGBoost")
    y_pred_best = best_model.predict(X_test)
    plotter.plot_confusion_matrix(y_test, y_pred_best, "XGBoost")
    trainer.save_model(best_model, "outputs/models/xgboost_best.pkl")

    # ------------------------------------------------------------------
    # STEP 6: Regression
    # ------------------------------------------------------------------
    if not skip_regression:
        print_section("STEP 6: REGRESSION (du bao trestbps)")
        reg_cols = [c for c in feature_names if c != "trestbps"]
        X_reg = df_proc[reg_cols].values
        y_reg = df_proc["trestbps"].values
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=seed
        )
        reg_df = trainer.train_regression(Xr_tr, yr_tr, Xr_te, yr_te)
        reporter.summarize_regression([
            Metrics.regression_summary(yr_te, trainer.models_[n].predict(Xr_te), n)
            for n in ["Linear Regression", "Ridge", "XGBRegressor"]
            if n in trainer.models_
        ])
        print(reg_df.to_string())

    # ------------------------------------------------------------------
    # STEP 7: Semi-supervised
    # ------------------------------------------------------------------
    if not skip_semi:
        print_section("STEP 7: SEMI-SUPERVISED (ban giam sat)")
        ss_trainer = SemiSupervisedTrainer(cfg)
        curve_df = ss_trainer.learning_curve_by_label_ratio(X_train, y_train, X_test, y_test)
        reporter.summarize_semi_supervised(curve_df)
        plotter.plot_learning_curve_semi(curve_df)
        print("\nLearning curve:")
        print(curve_df.to_string())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print_section(f"PIPELINE HOAN TAT ({elapsed:.1f}s)")
    reporter.print_insights(profile, miner.format_rules(disease_rules.head(20)) if len(disease_rules) > 0 else None, all_clf_results)
    print("\nOutputs:")
    for root, dirs, files in os.walk("outputs"):
        for f in files:
            print(f"  {os.path.join(root, f)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Data Mining Pipeline")
    parser.add_argument("--config", default="configs/params.yaml", help="Path to params.yaml")
    parser.add_argument("--skip-semi", action="store_true", help="Bo qua semi-supervised")
    parser.add_argument("--skip-regression", action="store_true", help="Bo qua regression")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run(config_path=args.config, skip_semi=args.skip_semi, skip_regression=args.skip_regression)

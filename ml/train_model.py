# ==============================================================================
# ML PIPELINE SCRIPT: ml/train_model.py
# Applied Machine Learning Pipeline for Student Dropout Early Warning System
# Performs data cleaning, EDA, model training, evaluation, CV, experiments,
# risk band analysis, feature importance, and saves model & metric artifacts.
# ==============================================================================

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Import settings from parent directory config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    DATA_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TARGET_COL,
    APP_FEATURES,
    EARLY_FEATURES,
    ENROLMENT_FEATURES,
    RISK_LOW_THRESHOLD,
    RISK_HIGH_THRESHOLD
)

# Set matplotlib style for clear, readable figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def run_pipeline():
    # Ensure output directories exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    print("==================================================================")
    print("      STUDENT DROPOUT EARLY WARNING SYSTEM - ML PIPELINE          ")
    print("==================================================================\n")

    # ==========================================================================
    # SECTION 1: LOAD DATA
    # ==========================================================================
    # Load dataset from data/data.csv using semicolon separator
    print("[SECTION 1] Loading Dataset...")
    df = pd.read_csv(DATA_PATH, sep=";")
    print(f"Dataset successfully loaded from '{DATA_PATH}'.")
    print(f"Initial Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # ==========================================================================
    # SECTION 2: CLEAN DATA
    # ==========================================================================
    # Strip leading/trailing whitespace from column names
    print("[SECTION 2] Cleaning & Preprocessing Data...")
    df.columns = df.columns.str.strip()
    
    # Verify shape, missing values, duplicates
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    print(f"Total Missing Values: {missing_count}")
    print(f"Total Duplicate Rows: {duplicate_count}")
    
    # Display raw target counts
    target_counts = df[TARGET_COL].value_counts()
    print("Class Distribution in raw data:")
    for label, count in target_counts.items():
        print(f"  - {label}: {count}")

    # Binary Target Creation: Dropout = 1, (Graduate + Enrolled) = 0
    df["dropout_binary"] = (df[TARGET_COL] == "Dropout").astype(int)
    
    # Define ALL_FEATURES (all columns except target columns)
    ALL_FEATURES = [c for c in df.columns if c not in [TARGET_COL, "dropout_binary"]]
    print(f"Total feature columns available: {len(ALL_FEATURES)}")

    # Plot 1: Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#dc3545", "#28a745", "#ffc107"]  # Red for Dropout, Green for Graduate, Amber for Enrolled
    counts = df[TARGET_COL].value_counts()
    ax.bar(counts.index, counts.values, color=colors, edgecolor="black", width=0.5)
    ax.set_title("Student Academic Outcome Class Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Academic Outcome (Target)", fontsize=10)
    ax.set_ylabel("Number of Students", fontsize=10)
    for i, count in enumerate(counts.values):
        ax.text(i, count + 30, str(count), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    class_dist_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    fig.savefig(class_dist_path, dpi=300)
    plt.close()
    print(f"Saved figure: {class_dist_path}\n")

    # ==========================================================================
    # SECTION 3: EDA PLOTS
    # ==========================================================================
    print("[SECTION 3] Generating Exploratory Data Analysis (EDA) Plots...")

    # Plot 2: Correlation Heatmap (Top 10 features vs dropout binary)
    numeric_df = df[ALL_FEATURES].select_dtypes(include=[np.number])
    correlations = numeric_df.apply(lambda col: col.corr(df["dropout_binary"])).abs()
    top10_corr_features = correlations.sort_values(ascending=False).head(10).index.tolist()
    
    heatmap_data = df[top10_corr_features + ["dropout_binary"]].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        cbar_kws={"label": "Correlation Coefficient"}
    )
    ax.set_title("Top 10 Features Correlated with Student Dropout", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    corr_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    fig.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved figure: {corr_path}")

    # Plot 3: Dropout Rate by Tuition Fees Status
    fig, ax = plt.subplots(figsize=(7, 5))
    tuition_df = df.groupby("Tuition fees up to date")["dropout_binary"].mean().reset_index()
    tuition_df["Tuition fees up to date"] = tuition_df["Tuition fees up to date"].map({0: "No (Unpaid)", 1: "Yes (Up to Date)"})
    bars = ax.bar(tuition_df["Tuition fees up to date"], tuition_df["dropout_binary"] * 100, color=["#dc3545", "#28a745"], edgecolor="black", width=0.4)
    ax.set_title("Dropout Rate by Tuition Fees Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tuition Fees Up To Date", fontsize=10)
    ax.set_ylabel("Dropout Rate (%)", fontsize=10)
    ax.set_ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    tuition_path = os.path.join(FIGURES_DIR, "dropout_by_tuition_status.png")
    fig.savefig(tuition_path, dpi=300)
    plt.close()
    print(f"Saved figure: {tuition_path}")

    # Plot 4: Dropout Rate by Scholarship Status
    fig, ax = plt.subplots(figsize=(7, 5))
    scholar_df = df.groupby("Scholarship holder")["dropout_binary"].mean().reset_index()
    scholar_df["Scholarship holder"] = scholar_df["Scholarship holder"].map({0: "No Scholarship", 1: "Scholarship Holder"})
    bars = ax.bar(scholar_df["Scholarship holder"], scholar_df["dropout_binary"] * 100, color=["#e74c3c", "#3498db"], edgecolor="black", width=0.4)
    ax.set_title("Dropout Rate by Scholarship Status", fontsize=12, fontweight="bold")
    ax.set_xlabel("Scholarship Holder Status", fontsize=10)
    ax.set_ylabel("Dropout Rate (%)", fontsize=10)
    ax.set_ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    scholar_path = os.path.join(FIGURES_DIR, "dropout_by_scholarship.png")
    fig.savefig(scholar_path, dpi=300)
    plt.close()
    print(f"Saved figure: {scholar_path}\n")

    # ==========================================================================
    # SECTION 4: TRAIN / TEST SPLIT
    # ==========================================================================
    # Prepare 80/20 Stratified Split using the 16 APP_FEATURES
    print("[SECTION 4] Splitting Dataset (80/20 Stratified Split)...")
    X = df[APP_FEATURES]
    y = df["dropout_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train Set Shape: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test Set Shape : {X_test.shape[0]} rows, {X_test.shape[1]} features\n")

    # ==========================================================================
    # SECTION 5: TRAIN & COMPARE MODELS
    # ==========================================================================
    # Models to compare on APP_FEATURES:
    # 1. Decision Tree (max_depth=6)
    # 2. Random Forest (n_estimators=200, class_weight="balanced") -> MAIN MODEL
    # 3. Gradient Boosting (default settings)
    # 4. Logistic Regression with StandardScaler
    # Note: Logistic Regression was avoided in the original paper due to multicollinearity
    # between curricular unit attributes, but included here as a baseline.
    print("[SECTION 5] Training Classification Models...")
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  - Trained {name} successfully.")
    print()

    # ==========================================================================
    # SECTION 6: EVALUATE MODELS ON TEST SET
    # ==========================================================================
    print("[SECTION 6] Evaluating Models on Test Set...")
    eval_results = []
    roc_curves = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        
        eval_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })
        
        # Save ROC curve data points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)

    model_comp_df = pd.DataFrame(eval_results)
    model_comp_csv = os.path.join(TABLES_DIR, "model_comparison.csv")
    model_comp_df.to_csv(model_comp_csv, index=False)
    print(f"Saved experiment table: {model_comp_csv}")
    print(model_comp_df.to_string(index=False))
    print()

    # Plot 5: Confusion Matrix for Random Forest (MAIN Model)
    rf_model = trained_models["Random Forest"]
    rf_y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, rf_y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-Dropout", "Dropout"],
                yticklabels=["Non-Dropout", "Dropout"])
    ax.set_title("Random Forest Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("Actual Label", fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_rf.png")
    fig.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved figure: {cm_path}")

    # Plot 6: ROC Curves for All Models
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)", linewidth=1.5)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve_all_models.png")
    fig.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved figure: {roc_path}\n")

    # ==========================================================================
    # SECTION 7: CROSS-VALIDATION
    # ==========================================================================
    print("[SECTION 7] Running 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_records = []

    for name, model in models.items():
        # Evaluate F1 score for binary dropout target
        cv_scores = cross_validate(
            model, X, y, cv=skf, scoring=["f1", "accuracy"]
        )
        f1_mean = round(cv_scores["test_f1"].mean(), 4)
        f1_std = round(cv_scores["test_f1"].std(), 4)
        acc_mean = round(cv_scores["test_accuracy"].mean(), 4)
        acc_std = round(cv_scores["test_accuracy"].std(), 4)
        
        cv_records.append({
            "Model": name,
            "CV_F1_Mean": f1_mean,
            "CV_F1_Std": f1_std,
            "CV_Accuracy_Mean": acc_mean,
            "CV_Accuracy_Std": acc_std
        })

    cv_df = pd.DataFrame(cv_records)
    cv_csv = os.path.join(TABLES_DIR, "cv_results.csv")
    cv_df.to_csv(cv_csv, index=False)
    print(f"Saved experiment table: {cv_csv}")
    print(cv_df.to_string(index=False))
    print()

    # ==========================================================================
    # SECTION 8: EXPERIMENTS
    # ==========================================================================
    print("[SECTION 8] Running Feature-Set and 3-Class Experiments...")
    
    # 8b. Feature-set comparison using Random Forest
    feature_sets = {
        "ALL_FEATURES": ALL_FEATURES,
        "APP_FEATURES": APP_FEATURES,
        "EARLY_FEATURES": EARLY_FEATURES,
        "ENROLMENT_FEATURES": ENROLMENT_FEATURES
    }
    
    feat_set_records = []
    for f_name, f_list in feature_sets.items():
        X_sub = df[f_list]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )
        rf_exp = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE)
        rf_exp.fit(X_tr, y_tr)
        
        y_p = rf_exp.predict(X_te)
        y_pb = rf_exp.predict_proba(X_te)[:, 1]
        
        feat_set_records.append({
            "Feature_Set": f_name,
            "Num_Features": len(f_list),
            "Accuracy": round(accuracy_score(y_te, y_p), 4),
            "Precision": round(precision_score(y_te, y_p, pos_label=1), 4),
            "Recall": round(recall_score(y_te, y_p, pos_label=1), 4),
            "F1_Score": round(f1_score(y_te, y_p, pos_label=1), 4),
            "ROC_AUC": round(roc_auc_score(y_te, y_pb), 4)
        })
        
    feat_set_df = pd.DataFrame(feat_set_records)
    feat_set_csv = os.path.join(TABLES_DIR, "feature_set_comparison.csv")
    feat_set_df.to_csv(feat_set_csv, index=False)
    print(f"Saved experiment table: {feat_set_csv}")
    print(feat_set_df.to_string(index=False))
    print()

    # 8c. 3-Class Target Comparison (Dropout / Enrolled / Graduate)
    print("Running 3-Class Target Comparison (matching original UCI benchmark paper)...")
    y_3class = df[TARGET_COL]
    X_tr_3c, X_te_3c, y_tr_3c, y_te_3c = train_test_split(
        X, y_3class, test_size=0.20, stratify=y_3class, random_state=RANDOM_STATE
    )
    
    three_class_records = []
    models_3c = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    }
    
    for name, model_3c in models_3c.items():
        model_3c.fit(X_tr_3c, y_tr_3c)
        y_p_3c = model_3c.predict(X_te_3c)
        
        acc_3c = accuracy_score(y_te_3c, y_p_3c)
        f1_weighted = f1_score(y_te_3c, y_p_3c, average="weighted")
        
        three_class_records.append({
            "Model": name,
            "Accuracy": round(acc_3c, 4),
            "Weighted_F1": round(f1_weighted, 4)
        })
        
    three_class_df = pd.DataFrame(three_class_records)
    three_class_csv = os.path.join(TABLES_DIR, "three_class_comparison.csv")
    three_class_df.to_csv(three_class_csv, index=False)
    print(f"Saved experiment table: {three_class_csv}")
    print(three_class_df.to_string(index=False))
    print()

    # ==========================================================================
    # SECTION 9: RISK BANDS ANALYSIS
    # ==========================================================================
    # Evaluate risk bands on test set using Random Forest model
    print("[SECTION 9] Analyzing Risk Bands on Test Set...")
    rf_test_probs = rf_model.predict_proba(X_test)[:, 1]
    
    test_analysis_df = pd.DataFrame({
        "Actual_Dropout": y_test.values,
        "Dropout_Prob": rf_test_probs
    })

    def assign_risk_band(prob):
        if prob < RISK_LOW_THRESHOLD:
            return "Low Risk"
        elif prob < RISK_HIGH_THRESHOLD:
            return "Medium Risk"
        else:
            return "High Risk"

    test_analysis_df["Risk_Band"] = test_analysis_df["Dropout_Prob"].apply(assign_risk_band)

    band_order = ["Low Risk", "Medium Risk", "High Risk"]
    band_ranges = {
        "Low Risk": f"< {RISK_LOW_THRESHOLD}",
        "Medium Risk": f"{RISK_LOW_THRESHOLD} to {RISK_HIGH_THRESHOLD}",
        "High Risk": f">= {RISK_HIGH_THRESHOLD}"
    }

    risk_band_summary = []
    for band in band_order:
        band_subset = test_analysis_df[test_analysis_df["Risk_Band"] == band]
        total_st = len(band_subset)
        actual_dropouts = int(band_subset["Actual_Dropout"].sum())
        dropout_pct = round((actual_dropouts / total_st * 100), 2) if total_st > 0 else 0.0
        
        risk_band_summary.append({
            "Risk_Band": band,
            "Prob_Range": band_ranges[band],
            "Total_Students": total_st,
            "Actual_Dropouts": actual_dropouts,
            "Dropout_Percentage": dropout_pct
        })

    risk_band_df = pd.DataFrame(risk_band_summary)
    risk_band_csv = os.path.join(TABLES_DIR, "risk_band_table.csv")
    risk_band_df.to_csv(risk_band_csv, index=False)
    print(f"Saved experiment table: {risk_band_csv}")
    print(risk_band_df.to_string(index=False))
    print()

    # ==========================================================================
    # SECTION 10: FEATURE IMPORTANCE
    # ==========================================================================
    print("[SECTION 10] Plotting Top 10 Feature Importances (Random Forest)...")
    importances = rf_model.feature_importances_
    feat_imp_series = pd.Series(importances, index=APP_FEATURES).sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_imp_series.index, feat_imp_series.values, color="#2c3e50", edgecolor="black")
    ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    for i, val in enumerate(feat_imp_series.values):
        ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontweight="bold")
    plt.tight_layout()
    feat_imp_path = os.path.join(FIGURES_DIR, "feature_importance_rf.png")
    fig.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"Saved figure: {feat_imp_path}\n")

    # ==========================================================================
    # SECTION 11: SAVE MODEL & METRICS + PRINT SUMMARY
    # ==========================================================================
    print("[SECTION 11] Saving Model and Summary Metrics...")
    
    # Save Random Forest model
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Saved trained Random Forest model to '{MODEL_PATH}'.")

    # Save metrics JSON
    rf_eval = [r for r in eval_results if r["Model"] == "Random Forest"][0]
    rf_cv = [c for c in cv_records if c["Model"] == "Random Forest"][0]
    
    metrics_data = {
        "final_model": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "dataset_shape": list(df.shape),
        "app_features": APP_FEATURES,
        "rf_test_metrics": rf_eval,
        "rf_cv_metrics": rf_cv,
        "risk_band_validation": risk_band_summary
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved summary metrics to '{METRICS_PATH}'.")

    # Final Clean Console Summary
    print("\n" + "=" * 66)
    print("                PHASE 1 PIPELINE EXECUTION SUMMARY               ")
    print("=" * 66)
    print(f"1. Dataset Loaded       : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"2. Dropout Target       : {df['dropout_binary'].sum()} dropouts ({(df['dropout_binary'].mean()*100):.2f}%)")
    print(f"3. Features Selected    : {len(APP_FEATURES)} features (APP_FEATURES)")
    print(f"4. Main Model           : Random Forest Classifier (200 trees, balanced)")
    print("5. Test Set Performance :")
    print(f"   - Accuracy  : {rf_eval['Accuracy'] * 100:.2f}%")
    print(f"   - Precision : {rf_eval['Precision'] * 100:.2f}%")
    print(f"   - Recall    : {rf_eval['Recall'] * 100:.2f}%")
    print(f"   - F1-Score  : {rf_eval['F1_Score']:.4f}")
    print(f"   - ROC-AUC   : {rf_eval['ROC_AUC']:.4f}")
    print("6. 5-Fold Cross-Val     :")
    print(f"   - CV F1     : {rf_cv['CV_F1_Mean']:.4f} +/- {rf_cv['CV_F1_Std']:.4f}")
    print(f"   - CV Acc    : {rf_cv['CV_Accuracy_Mean']*100:.2f}% +/- {rf_cv['CV_Accuracy_Std']*100:.2f}%")
    print("7. Validation across Risk Bands on Test Set:")
    for b in risk_band_summary:
        print(f"   - {b['Risk_Band']:<11} ({b['Prob_Range']:<11}): {b['Total_Students']:>3} students | Actual Dropouts: {b['Actual_Dropouts']:>3} ({b['Dropout_Percentage']:.1f}%)")
    print("==================================================================\n")
    print("Phase 1 ML Pipeline executed successfully! Model and outputs are saved.")


if __name__ == "__main__":
    run_pipeline()

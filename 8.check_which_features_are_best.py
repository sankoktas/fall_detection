import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def main():
    # 1) Load your features CSV
    input_csv = "data/mar_11_features_3class.csv"
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # 2) Separate features X and label y
    #    (exclude start_time, end_time, etc. from X)
    feature_cols = [
        'acc_mean','acc_var','acc_kurtosis','acc_skewness','acc_median','acc_p10','acc_p25','acc_p50','acc_p75',
        'gyro_mean','gyro_var','gyro_kurtosis','gyro_skewness','gyro_median','gyro_p10','gyro_p25','gyro_p50','gyro_p75'
    ]
    X = df[feature_cols].copy()
    y = df["label"].copy()  # 'NO_MOTION' / 'MOTION' / 'FALL'

    # 2a) Encode the label into numeric if needed
    #     e.g., {NO_MOTION:0, MOTION:1, FALL:2}
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Option A: Feature Importance via RandomForest
    # ------------------------------------------------
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X, y_encoded)
    
    importances = forest.feature_importances_
    # Sort features by importance (descending)
    indices = np.argsort(importances)[::-1]
    
    print("\n=== RandomForest Feature Importances ===")
    for rank, idx in enumerate(indices):
        print(f"{rank+1:2d}) {feature_cols[idx]:>15s}: {importances[idx]:.4f}")

    # Also do a quick cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_score = cross_val_score(forest, X, y_encoded, cv=cv, scoring="accuracy")
    print(f"\nRandomForest CV Accuracy: {rf_cv_score.mean():.3f} (+/- {rf_cv_score.std():.3f})")

    # Option B: Univariate feature selection
    # ------------------------------------------------
    # For example, using ANOVA F-test:
    selector_f = SelectKBest(score_func=f_classif, k="all")  # 'all' => get scores for all features
    selector_f.fit(X, y_encoded)
    f_scores = selector_f.scores_
    
    print("\n=== ANOVA F-test Scores (SelectKBest) ===")
    scores_f_sorted = sorted(zip(feature_cols, f_scores), key=lambda x: x[1], reverse=True)
    for name, score in scores_f_sorted:
        print(f"{name:15s}: {score:.4f}")
    
    # Or using Mutual Information:
    selector_mi = SelectKBest(score_func=mutual_info_classif, k="all")
    selector_mi.fit(X, y_encoded)
    mi_scores = selector_mi.scores_
    
    print("\n=== Mutual Information Scores (SelectKBest) ===")
    scores_mi_sorted = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)
    for name, score in scores_mi_sorted:
        print(f"{name:15s}: {score:.4f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import pandas as pd
import numpy as np

# For train/test split and cross validation
from sklearn.model_selection import train_test_split, GridSearchCV
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
# SMOTE for oversampling the minority class
from imblearn.over_sampling import SMOTE

def main():
    #----------------------------------------------------------------------
    # 1) LOAD DATA
    #----------------------------------------------------------------------
    input_csv = "data/timeseries_data.csv"
    df = pd.read_csv(input_csv)

    # Convert label "FALL"/"NO_FALL" -> numeric
    df["label_num"] = df["label"].apply(lambda x: 1 if x.upper() == "FALL" else 0)
    
    # Remove rows with NaNs or infinite
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    #----------------------------------------------------------------------
    # 2) PREPARE FEATURES & LABEL
    #----------------------------------------------------------------------
    feature_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    X = df[feature_cols].to_numpy()
    y = df["label_num"].to_numpy()

    #----------------------------------------------------------------------
    # 3) TRAIN/TEST SPLIT
    #----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #----------------------------------------------------------------------
    # 4) DEAL WITH IMBALANCE: SMOTE (Oversampling FALL)
    #----------------------------------------------------------------------
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Before SMOTE: y_train counts = {np.bincount(y_train)}")
    print(f"After  SMOTE: y_train_res counts = {np.bincount(y_train_res)}")

    #----------------------------------------------------------------------
    # 5) HYPERPARAMETER TUNING WITH GRIDSEARCHCV
    #----------------------------------------------------------------------
    # We do a small grid here as an example. Adjust as desired.
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        # "class_weight": ["balanced", {0:1, 1:4}]  # Another approach if not using SMOTE
    }

    # We'll optimize for F1 since we care about balancing precision/recall
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",    # Could also choose "recall" if you want maximum sensitivity
        cv=3,            # 3-fold cross-validation
        n_jobs=-1,       # Use all CPU cores if desired
        verbose=1
    )

    grid_search.fit(X_train_res, y_train_res)

    print("\nBest parameters from GridSearch:")
    print(grid_search.best_params_)
    best_rf = grid_search.best_estimator_

    #----------------------------------------------------------------------
    # 6) EVALUATE ON TEST SET (RAW PREDICTION, THRESHOLD=0.5)
    #----------------------------------------------------------------------
    # RandomForestClassifier by default does .predict() with threshold=0.5
    y_pred_test = best_rf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)

    print(f"\nInitial Test Accuracy (threshold=0.5): {acc_test:.3f}")

    #----------------------------------------------------------------------
    # 7) THRESHOLD TUNING FOR BEST F1
    #----------------------------------------------------------------------
    # We can get prediction probabilities: best_rf.predict_proba(X_test)[:, 1]
    y_prob_test = best_rf.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0, 1, 101)
    best_thr = 0.0
    best_f1 = 0.0

    for thr in thresholds:
        y_pred_thr = (y_prob_test >= thr).astype(int)
        f1 = f1_score(y_test, y_pred_thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"\nBest threshold on TEST set: {best_thr:.2f} with F1-score = {best_f1:.3f}")

    #----------------------------------------------------------------------
    # 8) CONFUSION MATRIX & CLASSIFICATION REPORT WITH BEST THRESHOLD
    #----------------------------------------------------------------------
    y_pred_best = (y_prob_test >= best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix (with tuned threshold):")
    print(cm)
    print("\nClassification Report (with tuned threshold):")
    print(classification_report(y_test, y_pred_best, target_names=["NO_FALL", "FALL"]))

if __name__ == "__main__":
    main()

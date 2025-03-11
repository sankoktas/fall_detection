#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def main():
    #----------------------------------------------------------------------
    # 1) LOAD YOUR THREE-CLASS DATA
    #----------------------------------------------------------------------
    input_csv = "data/mar_11_features_3class.csv"
    df = pd.read_csv(input_csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Example 3-class labels: "FALL", "MOTION", "NO MOTION"
    # We'll encode them to 0,1,2
    le = LabelEncoder()
    df["label_num"] = le.fit_transform(df["label"])

    #----------------------------------------------------------------------
    # 2) KEEP TOP 8 FEATURES
    #----------------------------------------------------------------------
    top_8_features = [
        "gyro_median", "gyro_p50", "gyro_mean", "gyro_p75",
        "gyro_p25", "gyro_p10", "gyro_var", "acc_var"
    ]

    X = df[top_8_features].values
    y = df["label_num"].values

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
    # 4) SMOTE (Multi-Class)
    #----------------------------------------------------------------------
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Before SMOTE: {np.bincount(y_train)}")
    print(f"After  SMOTE:  {np.bincount(y_train_res)}")

    #----------------------------------------------------------------------
    # 5) GRID SEARCH FOR RANDOM FOREST
    #----------------------------------------------------------------------
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20]
    }

    rf = RandomForestClassifier(random_state=42)

    # For multi-class, let's use f1_macro or f1_weighted
    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_macro",  
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_res, y_train_res)

    print("\nBest parameters from GridSearch:", grid.best_params_)
    best_rf = grid.best_estimator_

    #----------------------------------------------------------------------
    # 6) EVALUATE (No single threshold – standard multi-class)
    #----------------------------------------------------------------------
    y_pred_test = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    print(f"\nTest Accuracy: {acc:.3f}")

    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

if __name__ == "__main__":
    main()

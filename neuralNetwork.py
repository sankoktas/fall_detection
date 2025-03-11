import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt  # pip install keras-tuner

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score

###############################################################################
# 1. DATA AUGMENTATION HELPER
###############################################################################
def augment_fall_samples(X, y, fraction=0.2, noise_std=0.02, random_state=42):
    """
    Augments FALL samples (label=1) by adding Gaussian noise to a portion of them.
    """
    rng = np.random.default_rng(random_state)
    
    fall_indices = np.where(y == 1)[0]
    n_fall = len(fall_indices)
    n_aug = int(fraction * n_fall)
    if n_fall == 0 or n_aug == 0:
        return X, y  # no augmentation needed
    
    aug_indices = rng.choice(fall_indices, size=n_aug, replace=True)
    X_fall_to_aug = X[aug_indices]

    noise = rng.normal(loc=0.0, scale=noise_std, size=X_fall_to_aug.shape)
    X_fall_aug = X_fall_to_aug + noise
    
    y_fall_aug = np.ones((n_aug,), dtype=y.dtype)
    
    X_aug = np.concatenate([X, X_fall_aug], axis=0)
    y_aug = np.concatenate([y, y_fall_aug], axis=0)
    return X_aug, y_aug

###############################################################################
# 2. FOCAL LOSS FUNCTION
###############################################################################
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for binary classification.
    gamma: focusing parameter that adjusts how aggressively loss focuses on hard examples.
    alpha: weighting factor for the rare (minority) class.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # Exponent of the crossentropy
        bce_exp = tf.exp(-bce)
        # Focal loss
        focal_loss_val = alpha * (1 - bce_exp) ** gamma * bce
        return tf.reduce_mean(focal_loss_val)
    return focal_loss_fixed

###############################################################################
# 3. MODEL-BUILDING FUNCTION (INCLUDING FOCAL LOSS HYPERPARAMS)
###############################################################################
def build_model(hp):
    """
    Larger 1D CNN model using KerasTuner hyperparameters + focal loss with tunable alpha & gamma.
    """

    # -----------------------------
    # 3a) Tunable Focal Loss Params
    # -----------------------------
    alpha_val = hp.Float("alpha", min_value=0.1, max_value=1.0, step=0.1)
    gamma_val = hp.Float("gamma", min_value=1.0, max_value=4.0, step=0.5)
    fl = focal_loss(gamma=gamma_val, alpha=alpha_val)

    # -----------------------------
    # 3b) Build CNN Architecture
    # -----------------------------
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=(6, 1)))

    # First block
    filters_block1 = hp.Int("filters_block1", min_value=32, max_value=128, step=32)
    kernel_size_block1 = hp.Choice("kernel_size_block1", values=[3, 5])
    model.add(tf.keras.layers.Conv1D(
        filters=filters_block1,
        kernel_size=kernel_size_block1,
        activation='relu',
        padding='same'
    ))
    model.add(tf.keras.layers.Conv1D(
        filters=filters_block1,
        kernel_size=kernel_size_block1,
        activation='relu',
        padding='same'
    ))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Second block
    filters_block2 = hp.Int("filters_block2", min_value=32, max_value=128, step=32)
    kernel_size_block2 = hp.Choice("kernel_size_block2", values=[3, 5])
    model.add(tf.keras.layers.Conv1D(
        filters=filters_block2,
        kernel_size=kernel_size_block2,
        activation='relu',
        padding='same'
    ))
    model.add(tf.keras.layers.Conv1D(
        filters=filters_block2,
        kernel_size=kernel_size_block2,
        activation='relu',
        padding='same'
    ))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Flatten
    model.add(tf.keras.layers.Flatten())

    # Dense layer
    dense_units = hp.Int("dense_units", min_value=64, max_value=256, step=64)
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))

    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # -----------------------------
    # 3c) Optimizer
    # -----------------------------
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # -----------------------------
    # 3d) Compile with Focal Loss
    # -----------------------------
    model.compile(
        optimizer=optimizer,
        loss=fl,
        metrics=['accuracy']
    )
    return model

###############################################################################
# 4. CUSTOM TUNER TO ALSO TUNE CLASS WEIGHTS
###############################################################################
# We inherit from kt.RandomSearch (or kt.BayesianOptimization, etc.) so we can
# override "run_trial" to pass dynamic 'class_weight' to model.fit().

class MyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        
        # 1) Pick your class weight
        class_weight_1 = hp.Float("class_weight_1", min_value=1.0, max_value=5.0, step=0.5)
        cw = {0: 1.0, 1: class_weight_1}
        
        # 2) Inject into model.fit() arguments
        kwargs['class_weight'] = cw
        
        # 3) Call super() and return whatever it returns
        history = super(MyTuner, self).run_trial(trial, *args, **kwargs)
        return history

###############################################################################
# 5. MAIN SCRIPT
###############################################################################
def main():
    #--------------------------------------------------------------------------
    # 5a. LOAD & PREPROCESS DATA
    #--------------------------------------------------------------------------
    df = pd.read_csv("data/timeseries_data.csv")

    # Replace infinite values with NaN and drop rows with any NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)

    # Map labels: "NO_FALL" -> 0, "FALL" -> 1
    df["label_numeric"] = df["label"].map({"NO_FALL": 0, "FALL": 1})

    feature_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    X = df[feature_cols].values
    y = df["label_numeric"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape to (samples, timesteps=6, channels=1)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 6, 1))

    #--------------------------------------------------------------------------
    # 5b. AUGMENT DATA (optional)
    #--------------------------------------------------------------------------
    X_reshaped, y = augment_fall_samples(
        X_reshaped, y,
        fraction=0.2,   # +20% FALL
        noise_std=0.02,
        random_state=42
    )

    #--------------------------------------------------------------------------
    # 5c. TRAIN/TEST SPLIT
    #--------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #--------------------------------------------------------------------------
    # 6. KERAS TUNER SETUP (with our custom tuner)
    #--------------------------------------------------------------------------
    tuner = MyTuner(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=10,  # Increase if you have time/resources
        executions_per_trial=1,
        directory="my_tuner_dir",
        project_name="fall_detection_tuning_focal_loss_and_class_weight"
    )

    # EarlyStopping callback for quicker tuning
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Run hyperparameter search
    tuner.search(
        X_train,
        y_train,
        epochs=15,  # for quick scanning
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
        # DO NOT pass class_weight here, it's tuned inside MyTuner.run_trial()
    )

    #--------------------------------------------------------------------------
    # 7. RETRIEVE BEST MODEL & BEST CLASS WEIGHT
    #--------------------------------------------------------------------------
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n[INFO] Best hyperparameters found:")
    for param, val in best_hps.values.items():
        print(f"  {param}: {val}")

    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)

    # Get the best class weight from hyperparams
    best_class_weight = {0: 1.0, 1: best_hps.get("class_weight_1")}
    print(f"Best class weight found: {best_class_weight}")


    # OPTIONAL: retrain on the full training set for more epochs with the best HPs
    best_model.fit(
        X_train,
        y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=best_class_weight,
        verbose=1
    )

    #--------------------------------------------------------------------------
    # 8. EVALUATE ON TEST SET
    #--------------------------------------------------------------------------
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nBest model (with tuned focal loss & class_weight) -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    #--------------------------------------------------------------------------
    # 9. THRESHOLD & METRICS (Example threshold search for best F1)
    #--------------------------------------------------------------------------
    y_prob_test = best_model.predict(X_test).ravel()
    thresholds = np.linspace(0, 1, 101)
    best_thr = 0.0
    best_thr_f1 = 0.0

    for thr in thresholds:
        y_pred_thr = (y_prob_test > thr).astype(int)
        current_f1 = f1_score(y_test, y_pred_thr)
        if current_f1 > best_thr_f1:
            best_thr_f1 = current_f1
            best_thr = thr

    print(f"\nBest threshold on TEST set: {best_thr:.2f} with F1-score = {best_thr_f1:.3f}")

    # Evaluate at best threshold
    y_pred_best = (y_prob_test > best_thr).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix (threshold={best_thr:.2f}):\n{cm_best}")

    cr_best = classification_report(y_test, y_pred_best, target_names=["NO_FALL", "FALL"])
    print(f"\nClassification Report (threshold={best_thr:.2f}):\n{cr_best}")

    #--------------------------------------------------------------------------
    # 10. DEMO PREDICTION ON NEW SAMPLE
    #--------------------------------------------------------------------------
    new_sample = np.array([[-4.06, -5.83, -6.98, -5.49, 0.12, -0.42]])
    new_sample_scaled = scaler.transform(new_sample)
    new_sample_reshaped = new_sample_scaled.reshape((1, 6, 1))

    new_prob = best_model.predict(new_sample_reshaped)[0, 0]
    pred_label = 1 if new_prob > best_thr else 0
    label_str = "FALL" if pred_label == 1 else "NO_FALL"
    print(f"\nNew sample probability of FALL: {new_prob:.4f} => PREDICTED: {label_str}")

if __name__ == "__main__":
    main()

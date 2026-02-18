"""
LSTM Air Quality - Incremental Training on Last 24 Hours (No Prediction)
"""

import numpy as np
import tensorflow as tf
import joblib
import os
import hopsworks
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
POLLUTANT_COLS   = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
LOOKBACK_HOURS   = 168
PREDICTION_HOURS = 72
INCREMENTAL_EPOCHS = 20
BATCH_SIZE       = 32
PATIENCE         = 10
INCREMENTAL_LR   = 0.0001

# ============================================================================
# FUNCTIONS
# ============================================================================

def create_incremental_sequences(X_data, y_data, lookback, horizon, new_hours=24):
    X, y = [], []
    total     = len(X_data)
    start_idx = total - new_hours - horizon
    end_idx   = total - horizon

    for i in range(start_idx, end_idx):
        if i < lookback:
            continue
        X.append(X_data[i - lookback : i])
        y.append(y_data[i : i + horizon])
    return np.array(X), np.array(y)


def run_incremental_training():

    # ------------------------------------------------------------------ #
    # STEP 1: LOAD MODEL AND SCALERS FROM HOPSWORKS
    # ------------------------------------------------------------------ #
    print("STEP 1: Loading model and scalers from Hopsworks...")

    project = hopsworks.login(
        project="AQI_predict_P",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    mr            = project.get_model_registry()
    lstm_model_hw = mr.get_model("LSTM_model", version=1)
    model_dir     = lstm_model_hw.download()

    model    = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model.keras"))
    scalers  = joblib.load(os.path.join(model_dir, "scalers.pkl"))
    scaler_X = scalers['scaler_X']
    scaler_Y = scalers['scaler_y']

    print("Model and scalers loaded ✓")

    # ------------------------------------------------------------------ #
    # STEP 2: FETCH RECENT DATA FROM FEATURE STORE
    # ------------------------------------------------------------------ #
    print("\nSTEP 2: Fetching recent data from Feature Store...")

    fs           = project.get_feature_store()
    feature_view = fs.get_feature_view(name="LSTM_features", version=1)

    rows_needed = LOOKBACK_HOURS + 24 + PREDICTION_HOURS  # = 264

    full_df   = feature_view.get_batch_data()
    full_df   = full_df.sort_values('datetime')
    recent_df = full_df.tail(rows_needed).reset_index(drop=True)

    ALL_FEATURES = [col for col in recent_df.columns if col != 'datetime']
    TARGET_COLS  = [col for col in POLLUTANT_COLS if col in ALL_FEATURES]

    print(f"Rows fetched : {len(recent_df)}  (need ≥ {rows_needed})")
    assert len(recent_df) >= rows_needed, \
        f"Not enough data! Need {rows_needed} rows, got {len(recent_df)}"

    # ------------------------------------------------------------------ #
    # STEP 3: SCALE USING EXISTING SCALERS — DO NOT REFIT
    # ------------------------------------------------------------------ #
    print("\nSTEP 3: Scaling data...")

    X_scaled = scaler_X.transform(recent_df[ALL_FEATURES].values)
    y_scaled = scaler_Y.transform(recent_df[TARGET_COLS].values)

    # ------------------------------------------------------------------ #
    # STEP 4: CREATE SEQUENCES FROM THE NEW 24-HOUR WINDOW ONLY
    # ------------------------------------------------------------------ #
    print("\nSTEP 4: Building sequences from new 24-hour window...")

    X_new, y_new = create_incremental_sequences(
        X_scaled, y_scaled,
        lookback=LOOKBACK_HOURS,
        horizon=PREDICTION_HOURS,
        new_hours=24
    )

    print(f"Sequences created — X: {X_new.shape}, y: {y_new.shape}")
    assert X_new.shape[0] > 0, "No sequences created! Check data length."

    # ------------------------------------------------------------------ #
    # STEP 5: TRAIN / VAL SPLIT
    # ------------------------------------------------------------------ #
    val_size = max(1, int(0.2 * len(X_new)))

    X_train, y_train = X_new[:-val_size], y_new[:-val_size]
    X_val,   y_val   = X_new[-val_size:],  y_new[-val_size:]

    print(f"Train sequences : {len(X_train)}")
    print(f"Val   sequences : {len(X_val)}")

    # ------------------------------------------------------------------ #
    # STEP 6: RECOMPILE WITH LOWER LR AND FINE-TUNE
    # ------------------------------------------------------------------ #
    print("\nSTEP 6: Fine-tuning model...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INCREMENTAL_LR),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-8, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=INCREMENTAL_EPOCHS,
        batch_size=min(BATCH_SIZE, len(X_train)),
        callbacks=callbacks,
        verbose=1
    )

    # ------------------------------------------------------------------ #
    # STEP 7: EVALUATE ON VALIDATION SEQUENCES
    # ------------------------------------------------------------------ #
    print("\nSTEP 7: Evaluating updated model...")

    val_pred_scaled = model.predict(X_val)
    val_pred  = scaler_Y.inverse_transform(
        val_pred_scaled.reshape(-1, len(TARGET_COLS))
    ).reshape(val_pred_scaled.shape)

    y_val_real = scaler_Y.inverse_transform(
        y_val.reshape(-1, len(TARGET_COLS))
    ).reshape(y_val.shape)

    metrics = {}
    print(f"\n{'='*50}\nIncremental Validation Performance\n{'='*50}")
    for i, col in enumerate(TARGET_COLS):
        mae  = mean_absolute_error(y_val_real[:,:,i].flatten(), val_pred[:,:,i].flatten())
        rmse = np.sqrt(mean_squared_error(y_val_real[:,:,i].flatten(), val_pred[:,:,i].flatten()))
        metrics[f"mae_{col}"]  = mae
        metrics[f"rmse_{col}"] = rmse
        print(f"{col:8s} → MAE: {mae:.3f}   RMSE: {rmse:.3f}")

    # ------------------------------------------------------------------ #
    # STEP 8: SAVE UPDATED MODEL TO HOPSWORKS AS NEW VERSION
    # ------------------------------------------------------------------ #
    print("\nSTEP 8: Saving updated model to Hopsworks...")

    os.makedirs("lstm_model_dir_updated", exist_ok=True)
    model.save("lstm_model_dir_updated/lstm_model.keras")
    joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_Y},
                "lstm_model_dir_updated/scalers.pkl")

    with open("lstm_model_dir_updated/update_log.txt", "a") as f:
        f.write(f"{pd.Timestamp.now()} — incremental update on last 24h data\n")

    existing_versions = [m.version for m in mr.get_models("LSTM_model")]
    new_version       = max(existing_versions) + 1

    updated_model = mr.tensorflow.create_model(
        name="LSTM_model",
        version=new_version,
        description=f"Incremental update: {pd.Timestamp.now().date()} — last 24h data",
        metrics=metrics
    )
    updated_model.save("lstm_model_dir_updated")

    print(f"Model saved to Hopsworks as version {new_version} ✓")
    print("\n" + "="*60)
    print("INCREMENTAL TRAINING COMPLETE")
    print("="*60)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_incremental_training()

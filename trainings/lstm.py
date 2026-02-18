import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import hopsworks
import joblib
import os

"""
LSTM Air Quality Prediction 
"""

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION 
# ============================================================================

# Column names in your CSV files
POLLUTANT_COLS = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']

# Model parameters
LOOKBACK_HOURS = 168
PREDICTION_HOURS = 120
LSTM_UNITS = [128, 64]    # LSTM layer sizes
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 30

# ============================================================================
# FUNCTIONS:
# ============================================================================

def create_sequences(X_data, y_data, lookback, horizon):
    X, y = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X.append(X_data[i : i + lookback])             # past window  → input
        y.append(y_data[i + lookback : i + lookback + horizon])  # future window → target
    return np.array(X), np.array(y)

def evaluate_split(actual, predicted, split_name):
    # Flatten for metric calculation
    actual_flat    = actual.reshape(-1, len(TARGET_COLS))
    predicted_flat = predicted.reshape(-1, len(TARGET_COLS))

    mae_dict  = {}
    rmse_dict = {}

    print(f"\n{'='*50}")
    print(f"{split_name} Performance")
    print(f"{'='*50}")

    for i, pollutant in enumerate(TARGET_COLS):
        mae  = mean_absolute_error(actual_flat[:, i], predicted_flat[:, i])
        rmse = np.sqrt(mean_squared_error(actual_flat[:, i], predicted_flat[:, i]))

        mae_dict[pollutant]  = mae
        rmse_dict[pollutant] = rmse

        print(f"{pollutant:8s} → MAE: {mae:.3f}   RMSE: {rmse:.3f}")
    return mae_dict, rmse_dict



# ============================================================================
# STEP 1: LOAD DATA FROM HOPSWORKS
# ============================================================================
project = hopsworks.login(
    project="AQI_predict_P",
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()
feature_view = fs.get_feature_view(name="LSTM_features", version=1)
merged_df = feature_view.get_batch_data()    

ALL_FEATURES = merged_df.columns.tolist()
TARGET_COLS = [col for col in POLLUTANT_COLS if col in ALL_FEATURES]

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================

# STEP 2.1: SPLIT THE DATAFRAME (chronologically)
print("STEP 2.1: SPLITTING DATA")

n = len(merged_df)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)   # 70% + 15% = 85%

train_df = merged_df.iloc[:train_end]
val_df   = merged_df.iloc[train_end:val_end]
test_df  = merged_df.iloc[val_end:]

print(f"Total rows      : {n}")
print(f"Training rows   : {len(train_df)}  ({len(train_df)/n*100:.1f}%)者に教えてください。")
print(f"Validation rows : {len(val_df)}  ({len(val_df)/n*100:.1f}%)者に教えてください。")
print(f"Test rows       : {len(test_df)}  ({len(test_df)/n*100:.1f}%)者に教えてください。")

# STEP 2.2: FIT SCALERS ON TRAINING DATA ONLY
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
scaler_X.fit(train_df[ALL_FEATURES].values)
scaler_Y.fit(train_df[TARGET_COLS].values)
print("Scalers fitted on training data only")

# STEP 2.3: TRANSFORM ALL THREE SPLITS
train_X_scaled = scaler_X.transform(train_df[ALL_FEATURES].values)
val_X_scaled   = scaler_X.transform(val_df[ALL_FEATURES].values)
test_X_scaled  = scaler_X.transform(test_df[ALL_FEATURES].values)

train_y_scaled = scaler_Y.transform(train_df[TARGET_COLS].values)
val_y_scaled   = scaler_Y.transform(val_df[TARGET_COLS].values)
test_y_scaled  = scaler_Y.transform(test_df[TARGET_COLS].values)

print(f"Train X scaled shape : {train_X_scaled.shape}")
print(f"Train y scaled shape : {train_y_scaled.shape}")
print(f"Val   X scaled shape : {val_X_scaled.shape}")
print(f"Val   y scaled shape : {val_y_scaled.shape}")
print(f"Test  X scaled shape : {test_X_scaled.shape}")
print(f"Test  y scaled shape : {test_y_scaled.shape}")


# STEP 2.4: BUILD SEQUENCES
X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, LOOKBACK_HOURS, PREDICTION_HOURS)
X_val,   y_val   = create_sequences(val_X_scaled,   val_y_scaled,   LOOKBACK_HOURS, PREDICTION_HOURS)
X_test,  y_test  = create_sequences(test_X_scaled,  test_y_scaled,  LOOKBACK_HOURS, PREDICTION_HOURS)

print(f"X_train : {X_train.shape}   → (sequences, lookback hours, features)")
print(f"y_train : {y_train.shape}   → (sequences, horizon hours,  pollutants)")
print(f"X_val   : {X_val.shape}")
print(f"y_val   : {y_val.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")


# STEP 2.5: SANITY CHECK
assert X_train.shape[1] == LOOKBACK_HOURS,   "Lookback mismatch in X_train"
assert y_train.shape[1] == PREDICTION_HOURS, "Horizon mismatch in y_train"
assert X_train.shape[2] == len(ALL_FEATURES),"Feature count mismatch in X_train"
assert y_train.shape[2] == len(TARGET_COLS), "Target count mismatch in y_train"

print(f"Input  features  : {X_train.shape[2]}  {ALL_FEATURES}")
print(f"Target features  : {y_train.shape[2]}  {TARGET_COLS}")
print(f"Lookback         : {X_train.shape[1]} hours (7 days)")
print(f"Prediction horizon: {y_train.shape[1]} hours (3 days)")
print("All checks passed ✓")
print ("PREPARING DATA STEP COMPLETED.")
print ('_' * 70 )

# ============================================================================
# STEP 3: BUILD LSTM MODEL
# ============================================================================
n_features = len(ALL_FEATURES)
n_pollutants = len(POLLUTANT_COLS)

model = Sequential()

# LSTM layers
model.add(LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=(LOOKBACK_HOURS, n_features)))
model.add(Dropout(DROPOUT_RATE))

model.add(LSTM(LSTM_UNITS[1], return_sequences=False))
model.add(Dropout(DROPOUT_RATE))

# Dense layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(DROPOUT_RATE))

# Output layer
model.add(Dense(PREDICTION_HOURS * n_pollutants))
model.add(tf.keras.layers.Reshape((PREDICTION_HOURS, n_pollutants)))

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=[
        'mae',
         tf.keras.metrics.RootMeanSquaredError(name='rmse')
    ])

print(model.summary())
print ("BUILDING MODEL STEP COMPLETED.")
print ('_' * 70)

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# STEP 5: SAVING MODEL AND SCALERS
# ============================================================================
# Create directory
os.makedirs("lstm_model_dir", exist_ok=True)

# Save model
model.save("lstm_model_dir/lstm_model.keras")

# Save scalers
joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_Y}, "lstm_model_dir/scalers.pkl")

# ============================================================================
# STEP 6: PREDICTIONS ON TRAIN,TEST,VALIDATE SETS TO EVALUATE MODEL PERFORMANCE
# ============================================================================
# Get scaled predictions
train_pred_scaled = model.predict(X_train)
val_pred_scaled   = model.predict(X_val)
test_pred_scaled  = model.predict(X_test)

# Inverse transform back to real values
train_pred = scaler_Y.inverse_transform(train_pred_scaled.reshape(-1, len(TARGET_COLS))).reshape(train_pred_scaled.shape)
val_pred   = scaler_Y.inverse_transform(val_pred_scaled.reshape(-1, len(TARGET_COLS))).reshape(val_pred_scaled.shape)
test_pred  = scaler_Y.inverse_transform(test_pred_scaled.reshape(-1, len(TARGET_COLS))).reshape(test_pred_scaled.shape)

# Inverse transform actual values too
y_train_real = scaler_Y.inverse_transform(y_train.reshape(-1, len(TARGET_COLS))).reshape(y_train.shape)
y_val_real   = scaler_Y.inverse_transform(y_val.reshape(-1, len(TARGET_COLS))).reshape(y_val.shape)
y_test_real  = scaler_Y.inverse_transform(y_test.reshape(-1, len(TARGET_COLS))).reshape(y_test.shape)

# Evaluating performance
mae_train, rmse_train = evaluate_split(y_train_real, train_pred, "TRAIN")
mae_validate, rmse_validate = evaluate_split(y_val_real,   val_pred,   "VALIDATION")
mae_test, rmse_test= evaluate_split(y_test_real,  test_pred,  "TEST")

#storing in dictionary: 
metrics = {f"mae_{col}": val for col, val in mae_test.items()}
metrics.update({f"rmse_{col}": val for col, val in rmse_test.items()})

# ============================================================================
# STEP 7: SAVING TO HOPSWORKS
# ============================================================================

# Connect to Hopsworks
project = hopsworks.login(
    project="AQI_predict_P",
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=api_key
)
mr = project.get_model_registry()

# Register and upload
lstm_model = mr.tensorflow.create_model(
    name="LSTM_model",
    version=1,
    description="LSTM model for prediction",
    metrics=metrics
)

lstm_model.save("lstm_model_dir")  # uploads entire directory (model + scalers)

print("Model saved to Hopsworks Model Registry")
print("Scalers saved alongside model")
print("SAVING MODEL STEP COMPLETED.")
print('_' * 70)

# ============================================================================
# STEP 8: MAKE PREDICTIONS FOR NEXT 3 DAYS
# ============================================================================
# Load recent data (last 168 hours from original merged data)
recent_data = merged_df[ALL_FEATURES].iloc[-LOOKBACK_HOURS:].values
recent_data_scaled = scaler_X.transform(recent_data)

# Reshape for prediction
X_pred = recent_data_scaled.reshape(1, LOOKBACK_HOURS, n_features)

# Predict
print("\nGenerating predictions...")
predictions_scaled = model.predict(X_pred, verbose=0)

# Inverse transform
predictions = scaler_Y.inverse_transform(predictions_scaled.reshape(-1, n_pollutants))

# Create DataFrame with predictions
last_datetime = merged_df['datetime'].iloc[-1]
future_dates = pd.date_range(
    start=last_datetime + pd.Timedelta(hours=1),
    periods=PREDICTION_HOURS,
    freq='H'
)

predictions_df = pd.DataFrame(
    predictions,
    index=future_dates,
    columns=POLLUTANT_COLS
)

# Save predictions
output_file = 'predictions_3_days.csv'
predictions_df.to_csv(output_file)

print(f"\nPredictions saved to: {output_file}")
print(f"Total predictions: {len(predictions_df)} hours")
print(f"Prediction period: {predictions_df.index[0]} to {predictions_df.index[-1]}")

print("\nPrediction summary statistics:")
print(predictions_df.describe())

print("\n" + "="*70)
print("DONE!")
print("="*70)
print("\nGenerated files:")
print("  1. pollutant_lstm_model.h5 - Trained model")
print("  2. scalers.pkl - Data scalers")
print("  3. predictions_3_days.csv - 3-day predictions") # Corrected comment here as well

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickleimport 


#=========================================================================================================================================
#SMALL GAPS: (24-48 HOURS)
#==========================================================================================================================================
# ============= CONFIGURATION =============
CSV_FILE = '/content/aqi_data_complete_with_gaps.csv'  # Your input file name
DATETIME_COLUMN = 'datetime_pkt'     # Name of your datetime column
POLLUTANT_COLUMNS = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']  # Your pollutant columns
OUTPUT_FILE = 'aqi_data_filled.csv'  # Output file name

# Gap filling parameters
MIN_GAP_HOURS = 24  # Minimum gap size to fill
MAX_GAP_HOURS = 48  # Maximum gap size to fill
# =========================================


def main():
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)

    # Convert datetime column
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN])
    df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    print(f"Total records: {len(df)}")
    print(f"Date range: {df[DATETIME_COLUMN].min()} to {df[DATETIME_COLUMN].max()}")

    # Analyze gaps before filling
    print("\n" + "="*50)
    print("GAP ANALYSIS BEFORE FILLING")
    print("="*50)
    analyze_gaps(df, POLLUTANT_COLUMNS)

    # Fill gaps using linear interpolation
    print("\n" + "="*50)
    print(f"FILLING GAPS ({MIN_GAP_HOURS}-{MAX_GAP_HOURS} HOURS)")
    print("="*50)
    df_filled = fill_gaps_linear_interpolation(
        df,
        POLLUTANT_COLUMNS,
        MIN_GAP_HOURS,
        MAX_GAP_HOURS
    )

    # Analyze gaps after filling
    print("\n" + "="*50)
    print("GAP ANALYSIS AFTER FILLING")
    print("="*50)
    analyze_gaps(df_filled, POLLUTANT_COLUMNS)

    # Save result
    df_filled.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Filled dataset saved to: {OUTPUT_FILE}")


def analyze_gaps(df, columns):
    """Analyze and display gap information"""
    # Check for missing values (any pollutant column is null)
    missing_mask = df[columns].isnull().any(axis=1)
    total_missing = missing_mask.sum()

    print(f"Total records with missing values: {total_missing}")

    if total_missing == 0:
        print("No gaps found!")
        return

    # Identify consecutive gaps
    df_temp = df.copy()
    df_temp['is_missing'] = missing_mask

    # Create gap groups (consecutive missing records get same ID)
    df_temp['gap_id'] = (df_temp['is_missing'] != df_temp['is_missing'].shift()).cumsum()

    # Only consider groups that are actually gaps (is_missing = True)
    gaps = df_temp[df_temp['is_missing']].groupby('gap_id').size()

    print(f"Number of separate gaps: {len(gaps)}")
    print(f"\nGap size distribution:")

    gap_counts = {}
    for size in gaps.values:
        gap_counts[size] = gap_counts.get(size, 0) + 1

    for size in sorted(gap_counts.keys()):
        count = gap_counts[size]
        print(f"  {size}-hour gaps: {count}")

    # Highlight 24-48 hour gaps
    gaps_24_48 = gaps[(gaps >= 24) & (gaps <= 48)]
    print(f"\nGaps eligible for interpolation (24-48 hours): {len(gaps_24_48)}")
    if len(gaps_24_48) > 0:
        print(f"  Total hours to fill: {gaps_24_48.sum()}")


def fill_gaps_linear_interpolation(df, columns, min_gap, max_gap):
    """Fill gaps using linear interpolation for specified gap sizes"""
    df_result = df.copy()

    # Identify missing records
    missing_mask = df_result[columns].isnull().any(axis=1)

    # Create gap IDs for consecutive missing records
    gap_id = (missing_mask != missing_mask.shift()).cumsum()

    # For each gap group, get its size
    gap_info = df_result.groupby(gap_id).agg({
        columns[0]: lambda x: x.isnull().all()  # Check if this group is a gap
    })
    gap_info.columns = ['is_gap']

    gap_sizes = df_result[missing_mask].groupby(gap_id).size()

    # Identify which gaps to fill (24-48 hours)
    gaps_to_fill = gap_sizes[(gap_sizes >= min_gap) & (gap_sizes <= max_gap)]

    print(f"Found {len(gaps_to_fill)} gaps to fill")

    total_filled = 0

    # Fill each pollutant column separately
    for col in columns:
        print(f"\n  Processing {col}...")
        col_filled = 0

        for gap_group_id in gaps_to_fill.index:
            # Get indices for this gap
            gap_mask = (gap_id == gap_group_id) & missing_mask
            gap_indices = df_result[gap_mask].index.tolist()

            if len(gap_indices) == 0:
                continue

            # Get boundary indices
            start_idx = gap_indices[0]
            end_idx = gap_indices[-1]

            # Get previous valid value (just before the gap)
            prev_idx = start_idx - 1

            # Get next valid value (just after the gap)
            next_idx = end_idx + 1

            # Check if boundaries exist and have valid values
            if prev_idx >= 0 and next_idx < len(df_result):
                prev_val = df_result.loc[prev_idx, col]
                next_val = df_result.loc[next_idx, col]

                # Only interpolate if both boundaries are non-null
                if pd.notna(prev_val) and pd.notna(next_val):
                    # Linear interpolation
                    n_points = len(gap_indices)

                    # Generate interpolated values
                    # Formula: value_i = prev_val + (next_val - prev_val) * (i+1) / (n_points+1)
                    interpolated = []
                    for i in range(n_points):
                        value = prev_val + (next_val - prev_val) * (i + 1) / (n_points + 1)
                        interpolated.append(round(value,2))

                    # Assign interpolated values
                    df_result.loc[gap_indices, col] = interpolated
                    col_filled += n_points
                    print(f"    Gap at index {start_idx}-{end_idx}: filled {n_points} values")
                else:
                    print(f"    Gap at index {start_idx}-{end_idx}: skipped (boundary values are null)")
            else:
                print(f"    Gap at index {start_idx}-{end_idx}: skipped (at dataset boundary)")

        print(f"  → {col}: {col_filled} values filled")
        total_filled += col_filled

    print(f"\n✓ Total values filled: {total_filled}")

    return df_result


if __name__ == "__main__":
    main()

#==================================================================================================================================================================================
# LARGE GAPS: (120 HOUR)
#===================================================================================================================================================================================

"""
LSTM Air Quality Prediction - Straightforward Implementation
Loads first 19,660 records from separate CSV files, trains model, predicts next 7 days
"""

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

WEATHER_FILE = '/content/wind_data_transformed.csv'
AQI_FILE = '/content/aqi_data_filled (1).csv'
time_file = '/content/time_features.csv'
WEATHER_DATETIME_COL = 'datetime'
AQI_DATETIME_COL = 'datetime_pkt'
time_datetime_col = 'datetime_unix'

# Column names in your CSV files
WEATHER_COLS = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'precipitation (mm)','surface_pressure (hPa)','cloud_cover (%)','wind_speed_10m (m/s)', 'shortwave_radiation (W/m²)', 'wind_direction_sin','wind_direction_cos']
POLLUTANT_COLS = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
time_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

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

    print(f"\n{'='*50}")
    print(f"{split_name} Performance")
    print(f"{'='*50}")

    for i, pollutant in enumerate(TARGET_COLS):
        mae  = mean_absolute_error(actual_flat[:, i], predicted_flat[:, i])
        rmse = np.sqrt(mean_squared_error(actual_flat[:, i], predicted_flat[:, i]))
        print(f"{pollutant:8s} → MAE: {mae:.3f}   RMSE: {rmse:.3f}")



# ============================================================================
# STEP 1: LOAD DATA (First 19,660 records only)
# ============================================================================
# Loading weather data:
print(f"\nLoading {WEATHER_FILE}...")
weather_df = pd.read_csv(WEATHER_FILE, parse_dates=[WEATHER_DATETIME_COL])
weather_df = weather_df.iloc[:19663]
print(f"Weather data loaded: {weather_df.shape}")

# Loading AQI data:
print(f"\nLoading {AQI_FILE}...")
aqi_df = pd.read_csv(AQI_FILE, parse_dates=[AQI_DATETIME_COL])
aqi_df = aqi_df.iloc[:19660]
print(f"AQI data loaded: {aqi_df.shape}")

# loading time featuers:
print(f"\nLoading {time_file}...")
time_df = pd.read_csv(time_file)
# Convert 'datetime_unix' to datetime objects (assuming Unix timestamps in seconds)
time_df[time_datetime_col] = pd.to_datetime(time_df[time_datetime_col], unit='s')
time_df = time_df.iloc[:19660]
print(f"time data loaded: {time_df.shape}")

# renaming datetime column in all three df to be same:
aqi_df = aqi_df.rename(columns={AQI_DATETIME_COL: 'datetime'})
weather_df = weather_df.rename(columns={WEATHER_DATETIME_COL: 'datetime'})
time_df = time_df.rename(columns={time_datetime_col: 'datetime'})

# merging data-frames:
merged_df = aqi_df.merge(weather_df, on='datetime', how='inner')
merged_df = merged_df.merge(time_df, on='datetime', how='inner')
print(f"Merged data frame: {merged_df.shape}")
print ("LOADING DATA STEP COMPLETED.")
print ('_' * 70)


# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================

ALL_FEATURES = WEATHER_COLS + POLLUTANT_COLS + time_cols  #(21 total)
TARGET_COLS  = POLLUTANT_COLS                               #(6 total)

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
model.save('pollutant_lstm_model.h5')
with open('scalers.pkl', 'wb') as f:
    pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_Y}, f)

print("Model saved to: pollutant_lstm_model.h5")
print("Scalers saved to: scalers.pkl")
print ("SAVING MODEL STEP COMPLETED.")
print ('_' * 70)

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
evaluate_split(y_train_real, train_pred, "TRAIN")
evaluate_split(y_val_real,   val_pred,   "VALIDATION")
evaluate_split(y_test_real,  test_pred,  "TEST")

# ============================================================================
# STEP 7: MAKE PREDICTIONS FOR NEXT 3 DAYS
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























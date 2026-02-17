import math
from datetime import datetime
import pytz
import hopsworks
import pandas as pd
import hopsworks
import pandas as pd
z

def cyclic (angle: float, total_angle: float) -> tuple[float, float]:
    if total_angle == 0:
        raise ValueError("total_angle must be non-zero.")

    radians = 2 * math.pi * (angle / total_angle)
    sin_val = math.sin(radians)
    cos_val = math.cos(radians)
    return sin_val, cos_val


def utc_unix_to_pst(utc_unix_timestamp: int | float) -> tuple:
    if not isinstance(utc_unix_timestamp, (int, float)):
        raise ValueError("Input must be an int or float Unix timestamp.")

    pst_zone        = pytz.timezone("Asia/Karachi")
    utc_dt          = datetime.fromtimestamp(utc_unix_timestamp, tz=pytz.utc)
    pst_dt          = utc_dt.astimezone(pst_zone)

    pst_offset_secs = int(pst_dt.utcoffset().total_seconds())
    unix_pst        = int(utc_unix_timestamp) + pst_offset_secs
    datetime_pst    = pst_dt.strftime("%Y-%m-%d %H:%M:%S")

    return unix_pst, datetime_pst


ef extract_time_features(unix_timestamp: int | float) -> tuple:
    if not isinstance(unix_timestamp, (int, float)):
        raise ValueError("Input must be an int or float Unix timestamp.")

    pst_zone = pytz.timezone("Asia/Karachi")
    dt    = datetime.fromtimestamp(unix_timestamp, tz=pst_zone)
    hour  = dt.hour
    day   = dt.weekday()
    month = dt.month

    return hour, day, month

def get_lag_feature(
    column_name       : str,
    unix_timestamp    : int ,
    lag_hours         : int,
    feature_group_name: str,
    feature_group_ver : int
) -> float:
    #Calculate lagged timestamp
    lag_offset_secs = lag_hours * 3600
    lagged_unix     = int(unix_timestamp) - lag_offset_secs
    #Connect to Hopsworks
    project = hopsworks.connect()
    fs = project.get_feature_store()

    #Get feature group
    feature_group = fs.get_feature_group(
                            name    = feature_group_name,
                            version = feature_group_ver
                      )

    #Build filter and read only the lagged row
    filterr = feature_group.feature("unix_timestamp") == lagged_unix
    value  = feature_group.filter(filterr).read()

    #Validate and return
    if value.empty:
        raise ValueError(
            f"No data found for column '{column_name}' "
            f"at lagged timestamp {lagged_unix} ({lag_hours}h lag from {unix_timestamp})."
        )

    return value


def get_rolling_feature(
    column_name       : str,
    unix_timestamp    : int | float,
    rolling_hours     : int,
    feature_group_name: str,
    feature_group_ver : int
) -> float:

    #Generate all expected timestamps in the rolling window
    current_unix = int(unix_timestamp)
    expected_timestamps = [current_unix - (i * 3600) for i in range(1, rolling_hours + 1)]

    #Connect to Hopsworks
    project = hopsworks.connect()
    fs = project.get_feature_store()

    #Get feature group
    feature_group = fs.get_feature_group(
                                name    = feature_group_name,
                                version = feature_group_ver
                         )

    #Filter only the rows within the rolling window
    filterr = feature_group.feature("unix_timestamp") >= expected_timestamps[-1]
    filterr &= feature_group.feature("unix_timestamp") <  current_unix
    df = feature_group.filter(filterr).read()

    #Validate all expected timestamps are present
    fetched_timestamps = set(df["unix_timestamp"].tolist())
    missing_timestamps = [ts for ts in expected_timestamps if ts not in fetched_timestamps]

    if missing_timestamps:
        raise ValueError(
            f"Missing timestamps in rolling window for column '{column_name}': "
            f"{missing_timestamps}"
        )

    #Calculate and return rolling mean
    return df[column_name].mean()


def convert_concentrations(components: dict) -> dict:
  
    # Molecular weights (g/mol)
    MW = {
        "co":  28.01,
        "o3":  48.00,
        "no2": 46.00,
        "so2": 64.06,
    }
    # µg/m³ to ppm:  value / (MW * 1000 / 24.45)
    # µg/m³ to ppb:  value / (MW * 1000 / 24450)
    # 24.45 L/mol is molar volume at 25°C, 1 atm

    converted = {}

    # CO → ppm
    if "co" in components:
        converted["co_ppm"] = round(components["co"] * 24.45 / (MW["co"] * 1000), 4)

    # O3, NO2, SO2 → ppb
    for pollutant in ["o3", "no2", "so2"]:
        if pollutant in components:
            converted[f"{pollutant}_ppb"] = round(components[pollutant] * 24.45 / MW[pollutant], 4)

    return converted


def calculate_aqi(
    timestamp_unix : int | float,
    pm2_5          : float,
    pm10           : float,
    o3             : float,
    no2            : float,
    so2            : float,
    co             : float,
) -> dict:
    #Define AQI Breakpoints (US EPA Standard)
    aqi_breakpoints = {
        "pm2_5": [
            (0.0,   12.0,   0,   50),
            (12.1,  35.4,  51,  100),
            (35.5,  55.4, 101,  150),
            (55.5, 150.4, 151,  200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ],
        "pm10": [
            (0,    54,    0,   50),
            (55,   154,  51,  100),
            (155,  254, 101,  150),
            (255,  354, 151,  200),
            (355,  424, 201,  300),
            (425,  504, 301,  400),
            (505,  604, 401,  500),
        ],
        "o3_8h": [
            (0,    54,    0,   50),
            (55,   70,   51,  100),
            (71,   85,  101,  150),
            (86,  105,  151,  200),
            (106, 200,  201,  300),
        ],
        "o3_1h": [
            (125, 164,  101,  150),
            (165, 204,  151,  200),
            (205, 404,  201,  300),
            (405, 504,  301,  400),
            (505, 604,  401,  500),
        ],
        "no2": [
            (0,    53,    0,   50),
            (54,   100,  51,  100),
            (101,  360, 101,  150),
            (361,  649, 151,  200),
            (650, 1249, 201,  300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500),
        ],
        "so2": [
            (0,    35,    0,   50),
            (36,   75,   51,  100),
            (76,  185,  101,  150),
            (186, 304,  151,  200),
            (305, 604,  201,  300),
            (605, 804,  301,  400),
            (805, 1004, 401,  500),
        ],
        "co": [
            (0.0,   4.4,   0,   50),
            (4.5,   9.4,  51,  100),
            (9.5,  12.4, 101,  150),
            (12.5, 15.4, 151,  200),
            (15.5, 30.4, 201,  300),
            (30.5, 40.4, 301,  400),
            (40.5, 50.4, 401,  500),
        ],
    }

    aqi_categories = {
        (0,   50) : "Good",
        (51,  100): "Moderate",
        (101, 150): "Unhealthy for Sensitive Groups",
        (151, 200): "Unhealthy",
        (201, 300): "Very Unhealthy",
        (301, 500): "Hazardous",
    }

    #Get hour of day in PST
    pst_zone    = pytz.timezone("Asia/Karachi")
    pst_dt      = datetime.fromtimestamp(timestamp_unix, tz=pst_zone)
    hour_of_day = pst_dt.hour

    #Connect to Hopsworks
    project       = hopsworks.connect()
    fs            = project.get_feature_store()
    feature_group = fs.get_feature_group(name="aqi_data", version=2)

    #Fetch 8-hour data for O3 and CO rolling means
    window_start_8h = int(timestamp_unix) - (8 * 3600)
    filter_8h       = feature_group.feature("timestamp_unix") >= window_start_8h
    filter_8h      &= feature_group.feature("timestamp_unix") <  int(timestamp_unix)
    df_8h           = feature_group.filter(filter_8h).read()

    if len(df_8h) < 8:
        raise ValueError(
            f"Expected 8 past records for O3/CO rolling mean but only found {len(df_8h)}. "
            f"{8 - len(df_8h)} timestamp(s) missing in the feature group."
        )

    df_8h      = df_8h.sort_values("timestamp_unix").reset_index(drop=True)
    o3_8h_mean = df_8h["o3"].mean()
    co_8h_mean = df_8h["co"].mean()

    #PM2.5 and PM10 : NowCast (hour < 18) or 24-hour mean
    if hour_of_day < 18:

        # Fetch past 12 hours for NowCast
        window_start_12h = int(timestamp_unix) - (12 * 3600)
        filter_12h       = feature_group.feature("timestamp_unix") >= window_start_12h
        filter_12h      &= feature_group.feature("timestamp_unix") <  int(timestamp_unix)
        df_12h           = feature_group.filter(filter_12h).read()

        if len(df_12h) < 12:
            raise ValueError(
                f"Expected 12 past records for NowCast but only found {len(df_12h)}. "
                f"{12 - len(df_12h)} timestamp(s) missing in the feature group."
            )

        df_12h       = df_12h.sort_values("timestamp_unix").reset_index(drop=True)
        pm2_5_values = [pm2_5] + df_12h["pm2_5"].tolist()[::-1]
        pm10_values  = [pm10]  + df_12h["pm10"].tolist()[::-1]

        # NowCast for PM2.5
        c_min_25   = min(pm2_5_values)
        c_max_25   = max(pm2_5_values)
        weight_25  = max(c_min_25 / c_max_25, 0.5) if c_max_25 != 0 else 0.5
        pm2_5_conc = (
            sum((weight_25 ** i) * pm2_5_values[i] for i in range(len(pm2_5_values))) /
            sum((weight_25 ** i)                    for i in range(len(pm2_5_values)))
        )

        # NowCast for PM10
        c_min_10   = min(pm10_values)
        c_max_10   = max(pm10_values)
        weight_10  = max(c_min_10 / c_max_10, 0.5) if c_max_10 != 0 else 0.5
        pm10_conc  = (
            sum((weight_10 ** i) * pm10_values[i] for i in range(len(pm10_values))) /
            sum((weight_10 ** i)                   for i in range(len(pm10_values)))
        )
    else:
        # Fetch past 24 hours for mean
        window_start_24h = int(timestamp_unix) - (24 * 3600)
        filter_24h       = feature_group.feature("timestamp_unix") >= window_start_24h
        filter_24h      &= feature_group.feature("timestamp_unix") <  int(timestamp_unix)
        df_24h           = feature_group.filter(filter_24h).read()

        if len(df_24h) < 24:
            raise ValueError(
                f"Expected 24 past records for PM mean but only found {len(df_24h)}. "
                f"{24 - len(df_24h)} timestamp(s) missing in the feature group."
            )

        df_24h     = df_24h.sort_values("timestamp_unix").reset_index(drop=True)
        pm2_5_conc = df_24h["pm2_5"].mean()
        pm10_conc  = df_24h["pm10"].mean()

    #Calculate sub-AQI for each pollutant
    concentrations = {
        "pm2_5" : round(pm2_5_conc, 1),
        "pm10"  : round(pm10_conc),
        "o3_8h" : round(o3_8h_mean),
        "o3_1h" : round(o3),
        "no2"   : round(no2),
        "so2"   : round(so2),
        "co"    : round(co_8h_mean, 1),
    }

    sub_aqis = {}
    for pollutant, concentration in concentrations.items():
        sub_aqis[pollutant] = None
        for (c_low, c_high, aqi_low, aqi_high) in aqi_breakpoints[pollutant]:
            if c_low <= concentration <= c_high:
                sub_aqis[pollutant] = round(
                    ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
                )
                break

    #Final AQI = max of all valid sub-AQIs
    valid_sub_aqis = {k: v for k, v in sub_aqis.items() if v is not None}
    aqi_value = max(valid_sub_aqis.values())

    #Get AQI category
    aqi_category = "Out of Range"
    for (low, high), category in aqi_categories.items():
        if low <= aqi_value <= high:
            aqi_category = category
            break

    return  aqi_value, aqi_category

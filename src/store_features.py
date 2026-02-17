Import feature_engineering
Import fetch_current
import hopsworks
import pandas as pd

aqi_data = fetch_current.get_aqi() 
aqi = aqi_data["aqi"]
no2 = aqi_data["components"]["no2"]
so2 = aqi_data["components"]["so2"]
o3 = aqi_data["components"]["o3"]
co = aqi_data["components"]["co"]
pm2_5 = aqi_data["components"]["pm2_5"]
pm10 = aqi_data["components"]["pm10"]
timestamp_a= aqi_data["timestamp"]
PKT = timezone(timedelta(hours=5))  # Pakistan Standard Time is UTC+5
datetime_a = datetime.fromtimestamp(unix_timestamp, tz=PKT).strftime("%Y-%m-%d %H:%M:%S")


weather_data = fetch_current.get_weather()
temperature  = weather_data["temperature_2m"]
relative_humidity = weather_data["relative_humidity_2m"]
precipitation = weather_data["precipitation"]
surface_pressure = weather_data["surface_pressure"]
wind_speed = weather_data["wind_speed_10m"]
wind_direction  = weather_data["wind_direction_10m"]
shortwave_radiation = weather_data["shortwave_radiation"]
timestamp_w = weather_data["time"]
datetime_w = datetime.strptime(timestamp_w, "%Y-%m-%dT%H:%M").strftime("%Y-%m-%d %H:%M:%S")
timestamp_w = int(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

if timestamp_w != timestamp_a and datetime_w != datetime_a : 
  print ("Dates mismatchted")
else: 
  datetime= datetime_a
  datetime_unix = timestamp_a

wind_direction_sin, wind_direction_cos = feature_engineering.cyclic (wind-direction, 360)

hour, day, month = feature_engineering.extract_time_features (datetime_unix) 
hour_sin, hour_cos = feature_engineering.cyclic (hour, 24)
day_sin, day_cos = feature_engineering.cyclic (day, 7)
month_sin, month_cos = feature_engineering.cyclic (month, 12)

pollutants = {
  "o3" : o3,
  "no2": no2,
  "so2": so2,
  "co": co
}
converted_pollutants = feature_engineering (pollutants)
o3 = converted_pollutants["o3"]
so2 = converted_pollutants["so2"]
no2 = converted_pollutants["no2"]
co = converted_pollutants["co"]

aqi_value = feature_engineering.calculate_aqi(
    timestamp_unix : int | float,
    "pm2_5": pm2_5,
    "pm10" : pm10,
    "o3" : o3,
    "no2" : no2,
    "so2" : so2,
    "co" : co,
)

aqi_lag_1 = feature_engineering.lag_feature( aqi_value, datetime_unix, 1, "aqi_data", 2)
aqi_lag_2 = feature_engineering.lag_feature( aqi_value, datetime_unix, 2, "aqi_data", 2)
aqi_lag_3 = feature_engineering.lag_feature( aqi_value, datetime_unix, 3, "aqi_data", 2)
aqi_lag_6 = feature_engineering.lag_feature( aqi_value, datetime_unix, 6, "aqi_data", 2)
aqi_lag_12 = feature_engineering.lag_feature( aqi_value, datetime_unix, 12, "aqi_data", 2)
pm2_5_lag_1 = feature_engineering.lag_feature( pm2_5, datetime_unix, 1, "aqi_data", 2)
pm2_5_lag_3 = feature_engineering.lag_feature( pm2_5, datetime_unix, 3, "aqi_data", 2)
pm2_5_lag_6 = feature_engineering.lag_feature( pm2_5, datetime_unix, 6, "aqi_data", 2)
pm10_lag_1 = feature_engineering.lag_feature( pm10, datetime_unix, 1, "aqi_data", 2)
om10_lag_3 = feature_engineering.lag_feature( pm10, datetime_unix, 3, "aqi_data", 2)
pm10_lag_6 = feature_engineering.lag_feature( pm10, datetime_unix, 6, "aqi_data", 2)
co_lag_1 = feature_engineering.lag_feature( co, datetime_unix, 1, "aqi_data", 2)
co_lag_3 = feature_engineering.lag_feature( co, datetime_unix, 3, "aqi_data", 2)
co_lag_6 = feature_engineering.lag_feature( co, datetime_unix, 6, "aqi_data", 2)
no2_lag_1 = feature_engineering.lag_feature( no2, datetime_unix, 1, "aqi_data", 2)
no2_lag_1 = feature_engineering.lag_feature( no2, datetime_unix, 1, "aqi_data", 2)
no2_lag_1 = feature_engineering.lag_feature( no2, datetime_unix, 1, "aqi_data", 2)



# Connect to Hopsworks
project = fetch_current.connect()

fs = project.get_feature_store()

weather_lag = fs.get_feature_group(name = "weather_lag_data", version= 1)
aqi_lag = fs.get_feature_group(name = "aqi_lag_data", version= 1)
weather_rolling = fs.get_feature_group(name = "weather_rolling_data", version= 1)
aqi_rolling = fs.get_feature_group(name = "aqi_rolling_data", version= 1)
aqi_data = fs.get_feature_group(name = "aqi_data", version= 2)
weather_data =fs.get_feature_group(name = "weather_data", version= 1)
time_features = fs.get_feature_groupe(name = "time_features", version= 1)



# Create a dataframe with the row you want to insert
row_weather_lag = pd.DataFrame([{
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "temperature_lag_1": temperature_lag_1,
    "temperature_lag_3": temperature_lag_3,
    "temperature_lag_6": temperature_lag_6,
    "pressure_lag_1": pressure_lag_1,
    "pressure_lag_3": pressure_lag_3,
    "pressure_lag_6": pressure_lag_6,
    "humidity_lag_1": hummidity_lag_1,
    "humidity_lag_3": humidity_lag_3,
    "humidity_lag_6": humidity_lag_6,
    "precipitation_lag_1": precipitation_lag_1,
    "precipitation_lag_3": precipitation_lag_3,
    "precipitation_lag_6": precipitation_lag_6,  
}])
weather_lag.insert(row_weather_lag)

row_aqi_lag = pd.DataFrame([{
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "temperature_lag_1": temperature_lag_1,
    "temperature_lag_3": temperature_lag_3,
    "temperature_lag_6": temperature_lag_6,
    "pressure_lag_1": pressure_lag_1,
    "pressure_lag_3": pressure_lag_3,
    "pressure_lag_6": pressure_lag_6,
    "humidity_lag_1": hummidity_lag_1,
    "humidity_lag_3": humidity_lag_3,
    "humidity_lag_6": humidity_lag_6,
    "precipitation_lag_1": precipitation_lag_1,
    "precipitation_lag_3": precipitation_lag_3,
    "precipitation_lag_6": precipitation_lag_6,  
}])
aqi_lag.insert(row_aqi_lag)
























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
no2_lag_3 = feature_engineering.lag_feature( no2, datetime_unix, 1, "aqi_data", 2)
no2_lag_6 = feature_engineering.lag_feature( no2, datetime_unix, 1, "aqi_data", 2)



# Connect to Hopsworks
project = fetch_current.connect()

fs = project.get_feature_store()

weather_lag = fs.get_feature_group(name = "weather_lag_data", version= 1) # 01
aqi_lag = fs.get_feature_group(name = "aqi_lag_data", version= 1)  #02
weather_rolling = fs.get_feature_group(name = "weather_rolling_data", version= 1)  #03
aqi_rolling = fs.get_feature_group(name = "aqi_rolling_data", version= 1)  #04
aqi_data = fs.get_feature_group(name = "aqi_data", version= 2)    #5
weather_data =fs.get_feature_group(name = "weather_data", version= 1)    #06
time_features = fs.get_feature_groupe(name = "time_features", version= 1)  #07



# Create a dataframe with the row you want to insert
row_weather_lag = pd.DataFrame([{   #01
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

row_aqi_lag = pd.DataFrame([{    #02
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

row_weather_rolling = pd.DataFrame([{    #04
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "temperature_rolling_6": temperature_rolling_6,
    "temperature_rolling_12": temperature_rolling_12,
    "temperature_rolling_24": temperature_rolling_24,
    "pressure_rolling_6": pressure_rolling_6,
    "pressure_rolling_12": pressure_rolling_12,
    "pressure_rolling_24": pressure_rolling_24,
    "humidity_rolling_6": hummidity_rolling_6,
    "humidity_rolling_12": humidity_rolling_12,
    "humidity_rolling_24": humidity_rolling_24,
    "precipitation_rolling_6": precipitation_rolling_6,
    "precipitation_rolling_12": precipitation_rolling_12,
    "precipitation_rolling_24": precipitation_rolling_24,  
}])
weather_rolling.insert(row_weather_rolling)

row_aqi_rolling = pd.DataFrame([{    #03
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "aqi_rolling_6": aqi_rolling_6,
    "aqi_rolling_12": aqi_rolling_12,
    "aqiv_rolling_6": aqiv_rolling_6
    "aqiv_rolling_12": aqiv_rolling_12,
    "pm2_5_rolling_6": pm2_5_rolling_6,
    "pm2_5_rolling_12": pm2_5_rolling_12,
    "pm10_rolling_6": pm10_rolling_6,
    "pm10_rolling_12":pm10_rolling_12,
    "n02_rolling_6": no2_rolling_6,
    "no2_rolling_12": no2_rolling_12,
    "so2_rolling_6": so2_rolling_6,
    "so2_rolling_12": so2_rolling_12,
    "co_rolling_6": co_rolling_6,  
    "co_rolling_12": co_rolling_12,
    "o3_rolling_6": o3_rolling_6,
    "o3_rolling_12": o3_rolling_12,
  
}])
aqi_rolling.insert(row_aqi_rolling)

row_weather_data = pd.DataFrame([{    #06
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "temperature": temperature_lag_1,
    "relative_humidity": temperature_lag_3,
    "surface_pressure": temperature_lag_6,
    "wind_speed": pressure_lag_1,
    "shortwave_radiation": pressure_lag_3,
    "precipitation": pressure_lag_6,
    "wind_direction_cos": hummidity_lag_1,
    "wind_direction_sin": humidity_lag_3,  
}])
weather_data.insert(row_weather_data)

row_aqi_data = pd.DataFrame([{   #05
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "aqi": aqi,
    "aqi_value": aqi_value,
    "no2": no2,
    "so2": so2,
    "co": co,
    "o3": o3,
    "pm2_5": pm2_5,
    "pm10": pm10,  
}])
aqi_data.insert(row_aqi_data)

row_time_features = pd.DataFrame([{    #07
    "datetime_unix" : datetime_unix,
    "datetime": datetime,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "day_sin": day_sin,
    "day_cos": day_cos,
    "month_sin": month_sin,
    "month_cos": month_cos, 
}])
time_features.insert(row_time_features)





















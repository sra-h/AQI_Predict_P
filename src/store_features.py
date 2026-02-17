Import feature_engineering
Import fetch_current

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
humidity     = weather_data["relative_humidity_2m"]
precipitation = weather_data["precipitation"]
pressure     = weather_data["surface_pressure"]
wind_speed   = weather_data["wind_speed_10m"]
wind_dir     = weather_data["wind_direction_10m"]
radiation    = weather_data["shortwave_radiation"]
timestamp_w = weather_data["time"]
datetime_w = datetime.strptime(timestamp_w, "%Y-%m-%dT%H:%M").strftime("%Y-%m-%d %H:%M:%S")
timestamp_w = int(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

if timestamp_w != timestamp_a and datetime_w != datetime_a : 
  print ("Dates mismatchted")





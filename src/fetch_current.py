import requests

def get_aqi() -> dict:
   
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": 33.0849, "lon": 72.6890, "appid": os.getenv("AQI_API_KEY")}

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    pollution = data["list"][0]

    return {
        "aqi": pollution["main"]["aqi"],
        "components": pollution["components"]  # CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3
    }

def get_weather() -> dict:
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 33.0849,
        "longitude": 72.6890,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation"
    ]}

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    return data["current"]

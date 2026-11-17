import os
import requests

BASE = "https://api.openweathermap.org"

def get_coordinates(city, api_key=None):
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENWEATHER_API_KEY in your environment or Streamlit secrets.")
    response = requests.get(
        f"{BASE}/geo/1.0/direct",
        params={"q": city, "limit": 1, "appid": api_key},
        timeout=20
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"No matching location found for: {city}")
    return results[0]

def get_current_weather(city, api_key=None):
    api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENWEATHER_API_KEY in your environment or Streamlit secrets.")
    place = get_coordinates(city, api_key)
    response = requests.get(
        f"{BASE}/data/2.5/weather",
        params={
            "lat": place["lat"], "lon": place["lon"],
            "appid": api_key, "units": "metric"
        },
        timeout=20
    )
    response.raise_for_status()
    data = response.json()
    rain = data.get("rain", {}).get("1h", 0.0)
    snow = data.get("snow", {}).get("1h", 0.0)
    weather_main = data.get("weather", [{}])[0].get("main", "Unknown")
    return {
        "city": f'{place["name"]}, {place.get("country", "")}',
        "latitude": place["lat"],
        "longitude": place["lon"],
        "temperature_C": data.get("main", {}).get("temp"),
        "feels_like_C": data.get("main", {}).get("feels_like"),
        "humidity_percent": data.get("main", {}).get("humidity"),
        "wind_speed_kmh": round(data.get("wind", {}).get("speed", 0) * 3.6, 1),
        "precipitation_mm": round(rain + snow, 2),
        "weather_condition": weather_main,
        "description": data.get("weather", [{}])[0].get("description", ""),
        "observed_at_utc": data.get("dt")
    }

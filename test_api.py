import requests
import pandas as pd

# Wetterdaten für Zürich abrufen
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 47.4,
    "longitude": 8.5,
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover"],
    "past_days": 7
}

response = requests.get(url, params=params)
data = response.json()

# In DataFrame umwandeln
df = pd.DataFrame({
    "time": data["hourly"]["time"],
    "temperature": data["hourly"]["temperature_2m"],
    "humidity": data["hourly"]["relative_humidity_2m"],
    "precipitation": data["hourly"]["precipitation"],
    "cloud_cover": data["hourly"]["cloud_cover"]
})

print(df.head(10))
print(f"\nDatensatz hat {len(df)} Zeilen")

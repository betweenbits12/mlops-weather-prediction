# Feature Pipeline
import requests
import pandas as pd
import numpy as np

def fetch_weather_data():
    """Holt Wetterdaten von Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 47.4,
        "longitude": 8.5,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover"],
        "past_days": 14  # Mehr Daten für bessere Features
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "cloud_cover": data["hourly"]["cloud_cover"]
    })

    return df

def engineer_features(df):
    """Berechnet aggregierte und RT-Features."""
    df = df.sort_values("time").reset_index(drop=True)

    # Aggregiertes Feature: Durchschnittliche Luftfeuchtigkeit der letzten 24h
    df["humidity_24h_avg"] = df["humidity"].rolling(window=24, min_periods=1).mean()

    # Label: Hat es in den NÄCHSTEN 2 Stunden geregnet? (Verschiebung nach oben)
    df["rain_next_2h"] = (df["precipitation"].shift(-1) > 0) | (df["precipitation"].shift(-2) > 0)
    df["rain_next_2h"] = df["rain_next_2h"].astype(int)

    return df

if __name__ == "__main__":
    print("Hole Wetterdaten...")
    df = fetch_weather_data()
    print(f"{len(df)} Zeilen geladen")

    print("Berechne Features...")
    df_features = engineer_features(df)

    print("\nErste 5 Zeilen mit Features:")
    print(df_features[["time", "humidity", "humidity_24h_avg", "cloud_cover", "rain_next_2h"]].head())

    print("\nFeature-Statistik:")
    print(f"Regen in den nächsten 2h: {df_features['rain_next_2h'].sum()} von {len(df_features)} Stunden")

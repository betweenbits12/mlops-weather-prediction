"""
Hinweis zur Feature Pipeline:
Hopsworks Featurestore konnte nicht verwendet werden, da das Paket
'hopsworks' mit Python 3.14 (Codespace-Standard) inkompatibel ist.
Der Build bricht ab mit: ModuleNotFoundError: No module named 'imp'.
Die geforderten Features (aggregiert + RT) werden daher lokal als
Parquet/CSV gespeichert. Die Architektur bleibt identisch.
"""

import requests
import pandas as pd
from pathlib import Path


def fetch_weather_data():
    """Holt Wetterdaten von Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 47.4,
        "longitude": 8.5,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover"],
        "past_days": 14
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

    # RT-Feature: Aktueller Bewoelkungsgrad
    df["cloud_cover_rt"] = df["cloud_cover"]

    # Label: Hat es in den NAECHSTEN 2 Stunden geregnet?
    df["rain_next_2h"] = (df["precipitation"].shift(-1) > 0) | (df["precipitation"].shift(-2) > 0)
    df["rain_next_2h"] = df["rain_next_2h"].astype(int)

    return df


def save_features_locally(df, output_dir="data"):
    """Speichert Features als Parquet (lokaler Ersatz fuer Featurestore)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = f"{output_dir}/weather_features.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Features gespeichert unter: {output_path}")

    csv_path = f"{output_dir}/weather_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Features auch als CSV: {csv_path}")


if __name__ == "__main__":
    print("=== Feature Pipeline ===")
    print("Hinweis: Hopsworks ist mit Python 3.14 inkompatibel.")
    print("Features werden lokal als Parquet gespeichert.")
    print()

    print("1. Wetterdaten abrufen...")
    df_raw = fetch_weather_data()
    print(f"   {len(df_raw)} Zeilen geladen")

    print("2. Features berechnen...")
    df_features = engineer_features(df_raw)
    print("   - Aggregiertes Feature: humidity_24h_avg")
    print("   - RT-Feature: cloud_cover_rt")
    print("   - Label: rain_next_2h")

    print("3. Features speichern...")
    save_features_locally(df_features)

    print("\nFeature-Statistik:")
    print(f"   Regen in den naechsten 2h: {df_features['rain_next_2h'].sum()} von {len(df_features)} Stunden")
    print(f"   Daten von {df_features['time'].min()} bis {df_features['time'].max()}")

import pandas as pd
import requests
import joblib

def load_model(path="models/rain_predictor.pkl"):
    return joblib.load(path)

def get_data():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": 47.4, "longitude": 8.5, "hourly": ["relative_humidity_2m", "cloud_cover"], "past_days": 2}
    data = requests.get(url, params=params).json()
    return pd.DataFrame({"humidity": data["hourly"]["relative_humidity_2m"], "cloud_cover": data["hourly"]["cloud_cover"]})

if __name__ == "__main__":
    print("=== Inference ===")
    clf = load_model()
    df = get_data()
    df["humidity_24h_avg"] = df["humidity"].rolling(window=24, min_periods=1).mean()
    df["cloud_cover_rt"] = df["cloud_cover"]  # Umbenennen!
    latest = df.iloc[-1:]
    X = latest[["humidity_24h_avg", "cloud_cover_rt"]]
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    print(f"Regen in 2h: {'JA' if pred else 'NEIN'}")
    print(f"Wahrscheinlichkeiten: Kein Regen {prob[0]:.1%}, Regen {prob[1]:.1%}")
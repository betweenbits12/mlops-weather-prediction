"""
Training Pipeline:
Laedt lokale Features, trainiert Modell und speichert es.
Hopsworks ist mit Python 3.14 inkompatibel, daher lokale Speicherung.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path


def load_features(input_path="data/weather_features.parquet"):
    """Laedt Features aus lokaler Datei (Parquet)."""
    df = pd.read_parquet(input_path)
    print(f"Features geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
    return df


def prepare_data(df):
    """Bereitet Features (X) und Label (y) vor."""
    df = df.dropna()
    feature_cols = ["humidity_24h_avg", "cloud_cover_rt"]
    X = df[feature_cols]
    y = df["rain_next_2h"]

    print(f"Trainingsset: {len(X)} Zeilen")
    print(f"Features: {feature_cols}")
    print(f"Regen-Ja: {y.sum()}, Regen-Nein: {len(y) - y.sum()}")

    return X, y


def train_model(X, y):
    """Trainiert einen RandomForestClassifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return clf, metrics


def save_model(clf, output_dir="models"):
    """Speichert Modell lokal (joblib)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{output_dir}/rain_predictor.pkl"
    joblib.dump(clf, model_path)
    print(f"Modell gespeichert: {model_path}")
    return model_path


if __name__ == "__main__":
    print("=== Training Pipeline ===")
    print("Laedt lokale Features (Parquet), trainiert Modell, speichert joblib.")
    print()

    print("1. Features laden...")
    df = load_features()

    print("1. Features laden...")
    df = load_features()

    print("\n2. Daten vorbereiten...")
    X, y = prepare_data(df)

    print("\n3. Modell trainieren...")
    clf, metrics = train_model(X, y)

    print("\n4. Metriken (Testset):")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    print("\n5. Modell speichern...")
    model_path = save_model(clf)

    print("\nTraining Pipeline abgeschlossen!")

# Wettervorhersage MLOps Pipeline

## Projektübersicht
Dieses Repository enthält eine MLOps-Pipeline zur Vorhersage von Niederschlag
in den nächsten 2 Stunden. Die Pipeline wurde im Rahmen der Projektarbeit
zum Thema FTI-Architektur und Featurestore erstellt.

## Datenquelle
- **Open-Meteo Weather API** (https://open-meteo.com/)
- Kostenlos, keine API-Key nötig
- Koordinaten: Zürich (47.4°N, 8.5°E)

## Features

| Feature | Typ | Beschreibung |
|---------|-----|--------------|
| `humidity_24h_avg` | Aggregiert (Batch) | Durchschnittliche Luftfeuchtigkeit der letzten 24 Stunden |
| `cloud_cover_rt` | Real-Time (RT) | Aktueller Bewölkungsgrad in % |

**Label:** `rain_next_2h` (Hat es in den nächsten 2 Stunden geregnet? Ja/Nein)

## Bekannte Limitation

- **Python-Version:** Entwickelt und getestet mit Python 3.14.2 (GitHub Codespaces Standard).
  Das Paket `hopsworks` ist aktuell nur bis Python 3.12/3.13 kompatibel
  (verwendet veraltetes `imp`-Modul).
- **Lösung:** Features und Modell werden lokal als Parquet/CSV und joblib
  gespeichert, was gemäss Aufgabenstellung zulässig ist.

## Umsetzung ohne Hopsworks (Lokale Speicherung)

Da Hopsworks aufgrund der Python-Version nicht nutzbar ist, wurden die
Hopsworks-Konzepte wie folgt lokal abgebildet:

| Hopsworks-Konzept | Lokale Alternative | Datei |
|-------------------|--------------------|-------|
| Feature Group | Parquet/CSV-Datei | `data/weather_features.parquet` |
| Feature View | Direktes Laden + Spalten-Mapping (`cloud_cover` → `cloud_cover_rt`) | Code in `training_pipeline.py` / `inference_pipeline.py` |
| Model Registry | `joblib`-Datei | `models/rain_predictor.pkl` |

Die FTI-Architektur (Feature, Training, Inference) ist dadurch vollständig
erhalten, lediglich die Speicherung erfolgt im lokalen Dateisystem statt in
einer verwalteten Featurestore-/Model-Registry-Cloud-Lösung.

## Sicherheitshinweis

Die Datei `.env` enthält einen Hopsworks API-Key und Minio-Zugangsdaten.
Sie wurde der Vollständigkeit halber im Repository belassen, damit die
Konfiguration nachvollziehbar ist. Ausserhalb einer Projektarbeit würden
solche Credentials niemals im Klartext in einem Repository liegen, sondern
über Secrets Management (z. B. GitHub Secrets, Vault, Environment Variables)
bereitgestellt werden..

## Architektur

### 1. Feature Pipeline (`feature_pipeline.py`)
- Holt historische Wetterdaten von Open-Meteo API (14 Tage)
- Berechnet aggregierte Features (24h Rolling Average)
- Berechnet RT-Features (aktuelle Bewölkung)
- Speichert Features lokal als Parquet und CSV

### 2. Training Pipeline (`training_pipeline.py`)
- Lädt Features aus lokaler Parquet-Datei
- Trainiert einen `RandomForestClassifier`
- Evaluiert mit Accuracy, Precision, Recall, F1
- Speichert Modell als `joblib`-Datei

### 3. Inference Pipeline (`inference_pipeline.py`)
- Lädt trainiertes Modell
- Holt aktuelle Wetterdaten von der API
- Berechnet Features in Echtzeit
- Sagt: Regen in 2h? (Ja/Nein) mit Wahrscheinlichkeiten

## Installation

```bash
pip install pandas requests scikit-learn pyarrow joblib

Ausführung
Die drei Pipelines müssen **nacheinander** ausgeführt werden:
# 1. Features erstellen
# Output: data/weather_features.parquet + .csv
python feature_pipeline.py

# 2. Modell trainieren
# Output: models/rain_predictor.pkl
python training_pipeline.py

# 3. Vorhersage
# Output: Vorhersage Ja/Nein mit Wahrscheinlichkeit
python inference_pipeline.py

Alternativ in einem Durchlauf:
python feature_pipeline.py && python training_pipeline.py && python inference_pipeline.py


Maya Richter: Erstellt im Rahmen der CAS MLOps Projektarbeit.
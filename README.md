# Wettervorhersage MLOps Pipeline

## Projektübersicht
Dieses Repository enthält eine MLOps-Pipeline zur Vorhersage von Niederschlag in den nächsten 2 Stunden. Die Pipeline wurde im Rahmen der Projektarbeit zum Thema FTI-Architektur und Featurestore erstellt.

## Datenquelle
- **Open-Meteo Weather API** (https://open-meteo.com/)
- Kostenlos, keine API-Key nötig
- Koordinaten: Zürich (47.4°N, 8.5°E)

## Features

| Feature | Typ | Beschreibung |
|---------|-----|--------------|
| `humidity_24h_avg` | Aggregiert (Batch) | Durchschnittliche Luftfeuchtigkeit der letzten 24 Stunden |
| `cloud_cover_rt` | Real-Time (RT) | Aktueller Bewölkungsgrad in % |

## Label:

- `rain_next_2h` (Hat es in den nächsten 2 Stunden geregnet? Ja/Nein)
*Anwendungsfall:* "Muss ich einen Schirm einpacken?"
Das Modell gibt die Wahrscheinlichkeit für Regen in 2h aus. Bei >50% Wahrscheinlichkeit empfiehlt sich ein Schirm. 
Die Feature-Auswahl folgt meteorologischen Überlegungen: Die Luftfeuchtigkeit der letzten 24 Stunden (`humidity_24h_avg`) ist ein Indikator für die Feuchtigkeitsaufsättigung der Atmosphäre, während der aktuelle Bewölkungsgrad (`cloud_cover_rt`) ein direkter, echtzeitbasierter Hinweis auf potenzielle Niederschläge ist. Das Label `rain_next_2h` wird aus den historischen Niederschlagsdaten der API abgeleitet (Schwellenwert: > 0 mm).

## Modell
Als Modell wird ein `RandomForestClassifier` verwendet. Dieser eignet sich besonders für tabellarische Wetterdaten, da er nichtlineare Zusammenhänge modellieren kann, robust gegenüber Ausreissern ist und mit wenig Hyperparameter-Tuning gute Ergebnisse liefert. 
Die Evaluation erfolgt über Accuracy, Precision, Recall und F1-Score, um sowohl die Gesamtperformance als auch die Balance zwischen false positives und false negatives zu bewerten.

## Bekannte Limitation

- **Python-Version:** Entwickelt und getestet mit Python 3.14.2 (GitHub Codespaces Standard).
  Das Paket `hopsworks` ist aktuell nur bis Python 3.12/3.13 kompatibel
  (verwendet veraltetes `imp`-Modul).
- **Lösung:** Features und Modell werden lokal als Parquet/CSV und joblib
  gespeichert, was gemäss Aufgabenstellung zulässig ist.

## Umsetzung ohne Hopsworks (Lokale Speicherung)

Da Hopsworks aufgrund der Python-Version nicht nutzbar ist, wurden die Hopsworks-Konzepte wie folgt lokal abgebildet:

| Hopsworks-Konzept | Lokale Alternative | Datei |
|-------------------|--------------------|-------|
| Feature Group | Parquet/CSV-Datei | `data/weather_features.parquet` |
| Feature View | Direktes Laden + Spalten-Mapping (`cloud_cover` → `cloud_cover_rt`) | Code in `training_pipeline.py` / `inference_pipeline.py` |
| Model Registry | `joblib`-Datei | `models/rain_predictor.pkl` |

Die FTI-Architektur (Feature, Training, Inference) ist dadurch vollständig erhalten, lediglich die Speicherung erfolgt im lokalen Dateisystem statt in einer verwalteten Featurestore-/Model-Registry-Cloud-Lösung.

## Sicherheitshinweis

Die Datei `.env` enthält einen Hopsworks API-Key und Minio-Zugangsdaten.
Sie wurde der Vollständigkeit halber im Repository belassen, damit die Konfiguration nachvollziehbar ist. Ausserhalb einer Projektarbeit würden solche Credentials niemals im Klartext in einem Repository liegen, sondern über Secrets Management (z. B. GitHub Secrets, Vault, Environment Variables) bereitgestellt werden..

## Architektur

### 1. Feature Pipeline (`feature_pipeline.py`)
Die Pipeline aggregiert stündliche API-Daten über 14 Tage, berechnet das 24h-Rolling-Window für die Luftfeuchtigkeit und verknüpft diese mit dem aktuellen Echtzeit-Wert zur vollständigen Feature-Matrix.
- Holt historische Wetterdaten von Open-Meteo API (14 Tage)
- Berechnet aggregierte Features (24h Rolling Average)
- Berechnet RT-Features (aktuelle Bewölkung)
- Speichert Features lokal als Parquet und CSV

### 2. Training Pipeline (`training_pipeline.py`)
Nach dem Laden der Parquet-Datei wird ein stratifizierter Train-Test-Split durchgeführt (80/20), um sicherzustellen, dass sowohl Regen- als auch Nicht-Regen-Tage in beiden Sets vertreten sind.
- Lädt Features aus lokaler Parquet-Datei
- Trainiert einen `RandomForestClassifier`
- Evaluiert mit Accuracy, Precision, Recall, F1
- Speichert Modell als `joblib`-Datei

### 3. Inference Pipeline (`inference_pipeline.py`)
Die Echtzeit-Inferenz holt die aktuellsten stündlichen Messungen, berechnet daraus die erforderlichen Features in der identischen Struktur wie im Training und gibt die binäre Vorhersage inklusive Wahrscheinlichkeit aus.
- Lädt trainiertes Modell
- Holt aktuelle Wetterdaten von der API
- Berechnet Features in Echtzeit
- Sagt: Regen in 2h? (Ja/Nein) mit Wahrscheinlichkeiten

### Weitere Dateien
- `test_api.py`
- Hilfsskript zur Prüfung der Open-Meteo API-Verbindung (nicht Teil der Pipeline).

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

## Reflexion und Limitationen

Die Pipelines sind voll funktionsfähig und wurden erfolgreich getestet (`feature → training → inference` in einem Durchlauf).

Die einzige bewusste Abweichung vom empfohlenen Hopsworks-Workflow ist die lokale Speicherung: Aufgrund der Python-3.14-Inkompatibilität von `hopsworks` (verwendet entferntes `imp`-Modul) war keine Verbindung zum Featurestore möglich. Stattdessen wurden die Hopsworks-Konzepte (Feature Group, Feature View, Model Registry) durch lokale Parquet/CSV- und joblib-Dateien bgebildet. Die FTI-Architektur bleibt dadurch logisch erhalten, lediglich die Persistenzschicht ist vereinfacht.

Für den produktiven Einsatz wäre ein Wechsel zu Python 3.12/3.13 und die Integration von Hopsworks als nächster Schritt empfohlen.

Maya Richter: Erstellt im Rahmen der CAS MLOps Projektarbeit.
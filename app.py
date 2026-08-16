from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import os
import io

app = Flask(__name__)
CORS(app)

# ── Modell von Google Drive laden ─────────────────────────────────────────────
def laden_von_gdrive(file_id: str):
    import pickle
    from bs4 import BeautifulSoup
    
    session = requests.Session()
    
    # Erste Anfrage um den Bestätigungs-Token zu holen
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = session.get(url)
    
    # Token aus HTML extrahieren
    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form")
    
    if form:
        # Direkte Download-URL mit Token verwenden
        download_url = "https://drive.usercontent.google.com/download"
        params = {"id": file_id, "export": "download", "confirm": "t"}
        r = session.get(download_url, params=params, stream=True)
    else:
        r = session.get(url, stream=True)
    
    buffer = io.BytesIO()
    for chunk in r.iter_content(chunk_size=32768):
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)
    return pickle.load(buffer)

print("Lade Modell von Google Drive ...")
MODELL_ID     = "14cNkND8oRmdh2NWQ7THryMarh37PD95h"
STOP_MEANS_ID = "1vDvL-aYXBRaEyaRf6UJLNblxnSX0db03"

modell     = laden_von_gdrive(MODELL_ID)
stop_means = laden_von_gdrive(STOP_MEANS_ID)
global_mean = float(np.mean(list(stop_means.values())))
print("Modell erfolgreich geladen!")

# ── Wetter von Open-Meteo abrufen ─────────────────────────────────────────────
def wetter_abrufen(datum: str, stunde: int) -> dict:
    try:
        url = (
            f"https://forecast.open-meteo.com/v1/forecast"
            f"?latitude=51.9607&longitude=7.6261"
            f"&hourly=temperature_2m,apparent_temperature,"
            f"relative_humidity_2m,rain,surface_pressure,wind_speed_10m"
            f"&timezone=Europe/Berlin"
            f"&start_date={datum}&end_date={datum}"
        )
        r = requests.get(url, timeout=10)
        data = r.json()
        idx = data["hourly"]["time"].index(f"{datum}T{stunde:02d}:00")
        return {
            "temperature_2m (°C)":        data["hourly"]["temperature_2m"][idx],
            "apparent_temperature (°C)":  data["hourly"]["apparent_temperature"][idx],
            "relative_humidity_2m (%)":   data["hourly"]["relative_humidity_2m"][idx],
            "rain (mm)":                  data["hourly"]["rain"][idx],
            "surface_pressure (hPa)":     data["hourly"]["surface_pressure"][idx],
            "wind_speed_10m (km/h)":      data["hourly"]["wind_speed_10m"][idx],
        }
    except:
        return {
            "temperature_2m (°C)":        20.0,
            "apparent_temperature (°C)":  19.0,
            "relative_humidity_2m (%)":   70.0,
            "rain (mm)":                  0.0,
            "surface_pressure (hPa)":     1013.0,
            "wind_speed_10m (km/h)":      10.0,
        }

# ── Schulferien NRW ───────────────────────────────────────────────────────────
FERIEN_NRW_2026 = [
    ("2026-07-20", "2026-09-01"),
    ("2026-10-05", "2026-10-17"),
]

def ist_schulferien(datum_str: str) -> int:
    from datetime import date
    datum = date.fromisoformat(datum_str)
    for start, ende in FERIEN_NRW_2026:
        if date.fromisoformat(start) <= datum <= date.fromisoformat(ende):
            return 1
    return 0

# ── Vorhersage-Endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data       = request.json
    linie      = str(data.get("linie", "2"))
    datum      = data.get("datum")
    stunde     = int(data.get("stunde", 8))
    minute     = int(data.get("minute", 0))
    stop_id    = data.get("stop_id", None)

    dt        = datetime.strptime(datum, "%Y-%m-%d")
    wochentag = dt.weekday()

    wetter      = wetter_abrufen(datum, stunde)
    hour_sin    = np.sin(2 * np.pi * stunde / 24.0)
    hour_cos    = np.cos(2 * np.pi * stunde / 24.0)
    is_rush     = int((7 <= stunde <= 9 or 16 <= stunde <= 18) and wochentag < 5)
    rain_rush   = wetter["rain (mm)"] * is_rush
    schulferien = ist_schulferien(datum)
    stop_encoded = stop_means.get(str(stop_id), global_mean) if stop_id else global_mean

    eingabe = {
        "Direction_ID":               1,
        "hour_sin":                   hour_sin,
        "hour_cos":                   hour_cos,
        "is_rush_hour":               is_rush,
        "rain_during_rush":           rain_rush,
        "temperature_2m (°C)":        wetter["temperature_2m (°C)"],
        "apparent_temperature (°C)":  wetter["apparent_temperature (°C)"],
        "relative_humidity_2m (%)":   wetter["relative_humidity_2m (%)"],
        "rain (mm)":                  wetter["rain (mm)"],
        "surface_pressure (hPa)":     wetter["surface_pressure (hPa)"],
        "wind_speed_10m (km/h)":      wetter["wind_speed_10m (km/h)"],
        "schulferien":                schulferien,
        "abweichung_aktiv":           0,
        "Current_Stop_Encoded":       stop_encoded,
    }

    FEATURE_COLS = modell.feature_names_in_
    for col in FEATURE_COLS:
        if col.startswith("Line_"):
            eingabe[col] = 1 if col == f"Line_{linie}" else 0
        elif col.startswith("day_of_week_"):
            tag = int(col.split("_")[-1])
            eingabe[col] = 1 if wochentag == tag else 0

    df_eingabe = pd.DataFrame([eingabe])
    for col in FEATURE_COLS:
        if col not in df_eingabe.columns:
            df_eingabe[col] = 0
    df_eingabe = df_eingabe[FEATURE_COLS]

    delay = float(modell.predict(df_eingabe)[0])

    return jsonify({
        "linie":       linie,
        "datum":       datum,
        "uhrzeit":     f"{stunde:02d}:{minute:02d}",
        "delay_min":   round(delay, 1),
        "wetter":      wetter,
        "schulferien": bool(schulferien),
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Bus Delay Predictor API"})

if __name__ == "__main__":
    app.run(debug=False)

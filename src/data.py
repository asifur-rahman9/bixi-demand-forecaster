"""Loading the dataset and getting weather from Open-Meteo."""

import pandas as pd
import requests

from src import DATA_DIR, TZ

# Downtown Montreal (one weather point is enough for the city)
LAT, LON = 45.5019, -73.5674
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def get_json(url, params=None, timeout=10):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_open_meteo(payload):
    hourly = payload["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
    })
    # When daylight saving time ends the same hour shows up twice, keep the first one
    return df.drop_duplicates("ts").reset_index(drop=True)


def fetch_weather_archive(start, end):
    params = {"latitude": LAT, "longitude": LON, "start_date": start, "end_date": end,
              "hourly": "temperature_2m,precipitation", "timezone": TZ}
    return parse_open_meteo(get_json(ARCHIVE_URL, params, timeout=30))


def fetch_weather_forecast(hours=48):
    params = {"latitude": LAT, "longitude": LON, "hourly": "temperature_2m,precipitation",
              "timezone": TZ, "forecast_hours": hours}
    return parse_open_meteo(get_json(FORECAST_URL, params))


def load_demand(path=DATA_DIR / "hourly_demand.csv.gz"):
    return pd.read_csv(path, parse_dates=["ts"], dtype={"station_id": str})


def load_stations(path=DATA_DIR / "stations.csv"):
    return pd.read_csv(path, dtype={"station_id": str})

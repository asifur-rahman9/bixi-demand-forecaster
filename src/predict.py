"""Loads the trained models and forecasts departures and arrivals for a station."""

import json

import joblib
import numpy as np

from src import MODEL_DIR
from src.features import TARGETS, build_features


class Forecaster:
    def __init__(self, models, metadata):
        self.models = models
        self.metadata = metadata
        self.stations = list(metadata["stations"])

    @classmethod
    def load(cls, model_dir=MODEL_DIR):
        models = joblib.load(model_dir / "model.joblib")
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(models, metadata)

    def predict(self, station_id, weather):
        """Hourly forecast for one station. weather has columns ts, temperature, precipitation."""
        station_id = str(station_id)
        if station_id not in self.stations:
            raise ValueError(f"Station {station_id} is not one of the trained stations")
        rows = weather[["ts", "temperature", "precipitation"]].assign(station_id=station_id)
        X = build_features(rows, self.stations)
        out = rows[["ts"]].reset_index(drop=True)
        for target in TARGETS:
            pred = np.clip(self.models[target].predict(X), 0, None)  # no negative trips
            out[target] = np.rint(pred).astype(int)
        return out

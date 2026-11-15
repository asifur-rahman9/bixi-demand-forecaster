import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def demand():
    """Small fake dataset with an evening rush hour, so the model has something to learn."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2025-06-01", periods=24 * 42, freq="h")
    frames = []
    for i, sid in enumerate(["10", "20", "30"]):
        peak = np.exp(-((hours.hour.to_numpy() - 17) ** 2) / 8) * (6 + 2 * i)
        frames.append(pd.DataFrame({
            "station_id": sid,
            "ts": hours,
            "departures": rng.poisson(1 + peak),
            "arrivals": rng.poisson(1 + peak[::-1]),
            "temperature": 20 + 5 * np.sin(np.arange(len(hours)) / 24),
            "precipitation": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)

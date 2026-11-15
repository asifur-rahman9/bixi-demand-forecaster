import json
from pathlib import Path

from src.data import parse_open_meteo
from src.live import classify, combine, parse_station_information, parse_station_status

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text())


def test_parse_and_classify_gbfs():
    info = parse_station_information(load_fixture("station_information.json"))
    status = parse_station_status(load_fixture("station_status.json"))
    live = combine(info, status)
    assert {"name", "lat", "lon", "bikes", "docks", "state"} <= set(live.columns)
    assert set(live["state"]) <= {"ok", "nearly empty", "nearly full", "offline"}
    # the placeholder station at (0, 0) is dropped
    assert info["lat"].between(45.3, 45.8).all()


def test_classify():
    assert classify(bikes=1, docks=10) == "nearly empty"
    assert classify(bikes=10, docks=2) == "nearly full"
    assert classify(bikes=10, docks=10) == "ok"
    assert classify(bikes=10, docks=10, in_service=False) == "offline"


def test_open_meteo_drops_repeated_dst_hour():
    payload = {"hourly": {"time": ["2025-11-02T00:00", "2025-11-02T01:00", "2025-11-02T01:00"],
                          "temperature_2m": [5, 4, 4], "precipitation": [0, 0, 0]}}
    df = parse_open_meteo(payload)
    assert len(df) == 2

"""Live station availability from BIXI's public GBFS feed."""

import pandas as pd

from src import TZ
from src.data import get_json

GBFS = "https://gbfs.velobixi.com/gbfs/en"
LOW = 2  # 2 or fewer bikes (or docks) counts as "nearly empty" (or "nearly full")


def parse_station_information(payload):
    df = pd.DataFrame(payload["data"]["stations"])[["station_id", "name", "lat", "lon", "capacity"]]
    df["station_id"] = df["station_id"].astype(str)
    # The feed has a few placeholder stations at (0, 0), keep only ones in the Montreal area
    in_area = df["lat"].between(45.3, 45.8) & df["lon"].between(-74.1, -73.3)
    return df[in_area].reset_index(drop=True)


def parse_station_status(payload):
    df = pd.DataFrame(payload["data"]["stations"])
    df = df[["station_id", "num_bikes_available", "num_ebikes_available", "num_docks_available",
             "is_renting", "is_returning", "last_reported"]]
    df = df.rename(columns={"num_bikes_available": "bikes", "num_ebikes_available": "ebikes",
                            "num_docks_available": "docks"})
    df["station_id"] = df["station_id"].astype(str)
    df["last_reported"] = pd.to_datetime(df["last_reported"], unit="s", utc=True).dt.tz_convert(TZ)
    return df


def classify(bikes, docks, in_service=True):
    if not in_service:
        return "offline"
    if bikes <= LOW:
        return "nearly empty"
    if docks <= LOW:
        return "nearly full"
    return "ok"


def combine(info, status):
    df = info.merge(status, on="station_id")
    in_service = (df["is_renting"] == 1) & (df["is_returning"] == 1)
    df["state"] = [classify(b, d, s) for b, d, s in zip(df["bikes"], df["docks"], in_service)]
    return df


def fetch_live():
    info = parse_station_information(get_json(f"{GBFS}/station_information.json"))
    status = parse_station_status(get_json(f"{GBFS}/station_status.json"))
    return combine(info, status)

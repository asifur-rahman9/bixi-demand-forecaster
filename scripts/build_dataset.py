"""Build data/hourly_demand.csv.gz and data/stations.csv from BIXI open data.

Usage:
    python scripts/build_dataset.py path/to/DonneesOuvertes2025_....zip

Steps:
1. Stream the yearly trips CSV out of the zip (station + timestamp columns only).
2. Count departures and arrivals per station per local hour.
3. Keep the N busiest stations and match them to live GBFS station IDs
   (exact name match first, then nearest coordinates within 75 m).
4. Zero-fill missing hours for the season, then join hourly Open-Meteo weather.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "src" can be imported when running this file directly

from src import data as bdata  # noqa: E402
from src import live  # noqa: E402

TZ = "America/Montreal"
SIDES = {
    side: (f"{prefix}STATIONNAME", f"{prefix}STATIONLATITUDE", f"{prefix}STATIONLONGITUDE",
           f"{prefix}TIMEMS")
    for side, prefix in [("departures", "START"), ("arrivals", "END")]
}


def _norm(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(name.lower().replace("-", " ").split())


def _local_hour(ms: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ms.astype("int64"), unit="ms", utc=True)
        .dt.tz_convert(TZ)
        .dt.tz_localize(None)  # remove the timezone first, otherwise floor() fails on the DST hour
        .dt.floor("h")
    )


def count_trips(zip_path: Path, chunksize: int = 2_000_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hourly counts per station name with departures/arrivals, station coords)."""
    counts: dict[str, list[pd.Series]] = {k: [] for k in SIDES}
    coords: dict[str, tuple[float, float]] = {}
    usecols = [c for cols in SIDES.values() for c in cols]
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            for i, chunk in enumerate(pd.read_csv(fh, usecols=usecols, chunksize=chunksize)):
                for side, (name_c, lat_c, lon_c, t_c) in SIDES.items():
                    part = chunk.dropna(subset=[name_c, t_c])
                    ts = _local_hour(part[t_c])
                    counts[side].append(part.groupby([part[name_c], ts]).size())
                    if side == "departures":
                        for row in part.drop_duplicates(name_c)[[name_c, lat_c, lon_c]].itertuples(
                            index=False
                        ):
                            coords.setdefault(row[0], (row[1], row[2]))
                print(f"  chunk {i + 1} done ({(i + 1) * chunksize:,} trips read)", flush=True)
    frames = []
    for side, parts in counts.items():
        s = pd.concat(parts).groupby(level=[0, 1]).sum().rename(side)
        s.index.names = ["name", "ts"]
        frames.append(s)
    hourly = pd.concat(frames, axis=1).fillna(0).astype("int64").reset_index()
    coord_df = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in coords.items()], columns=["name", "lat", "lon"]
    )
    return hourly, coord_df


def match_to_gbfs(top: pd.DataFrame, info: pd.DataFrame, max_m: float = 75.0) -> pd.DataFrame:
    info = info.assign(key=info["name"].map(_norm))
    by_name = dict(zip(info["key"], info["station_id"], strict=False))
    lat0 = np.radians(45.5)
    out = []
    for row in top.itertuples(index=False):
        sid = by_name.get(_norm(row.name))
        if sid is None:
            dy = (info["lat"] - row.lat) * 111_320
            dx = (info["lon"] - row.lon) * 111_320 * np.cos(lat0)
            dist = np.hypot(dx, dy)
            j = int(dist.idxmin())
            if dist[j] <= max_m:
                sid = info.loc[j, "station_id"]
        if sid is not None:
            g = info.loc[info["station_id"] == sid].iloc[0]
            out.append(
                {"station_id": str(sid), "trip_name": row.name, "name": g["name"],
                 "lat": g["lat"], "lon": g["lon"], "capacity": int(g["capacity"])}
            )
    return pd.DataFrame(out).drop_duplicates("station_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--stations", type=int, default=50)
    ap.add_argument("--start", default="2025-04-15")
    ap.add_argument("--end", default="2025-11-15")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    print("Counting trips (this streams ~2.8 GB, a few minutes)...")
    hourly, coords = count_trips(args.zip_path)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    hourly = hourly[(hourly["ts"] >= start) & (hourly["ts"] < end)]

    totals = hourly.groupby("name")["departures"].sum().sort_values(ascending=False)
    top = coords.set_index("name").loc[totals.index[: args.stations * 2]].reset_index()

    print("Matching stations to the live GBFS feed...")
    info = live.fetch_station_information()
    matched = match_to_gbfs(top, info).head(args.stations)
    print(f"  matched {len(matched)} stations")

    hourly = hourly.merge(
        matched[["trip_name", "station_id"]], left_on="name", right_on="trip_name"
    )
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    grid = pd.MultiIndex.from_product([matched["station_id"], hours], names=["station_id", "ts"])
    demand = (
        hourly.groupby(["station_id", "ts"])[["departures", "arrivals"]].sum()
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    print("Fetching hourly weather from Open-Meteo...")
    last_day = (end - pd.Timedelta(days=1)).date().isoformat()
    weather = bdata.fetch_weather_archive(args.start, last_day)
    demand = demand.merge(weather, on="ts", how="left")
    demand[["temperature", "precipitation"]] = (
        demand[["temperature", "precipitation"]].interpolate(limit_direction="both")
    )

    args.out.mkdir(parents=True, exist_ok=True)
    demand.to_csv(args.out / "hourly_demand.csv.gz", index=False, float_format="%.1f")
    matched.drop(columns="trip_name").to_csv(args.out / "stations.csv", index=False)
    n = matched["station_id"].nunique()
    print(f"Wrote {len(demand):,} rows for {n} stations to {args.out}")


if __name__ == "__main__":
    main()

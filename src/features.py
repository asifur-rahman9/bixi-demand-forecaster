"""Builds the model inputs. Used by training, prediction and the app so they always match."""

import holidays
import pandas as pd

FEATURES = ["station_id", "hour", "day_of_week", "month", "is_weekend", "is_holiday",
            "temperature", "precipitation"]
TARGETS = ["departures", "arrivals"]


def is_holiday(ts):
    years = sorted(set(ts.dt.year))
    days = holidays.country_holidays("CA", subdiv="QC", years=years)
    return ts.dt.date.isin(days.keys()).astype("int8")


def build_features(df, stations):
    """df needs columns station_id, ts, temperature, precipitation.

    stations is the list from training, so the station categories are always in the same order.
    """
    ts = pd.to_datetime(df["ts"])
    X = pd.DataFrame(index=df.index)
    X["station_id"] = pd.Categorical(df["station_id"].astype(str), categories=list(stations))
    X["hour"] = ts.dt.hour
    X["day_of_week"] = ts.dt.day_of_week
    X["month"] = ts.dt.month
    X["is_weekend"] = (ts.dt.day_of_week >= 5).astype("int8")
    X["is_holiday"] = is_holiday(ts)
    X["temperature"] = df["temperature"].astype(float)
    X["precipitation"] = df["precipitation"].astype(float)
    return X[FEATURES]

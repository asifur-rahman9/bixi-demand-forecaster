import pandas as pd

from src.features import FEATURES, build_features


def test_columns_and_calendar_values():
    df = pd.DataFrame({"station_id": ["10", "10"],
                       "ts": pd.to_datetime(["2025-07-01 08:00", "2025-07-05 17:00"]),
                       "temperature": [20.0, 25.0], "precipitation": [0.0, 1.5]})
    X = build_features(df, ["10", "20"])
    assert list(X.columns) == FEATURES
    assert X["hour"].tolist() == [8, 17]
    assert X["is_weekend"].tolist() == [0, 1]  # Tuesday, Saturday
    assert X["is_holiday"].tolist() == [1, 0]  # July 1 is Canada Day


def test_station_categories_follow_training_order():
    df = pd.DataFrame({"station_id": ["20"], "ts": pd.to_datetime(["2025-07-02"]),
                       "temperature": [20.0], "precipitation": [0.0]})
    X = build_features(df, ["10", "20"])
    assert list(X["station_id"].cat.categories) == ["10", "20"]

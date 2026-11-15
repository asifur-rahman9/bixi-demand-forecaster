import pandas as pd
import pytest

from src.predict import Forecaster
from src.train import baseline_last_week, time_split, train


def test_time_split_keeps_the_future_out_of_training(demand):
    train_df, test_df = time_split(demand, test_weeks=2)
    assert train_df["ts"].max() < test_df["ts"].min()
    assert len(train_df) + len(test_df) == len(demand)


def test_last_week_baseline_looks_back_one_week(demand):
    _, test_df = time_split(demand, test_weeks=1)
    pred = baseline_last_week(demand, test_df, "departures")
    row = test_df.iloc[0]
    week_ago = demand[(demand["station_id"] == row["station_id"])
                      & (demand["ts"] == row["ts"] - pd.Timedelta(weeks=1))]
    assert pred[0] == week_ago["departures"].iloc[0]


def test_trained_model_beats_baseline_and_predicts(demand, tmp_path):
    meta = train(demand, tmp_path, test_weeks=1, max_iter=60)
    for target in ["departures", "arrivals"]:
        assert meta["results"][target]["model_mae"] < meta["results"][target]["baseline_mae"]

    fc = Forecaster.load(tmp_path)
    weather = pd.DataFrame({"ts": pd.date_range("2025-07-20 06:00", periods=6, freq="h"),
                            "temperature": 22.0, "precipitation": 0.0})
    out = fc.predict("10", weather)
    assert list(out.columns) == ["ts", "departures", "arrivals"]
    assert (out[["departures", "arrivals"]] >= 0).all().all()

    with pytest.raises(ValueError):
        fc.predict("99", weather)  # not a trained station

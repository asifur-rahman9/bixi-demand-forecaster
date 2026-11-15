"""Train and evaluate the demand models.

    python -m src.train

I hold out the last 4 weeks so the model is tested on data from after its training data,
and compare it to a simple baseline: the same hour one week earlier.
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import MODEL_DIR
from src.data import load_demand
from src.features import FEATURES, TARGETS, build_features


def time_split(df, test_weeks=4):
    cutoff = df["ts"].max().ceil("D") - pd.Timedelta(weeks=test_weeks)
    return df[df["ts"] < cutoff].copy(), df[df["ts"] >= cutoff].copy()


def baseline_last_week(full, test, target):
    """Guess: same station, same hour, one week earlier."""
    lag = full[["station_id", "ts", target]].copy()
    lag["ts"] = lag["ts"] + pd.Timedelta(weeks=1)
    merged = test[["station_id", "ts"]].merge(lag, on=["station_id", "ts"], how="left")
    return merged[target].to_numpy(dtype=float)


def mae(y, pred):
    ok = ~np.isnan(pred)
    return round(float(np.abs(y[ok] - pred[ok]).mean()), 2)


def make_model(max_iter=400):
    # Poisson loss works well for counts (trips per hour can't be negative)
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.08, max_iter=max_iter,
                                         max_leaf_nodes=63, categorical_features="from_dtype",
                                         random_state=0)


def train(df, out_dir=MODEL_DIR, test_weeks=4, max_iter=400):
    stations = sorted(df["station_id"].astype(str).unique())
    train_df, test_df = time_split(df, test_weeks)
    X_train = build_features(train_df, stations)
    X_test = build_features(test_df, stations)

    results = {}
    holdout = test_df[["station_id", "ts"] + TARGETS].copy()
    models = {}
    for target in TARGETS:
        model = make_model(max_iter).fit(X_train, train_df[target])
        pred = np.clip(model.predict(X_test), 0, None)
        holdout[f"{target}_pred"] = pred.round(2)
        y = test_df[target].to_numpy(dtype=float)
        results[target] = {
            "model_mae": mae(y, pred),
            "baseline_mae": mae(y, baseline_last_week(df, test_df, target)),
        }
        # After testing, train the final model on all the data
        models[target] = make_model(max_iter).fit(build_features(df, stations), df[target])

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_dir / "model.joblib", compress=3)
    holdout.to_csv(out_dir / "holdout.csv.gz", index=False)
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURES,
        "targets": TARGETS,
        "stations": stations,
        "data_start": str(df["ts"].min()),
        "data_end": str(df["ts"].max()),
        "holdout_start": str(test_df["ts"].min()),
        "rows": len(df),
        "results": results,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    meta = train(load_demand())
    print(f"Trained on {meta['rows']:,} rows for {len(meta['stations'])} stations")
    for target, r in meta["results"].items():
        print(f"{target}: model MAE {r['model_mae']}, last-week baseline MAE {r['baseline_mae']}")

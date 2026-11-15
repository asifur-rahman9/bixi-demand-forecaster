# Design notes

## Data

`scripts/build_dataset.py` runs once to build the dataset:

1. Reads the 2025 trip file (2.8 GB CSV inside a zip) in chunks, so it doesn't run out of memory.
2. Counts departures (start station) and arrivals (end station) for every station and hour.
3. Keeps the 50 busiest stations and matches them to the IDs in the live feed. The trip file only has station names and coordinates, so it matches by name first and then by the closest station within 75 m.
4. Fills in zero for hours with no trips, then adds hourly temperature and rain from Open-Meteo.

The result is 1.6 MB and committed, so the app and tests don't need to download anything.

One bug I ran into: rounding times to the hour crashed on the night daylight saving time ends (Nov 2, the 1:00 hour happens twice). Removing the timezone before rounding fixed it.

## Features

All model inputs come from `build_features` in `src/features.py`. Training, prediction and the app all use it, so they can't get out of sync (the first version had three copies).

- station, hour, day of week, month
- weekend and Quebec holidays (using the `holidays` package)
- temperature and precipitation

I didn't use "trips in the last hour" style features. That way a forecast only needs a time, a station and the weather, so the app can forecast 48 hours ahead.

## Model

scikit-learn's `HistGradientBoostingRegressor` with Poisson loss, since the targets are counts that can't be negative. One model for departures and one for arrivals.

## Testing the model

The first version used a random train/test split, which mixes future data into training. Now the last 4 weeks are held out, and the model is compared to a simple guess: the same station and hour one week earlier.

| Target | Model MAE | Same hour last week |
|---|---:|---:|
| Departures | 2.74 | 4.03 |
| Arrivals | 2.63 | 3.83 |

After testing, the final model is trained again on all the data so it includes the most recent weeks.

## Limitations

- Only one season (spring to fall) and 50 stations.
- One weather point for the whole city.
- The forecast is trips per hour. It doesn't account for BIXI trucks moving bikes between stations.

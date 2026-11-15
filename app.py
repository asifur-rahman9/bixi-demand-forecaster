"""BIXI Station Monitor. Run with: streamlit run app.py"""

import pandas as pd
import pydeck as pdk
import streamlit as st

from src import MODEL_DIR
from src.data import fetch_weather_forecast, load_stations
from src.live import fetch_live
from src.predict import Forecaster

st.set_page_config(page_title="BIXI Station Monitor", page_icon="🚲", layout="wide")

COLORS = {  # map dot colors (RGBA)
    "ok": [46, 157, 91, 200],
    "nearly empty": [214, 69, 61, 220],
    "nearly full": [47, 111, 219, 220],
    "offline": [138, 138, 138, 160],
}


@st.cache_resource
def get_forecaster():
    return Forecaster.load()


@st.cache_data
def get_stations():
    return load_stations()


@st.cache_data(ttl=60)  # refresh live data at most once a minute
def get_live():
    return fetch_live()


@st.cache_data(ttl=1800)
def get_weather():
    return fetch_weather_forecast(48)


@st.cache_data
def get_holdout():
    return pd.read_csv(MODEL_DIR / "holdout.csv.gz", parse_dates=["ts"], dtype={"station_id": str})


st.title("BIXI Station Monitor")
st.caption("Live bike availability for Montreal's BIXI stations, plus hourly demand forecasts "
           "based on the weather.")

fc = get_forecaster()
stations = get_stations()
names = dict(zip(stations["station_id"], stations["name"]))

tab_live, tab_forecast, tab_model = st.tabs(["Live", "Forecast", "Model"])

with tab_live:
    try:
        live = get_live()
    except Exception as e:
        st.error(f"Couldn't load the live BIXI feed: {e}")
        live = None

    if live is not None:
        counts = live["state"].value_counts()
        c = st.columns(4)
        c[0].metric("Stations", len(live))
        c[1].metric("Nearly empty", int(counts.get("nearly empty", 0)))
        c[2].metric("Nearly full", int(counts.get("nearly full", 0)))
        c[3].metric("Offline", int(counts.get("offline", 0)))

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=live.assign(color=live["state"].map(COLORS))[
                ["name", "lat", "lon", "bikes", "docks", "state", "color", "capacity"]],
            get_position="[lon, lat]", get_fill_color="color",
            get_radius="25 + capacity", radius_min_pixels=2, pickable=True)
        view = pdk.ViewState(latitude=45.515, longitude=-73.58, zoom=11.6)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style=None,
                                 tooltip={"text": "{name}\n{bikes} bikes, {docks} docks"}),
                        height=480)
        st.caption("Green: ok · Red: 2 or fewer bikes · Blue: 2 or fewer docks · Grey: offline")

        problems = live[live["state"].isin(["nearly empty", "nearly full"])]
        st.subheader(f"Needs attention ({len(problems)})")
        st.dataframe(problems.sort_values("state")[["name", "state", "bikes", "docks", "capacity"]],
                     hide_index=True, use_container_width=True)

with tab_forecast:
    try:
        weather = get_weather()
    except Exception as e:
        st.error(f"Couldn't load the weather forecast: {e}")
        weather = None

    if weather is not None:
        sid = st.selectbox("Station", fc.stations, format_func=lambda s: names.get(s, s))
        forecast = fc.predict(sid, weather)
        c = st.columns(3)
        c[0].metric("Departures, next 48 h", int(forecast["departures"].sum()))
        c[1].metric("Arrivals, next 48 h", int(forecast["arrivals"].sum()))
        c[2].metric("Temperature now", f"{weather['temperature'].iloc[0]:.0f} °C")
        st.markdown("**Forecast trips per hour, next 48 hours**")
        st.line_chart(forecast.set_index("ts")[["departures", "arrivals"]], height=300)

with tab_model:
    meta = fc.metadata
    st.markdown(
        f"Trained on **{meta['rows']:,}** station-hours from the **{len(meta['stations'])}** "
        f"busiest stations ({meta['data_start'][:10]} to {meta['data_end'][:10]}). "
        f"The scores are from the last 4 weeks (from {meta['holdout_start'][:10]}), "
        "which the model didn't see while training.")
    rows = [{"target": t, "model MAE": r["model_mae"], "same hour last week MAE": r["baseline_mae"]}
            for t, r in meta["results"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    target = st.radio("Show", meta["targets"], horizontal=True)
    sid = st.selectbox("Station", fc.stations, format_func=lambda s: names.get(s, s), key="hold")
    hold = get_holdout()
    week = hold[hold["station_id"] == sid].sort_values("ts").tail(24 * 7)
    st.markdown("**Actual vs predicted, last week of the test period**")
    st.line_chart(week.set_index("ts")[[target, f"{target}_pred"]], height=300)

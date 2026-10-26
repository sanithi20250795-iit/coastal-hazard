"""Streamlit dashboard.  Run from the project root:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from config import LEVEL_NAMES, SKILLS  # noqa: E402
from src import live  # noqa: E402

COLORS = {0: "#2e9e5b", 1: "#f0a500", 2: "#d64545"}

st.set_page_config(page_title="Coastal Hazard Forecast", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Fetching the latest forecast ...")
def load():
    res = live.run()
    return res["df"], res["meta"]


st.title("Coastal Hazard Forecast")
c1, c2 = st.columns([1, 2])
c1.selectbox("Spot", ["Matara"], help="Matara is the only validated site.")
skill = c2.radio("Skill level", [s.capitalize() for s in SKILLS], horizontal=True).lower()

try:
    df, meta = load()
except Exception as exc:
    st.error(f"Forecast unavailable: {exc}")
    st.stop()

if meta["mode"] == "cached":
    st.warning(f"Forecast source unreachable. Showing the last stored forecast ({meta['age_hours']} h old).")
elif meta["mode"] == "stale":
    st.error(f"Forecast source unreachable and the stored forecast is {meta['age_hours']} h old. "
             "Showing the UNCORRECTED forecast with rule-based levels.")

lvl = df[f"level_{skill}"].astype(int)
top = int(lvl.max())
st.markdown(
    f"<div style='padding:16px;border-radius:8px;background:{COLORS[top]}22;border-left:8px solid {COLORS[top]}'>"
    f"<b>{LEVEL_NAMES[top]}</b> &nbsp; {live.summary_text(df, skill, meta)}</div>", unsafe_allow_html=True)

st.subheader("Next 48 hours")
blocks = "".join(
    f"<div title='{t:%a %H:%M}: {LEVEL_NAMES[int(v)]}' style='flex:1;height:34px;background:{COLORS[int(v)]};"
    f"margin:0 1px;border-radius:2px'></div>" for t, v in zip(df["time_local"], lvl))
st.markdown(f"<div style='display:flex'>{blocks}</div>", unsafe_allow_html=True)
st.caption(f"Now ({df['time_local'].iloc[0]:%a %H:%M}) to +{len(df)} h. Green = low, amber = moderate, red = high.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Wave height: raw forecast vs corrected for Matara")
    fig = go.Figure()
    fig.add_scatter(x=df["time_local"], y=df["f_hs"], name="Raw forecast", line=dict(color="#888"))
    fig.add_scatter(x=df["time_local"], y=df["hs_corr"], name="Corrected for Matara", line=dict(color="#0f7b6c"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Wave height (m)", xaxis_title="Local time")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Why this level")
    peak = df.loc[df[f"level_{skill}"].idxmax()]
    st.write(f"At the peak hour ({peak['time_local']:%a %H:%M}):")
    for item in (peak[f"drivers_{skill}"] or "no single dominant factor").split("; "):
        st.markdown(f"- {item}")
    st.metric("Corrected wave height", f"{peak['hs_corr']:.1f} m", f"{peak['hs_corr'] - peak['f_hs']:+.1f} m vs raw forecast")

st.download_button("Download daily summary", live.summary_text(df, skill, meta), file_name="daily_summary.txt")
st.caption(f"Forecast updated {meta['age_hours']} h ago ({meta['mode']}). {meta['note']} Validated at Matara only; "
           "not for emergency decisions.")

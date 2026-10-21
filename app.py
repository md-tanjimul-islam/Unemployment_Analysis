"""Streamlit dashboard: Unemployment Analysis for India (CodeAlpha Task 2).

Run locally:   streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from unemployment import analysis as an  # noqa: E402
from unemployment import config, data, plots  # noqa: E402

st.set_page_config(page_title="India Unemployment Analysis", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data...")
def load_all():
    df_area = data.load_state_area()
    df_state = data.load_state()
    panel = data.build_panel(df_area, df_state)
    return df_area, df_state, panel


df_area, df_state, panel_all = load_all()

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
st.sidebar.title("Filters")
regions = sorted(panel_all["macro_region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Macro-region", regions, default=regions)
available_states = sorted(panel_all[panel_all["macro_region"].isin(sel_regions)]["state"].unique())
sel_states = st.sidebar.multiselect("States / UTs", available_states, default=available_states)
st.sidebar.caption(
    "Aggregates are labour-force weighted across the selected states. "
    "Clear a filter to narrow the analysis."
)
st.sidebar.markdown("---")
st.sidebar.caption("CodeAlpha Data Science Internship · Task 2")

panel = panel_all[panel_all["state"].isin(sel_states)]
area_f = df_area[df_area["state"].isin(sel_states)]
state_f = df_state[df_state["state"].isin(sel_states)]

st.title("📊 Unemployment Analysis: India, 2019-2020")
st.caption(
    "Monthly estimates by state, May 2019 - Oct 2020, with a focus on the impact of COVID-19. "
    "Source: Centre for Monitoring Indian Economy (CMIE) data supplied with the CodeAlpha task."
)

if panel.empty:
    st.warning("No states selected. Choose at least one state in the sidebar.")
    st.stop()

nat = an.national_trend(panel)
impact = an.covid_impact_by_state(panel)

tab_over, tab_covid, tab_state, tab_ur, tab_map, tab_data = st.tabs(
    ["Overview", "COVID-19 impact", "States & regions", "Urban vs rural", "Map", "Insights & data"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    pre = nat[nat["phase"] == "Pre-COVID"]["unemployment_rate"].mean()
    peak = nat.loc[nat["unemployment_rate"].idxmax()]
    last = nat.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pre-COVID average", f"{pre:.1f}%")
    c2.metric("Peak", f"{peak['unemployment_rate']:.1f}%", f"{peak['date']:%b %Y}", delta_color="off")
    c3.metric(f"Latest ({last['date']:%b %Y})", f"{last['unemployment_rate']:.1f}%",
              f"{last['unemployment_rate'] - pre:+.1f} pp vs pre-COVID", delta_color="inverse")
    c4.metric("States / UTs", f"{panel['state'].nunique()}")

    st.plotly_chart(plots.national_trend_fig(nat), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(plots.phase_fig(an.phase_summary(panel)), width="stretch")
    with right:
        st.markdown("##### Month-over-month change (pp)")
        mom = nat[["date", "unemployment_rate", "mom_change_pp"]].copy()
        mom["date"] = mom["date"].dt.strftime("%b %Y")
        mom.columns = ["Month", "Rate (%)", "Change (pp)"]
        st.dataframe(mom.round(2), hide_index=True, width="stretch", height=340)

# --------------------------------------------------------------------------- #
# COVID impact
# --------------------------------------------------------------------------- #
with tab_covid:
    sig = an.covid_significance(panel)
    st.markdown(
        f"**Statistical test.** Across state-months, the average rate was **{sig['pre_mean']:.1f}%** before COVID "
        f"and **{sig['lockdown_mean']:.1f}%** during the Apr-May 2020 lockdown, a difference of "
        f"**{sig['difference_pp']:.1f} pp** (Welch t = {sig['welch_t']:.2f}, p = {sig['welch_p']:.1e}; "
        f"Mann-Whitney p = {sig['mannwhitney_p']:.1e}; Cohen's d = {sig['cohens_d']:.2f})."
    )
    top_n = st.slider("States to show", 5, max(5, len(impact)), min(15, len(impact)))
    st.plotly_chart(plots.covid_impact_fig(impact, top_n), width="stretch")

    st.markdown("##### State-level impact table")
    show = impact.copy()
    show["peak_month"] = show["peak_month"].dt.strftime("%b %Y")
    show = show.drop(columns=["latest_month"]).rename(columns={
        "state": "State", "baseline_rate": "Pre-COVID avg (%)", "peak_rate": "Peak (%)", "peak_month": "Peak month",
        "latest_rate": "Latest (%)", "macro_region": "Region", "increase_pp": "Rise to peak (pp)",
        "latest_vs_baseline_pp": "Latest vs baseline (pp)", "recovered_share": "Share of rise unwound",
    })
    st.dataframe(show, hide_index=True, width="stretch")

    loss = an.employment_loss(panel)
    st.markdown("##### Employment loss: Feb 2020 vs lockdown trough")
    st.plotly_chart(
        plots.px.bar(loss.head(15).iloc[::-1], x="loss_pct", y="state", orientation="h",
                     labels={"loss_pct": "Fewer employed people (%)", "state": ""},
                     color_discrete_sequence=[plots.PALETTE["danger"]]).update_layout(
            height=470, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10)),
        width="stretch",
    )
    st.caption("Estimated Employed counts are survey-based estimates and move more sharply than headcounts would.")

# --------------------------------------------------------------------------- #
# States & regions
# --------------------------------------------------------------------------- #
with tab_state:
    default = [s for s in ["Delhi", "Maharashtra", "Tamil Nadu", "Bihar"] if s in sel_states] or sel_states[:3]
    chosen = st.multiselect("Compare states", sel_states, default=default)
    if chosen:
        st.plotly_chart(plots.state_lines_fig(panel, chosen), width="stretch")
    st.plotly_chart(plots.region_fig(an.regional_trend(panel)), width="stretch")
    st.plotly_chart(plots.heatmap_fig(panel), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        month = st.selectbox("Ranking month", sorted(panel["date"].unique(), reverse=True),
                             format_func=lambda d: pd.Timestamp(d).strftime("%B %Y"))
        rank = an.state_ranking(panel, month).rename(columns={
            "state": "State", "macro_region": "Region", "unemployment_rate": "Unemployment (%)",
            "labour_participation_rate": "Labour participation (%)"})
        st.dataframe(rank.round(2), hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Most volatile states")
        st.dataframe(an.state_volatility(panel).head(12), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Urban vs rural
# --------------------------------------------------------------------------- #
with tab_ur:
    if area_f.empty:
        st.info("No urban/rural data for the selected states.")
    else:
        ur = an.urban_rural_trend(area_f)
        st.plotly_chart(plots.urban_rural_fig(ur), width="stretch")
        st.plotly_chart(plots.gap_fig(ur), width="stretch")
        st.caption("Urban/rural split is only available in the first file (May 2019 - Jun 2020).")

# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
with tab_map:
    months = sorted(state_f["date"].unique())
    if not months:
        st.info("No state-level data for the selection.")
    else:
        month = st.select_slider("Month", options=months, value=months[3] if len(months) > 3 else months[0],
                                 format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
        st.plotly_chart(plots.map_fig(state_f, pd.Timestamp(month)), width="stretch")
        st.caption("Bubble size and colour show the unemployment rate. Coordinates are state centroids.")

# --------------------------------------------------------------------------- #
# Insights & data
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Key insights")
    if len(sel_states) == panel_all["state"].nunique():
        for line in an.generate_insights(panel, area_f):
            st.markdown(f"- {line}")
    else:
        st.info("Automatic insights are written for the full dataset. Select all states to see them.")

    st.subheader("Same-month comparison, 2019 vs 2020")
    yoy = an.same_month_yoy(panel)
    yoy["month"] = pd.to_datetime(yoy["month"], format="%m").dt.strftime("%B")
    st.dataframe(yoy, hide_index=True, width="stretch")
    st.caption(
        "Only ~18 months of data exist, so formal seasonal decomposition is not possible. "
        "May and June are the only months observed in both years."
    )

    st.subheader("Data quality")
    st.json({k: str(v) for k, v in data.data_quality_report(df_area, df_state).items()})

    st.subheader("Harmonised dataset")
    out = panel.copy()
    out["date"] = out["date"].dt.date
    st.dataframe(out, hide_index=True, width="stretch", height=300)
    st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name="unemployment_panel_2019_2020.csv", mime="text/csv")

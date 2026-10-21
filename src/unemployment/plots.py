"""Interactive Plotly figures used by the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config

PALETTE = {
    "primary": "#2563eb",
    "danger": "#dc2626",
    "good": "#16a34a",
    "muted": "#64748b",
    "urban": "#7c3aed",
    "rural": "#d97706",
}
PHASE_COLORS = {
    "Pre-COVID": "#94a3b8",
    "Onset": "#f59e0b",
    "Lockdown": "#dc2626",
    "Recovery": "#16a34a",
}


def _style(fig: go.Figure, title: str, height: int = 430, yaxis_title: str = "Unemployment rate (%)") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=17)),
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        yaxis_title=yaxis_title,
        xaxis_title=None,
        template="plotly_white",
    )
    return fig


def _shade_lockdown(fig: go.Figure) -> go.Figure:
    fig.add_vrect(
        x0=config.LOCKDOWN_START, x1=config.LOCKDOWN_END,
        fillcolor=PALETTE["danger"], opacity=0.10, line_width=0,
        annotation_text="National lockdown", annotation_position="top left",
        annotation_font_size=11,
    )
    return fig


def national_trend_fig(nat: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nat["date"], y=nat["unemployment_rate"], mode="lines+markers", name="Monthly rate",
        line=dict(color=PALETTE["primary"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=nat["date"], y=nat["rolling_3m"], mode="lines", name="3-month average",
        line=dict(color=PALETTE["muted"], dash="dash", width=2),
    ))
    _shade_lockdown(fig)
    return _style(fig, "Aggregate unemployment rate (labour-force weighted)")


def state_lines_fig(panel: pd.DataFrame, states: list[str]) -> go.Figure:
    d = panel[panel["state"].isin(states)]
    fig = px.line(d, x="date", y="unemployment_rate", color="state", markers=True)
    _shade_lockdown(fig)
    return _style(fig, "Unemployment rate by state")


def covid_impact_fig(impact: pd.DataFrame, top_n: int = 15) -> go.Figure:
    d = impact.sort_values("increase_pp", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["state"], x=d["baseline_rate"], name="Pre-COVID average",
                         orientation="h", marker_color=PALETTE["muted"]))
    fig.add_trace(go.Bar(y=d["state"], x=d["peak_rate"], name="Peak (Mar-Jun 2020)",
                         orientation="h", marker_color=PALETTE["danger"]))
    fig.add_trace(go.Bar(y=d["state"], x=d["latest_rate"], name="Latest month",
                         orientation="h", marker_color=PALETTE["good"]))
    fig.update_layout(barmode="group", hovermode="y unified")
    return _style(fig, f"COVID-19 shock: top {top_n} states by rise in unemployment",
                  height=max(430, 28 * top_n * 3 // 2), yaxis_title=None).update_xaxes(title="Unemployment rate (%)")


def phase_fig(summary: pd.DataFrame) -> go.Figure:
    colors = [PHASE_COLORS[p] for p in summary["phase"]]
    fig = go.Figure(go.Bar(
        x=summary["label"], y=summary["avg_rate"], marker_color=colors,
        text=summary["avg_rate"].map("{:.1f}%".format), textposition="outside",
    ))
    fig.update_layout(hovermode="closest", showlegend=False)
    return _style(fig, "Average national unemployment by COVID-19 phase")


def heatmap_fig(panel: pd.DataFrame, cap: float = 40.0) -> go.Figure:
    pivot = panel.pivot_table(index="state", columns="date", values="unemployment_rate")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(go.Heatmap(
        z=pivot.clip(upper=cap).values,
        x=[d.strftime("%b %Y") for d in pivot.columns], y=pivot.index,
        colorscale="YlOrRd", zmax=cap, colorbar=dict(title="%"),
        customdata=pivot.values,
        hovertemplate="%{y}<br>%{x}: %{customdata:.1f}%<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), hovermode="closest")
    return _style(fig, f"Unemployment heatmap (colour capped at {cap:.0f}%)", height=760, yaxis_title=None)


def region_fig(reg: pd.DataFrame) -> go.Figure:
    fig = px.line(reg, x="date", y="unemployment_rate", color="macro_region", markers=True)
    _shade_lockdown(fig)
    return _style(fig, "Unemployment by macro-region")


def urban_rural_fig(ur: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ur["date"], y=ur["Urban"], name="Urban", mode="lines+markers",
                             line=dict(color=PALETTE["urban"], width=3)))
    fig.add_trace(go.Scatter(x=ur["date"], y=ur["Rural"], name="Rural", mode="lines+markers",
                             line=dict(color=PALETTE["rural"], width=3)))
    _shade_lockdown(fig)
    return _style(fig, "Urban vs rural unemployment (May 2019 - Jun 2020)")


def gap_fig(ur: pd.DataFrame) -> go.Figure:
    colors = [PALETTE["urban"] if v >= 0 else PALETTE["rural"] for v in ur["gap_urban_minus_rural"]]
    fig = go.Figure(go.Bar(x=ur["date"], y=ur["gap_urban_minus_rural"], marker_color=colors))
    fig.add_hline(y=0, line_color=PALETTE["muted"])
    fig.update_layout(showlegend=False)
    return _style(fig, "Urban minus rural unemployment gap (pp)", height=340, yaxis_title="Percentage points")


def map_fig(df_state: pd.DataFrame, month: pd.Timestamp) -> go.Figure:
    d = df_state[df_state["date"] == month].copy()
    d["size"] = d["unemployment_rate"].clip(lower=1)
    fig = px.scatter_geo(
        d, lat="lat", lon="lon", size="size", color="unemployment_rate",
        hover_name="state", color_continuous_scale="YlOrRd", range_color=(0, 40),
        hover_data={"size": False, "lat": False, "lon": False,
                    "unemployment_rate": ":.1f", "macro_region": True},
        size_max=38,
    )
    fig.update_geos(
        scope="asia", center=dict(lat=22.5, lon=80), projection_scale=4.3,
        showcountries=True, countrycolor="#cbd5e1", showland=True, landcolor="#f8fafc",
        showocean=True, oceancolor="#eff6ff",
    )
    fig.update_layout(
        title=dict(text=f"Unemployment rate by state, {month:%B %Y}", x=0.01, font=dict(size=17)),
        height=620, margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(title="%"), template="plotly_white",
    )
    return fig

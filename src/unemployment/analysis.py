"""Analytical functions: trends, COVID-19 impact, regional and urban/rural comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .data import add_phase, weighted_rate


def _round_numeric(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """Round numeric columns only (DataFrame.round warns on datetime columns)."""
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].round(digits)
    return df


# --------------------------------------------------------------------------- #
# National and regional trends
# --------------------------------------------------------------------------- #
def national_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """Labour-force-weighted unemployment rate across all states, per month."""
    rows = []
    for date, g in panel.groupby("date"):
        rows.append(
            {
                "date": date,
                "unemployment_rate": weighted_rate(g),
                "employed": g["employed"].sum(),
                "states_reporting": g["state"].nunique(),
            }
        )
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["mom_change_pp"] = out["unemployment_rate"].diff()
    out["rolling_3m"] = out["unemployment_rate"].rolling(3, min_periods=1).mean()
    return add_phase(out)


def regional_trend(panel: pd.DataFrame) -> pd.DataFrame:
    """Weighted unemployment rate per macro-region (North, South, ...) per month."""
    rows = []
    for (region, date), g in panel.dropna(subset=["macro_region"]).groupby(["macro_region", "date"]):
        rows.append({"macro_region": region, "date": date, "unemployment_rate": weighted_rate(g)})
    return pd.DataFrame(rows).sort_values(["macro_region", "date"]).reset_index(drop=True)


def urban_rural_trend(df_area: pd.DataFrame) -> pd.DataFrame:
    """Weighted unemployment rate for Urban vs Rural areas, with the gap in pp."""
    rows = []
    for (area, date), g in df_area.groupby(["area", "date"]):
        rows.append({"area": area, "date": date, "unemployment_rate": weighted_rate(g)})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="date", columns="area", values="unemployment_rate")
    wide["gap_urban_minus_rural"] = wide["Urban"] - wide["Rural"]
    return wide.reset_index()


# --------------------------------------------------------------------------- #
# COVID-19 impact
# --------------------------------------------------------------------------- #
def phase_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Average national rate in each COVID phase (average of monthly national rates)."""
    nat = national_trend(panel)
    out = (
        nat.groupby("phase", observed=True)["unemployment_rate"]
        .agg(["mean", "min", "max", "count"])
        .rename(columns={"mean": "avg_rate", "min": "min_rate", "max": "max_rate", "count": "months"})
        .reset_index()
    )
    out["label"] = out["phase"].map(config.PHASE_LABELS)
    return out.round(2)


def covid_impact_by_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-state baseline, peak and latest unemployment rates.

    * baseline  = mean rate in the Pre-COVID phase
    * peak      = highest monthly rate from March to June 2020
    * latest    = most recent month available
    * recovered = share of the peak-above-baseline jump that has been unwound
    """
    shock_window = panel[(panel["date"] >= "2020-03-01") & (panel["date"] <= "2020-06-30")]
    base = panel[panel["phase"] == "Pre-COVID"].groupby("state")["unemployment_rate"].mean()
    peak_idx = shock_window.groupby("state")["unemployment_rate"].idxmax().dropna()
    peak = panel.loc[peak_idx].set_index("state")[["unemployment_rate", "date"]]
    peak.columns = ["peak_rate", "peak_month"]
    latest = panel.sort_values("date").groupby("state").tail(1).set_index("state")[
        ["unemployment_rate", "date", "macro_region"]
    ]
    latest.columns = ["latest_rate", "latest_month", "macro_region"]

    out = pd.concat([base.rename("baseline_rate"), peak, latest], axis=1).dropna(
        subset=["baseline_rate", "peak_rate"]
    )
    # Drop states whose "latest" month is not the end of the series (e.g. Chandigarh).
    out = out[out["latest_month"] == panel["date"].max()]
    out["increase_pp"] = out["peak_rate"] - out["baseline_rate"]
    out["latest_vs_baseline_pp"] = out["latest_rate"] - out["baseline_rate"]
    jump = out["peak_rate"] - out["baseline_rate"]
    out["recovered_share"] = np.where(
        jump > 0, ((out["peak_rate"] - out["latest_rate"]) / jump).clip(0, 1), np.nan
    )
    return _round_numeric(out.sort_values("increase_pp", ascending=False)).reset_index()


def covid_significance(panel: pd.DataFrame) -> dict:
    """Welch t-test and Mann-Whitney U comparing state-month rates before vs during lockdown."""
    pre = panel.loc[panel["phase"] == "Pre-COVID", "unemployment_rate"].dropna()
    lock = panel.loc[panel["phase"] == "Lockdown", "unemployment_rate"].dropna()
    t = stats.ttest_ind(lock, pre, equal_var=False)
    u = stats.mannwhitneyu(lock, pre, alternative="greater")
    pooled = np.sqrt((pre.var(ddof=1) + lock.var(ddof=1)) / 2)
    return {
        "pre_mean": float(pre.mean()),
        "lockdown_mean": float(lock.mean()),
        "difference_pp": float(lock.mean() - pre.mean()),
        "welch_t": float(t.statistic),
        "welch_p": float(t.pvalue),
        "mannwhitney_p": float(u.pvalue),
        "cohens_d": float((lock.mean() - pre.mean()) / pooled),
        "n_pre": int(len(pre)),
        "n_lockdown": int(len(lock)),
    }


# --------------------------------------------------------------------------- #
# Patterns, seasonality and relationships
# --------------------------------------------------------------------------- #
def same_month_yoy(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare months that exist in both 2019 and 2020 (May and June here).

    With only ~18 months of data a true seasonal decomposition is not
    possible, so a same-month year-over-year comparison is the honest check.
    """
    nat = national_trend(panel)
    nat["year"] = nat["date"].dt.year
    nat["month"] = nat["date"].dt.month
    both = nat.pivot(index="month", columns="year", values="unemployment_rate").dropna()
    both.columns = [f"rate_{c}" for c in both.columns]
    both["change_pp"] = both["rate_2020"] - both["rate_2019"]
    return both.round(2).reset_index()


def state_volatility(panel: pd.DataFrame) -> pd.DataFrame:
    """Standard deviation and range of monthly rates per state."""
    g = panel.groupby("state")["unemployment_rate"]
    out = pd.DataFrame({"std": g.std(), "min": g.min(), "max": g.max(), "mean": g.mean()})
    out["range"] = out["max"] - out["min"]
    return out.round(2).sort_values("std", ascending=False).reset_index()


def participation_correlation(panel: pd.DataFrame) -> dict:
    """Spearman correlation between unemployment rate and labour participation."""
    d = panel.dropna(subset=["unemployment_rate", "labour_participation_rate"])
    rho, p = stats.spearmanr(d["unemployment_rate"], d["labour_participation_rate"])
    return {"spearman_rho": float(rho), "p_value": float(p), "n": int(len(d))}


def state_ranking(panel: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rank states by unemployment rate in one month (default: latest)."""
    date = panel["date"].max() if date is None else pd.Timestamp(date)
    out = panel[panel["date"] == date][["state", "macro_region", "unemployment_rate", "labour_participation_rate"]]
    return out.sort_values("unemployment_rate", ascending=False).reset_index(drop=True)


def employment_loss(panel: pd.DataFrame) -> pd.DataFrame:
    """Employed persons in Feb 2020 vs the lockdown trough, for states reporting in both."""
    feb = panel[panel["date"] == "2020-02-29"].set_index("state")["employed"]
    lock = panel[panel["phase"] == "Lockdown"]
    trough = lock.loc[lock.groupby("state")["employed"].idxmin()].set_index("state")
    out = pd.DataFrame({"employed_feb_2020": feb, "employed_trough": trough["employed"],
                        "trough_month": trough["date"]}).dropna()
    out["loss"] = out["employed_feb_2020"] - out["employed_trough"]
    out["loss_pct"] = out["loss"] / out["employed_feb_2020"] * 100
    return _round_numeric(out.sort_values("loss_pct", ascending=False)).reset_index()


# --------------------------------------------------------------------------- #
# Automatic insight generation (numbers always come from the data)
# --------------------------------------------------------------------------- #
def _strength(r: float) -> str:
    a = abs(r)
    word = "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return f"{word} {'negative' if r < 0 else 'positive'}" if word != "negligible" else word


def generate_insights(panel: pd.DataFrame, df_area: pd.DataFrame) -> list[str]:
    nat = national_trend(panel)
    pre = nat[nat["phase"] == "Pre-COVID"]["unemployment_rate"].mean()
    peak_row = nat.loc[nat["unemployment_rate"].idxmax()]
    last = nat.iloc[-1]
    impact = covid_impact_by_state(panel)
    sig = covid_significance(panel)
    ur = urban_rural_trend(df_area)
    ur_pre = ur[ur["date"] <= "2020-02-29"]["gap_urban_minus_rural"].mean()
    ur_peak = ur.loc[ur[["Urban", "Rural"]].mean(axis=1).idxmax()]
    loss = employment_loss(panel)
    reg = regional_trend(panel)
    reg_latest = reg[reg["date"] == reg["date"].max()].sort_values("unemployment_rate")
    vol = state_volatility(panel)
    rho = participation_correlation(panel)

    insights = [
        f"National unemployment averaged {pre:.1f}% before COVID-19, jumped to a peak of "
        f"{peak_row['unemployment_rate']:.1f}% in {peak_row['date']:%B %Y}, and was {last['unemployment_rate']:.1f}% "
        f"by {last['date']:%B %Y}.",
        f"The lockdown effect is statistically significant: state-month rates were on average "
        f"{sig['difference_pp']:.1f} percentage points higher in Apr-May 2020 than before COVID "
        f"(Welch p = {sig['welch_p']:.1e}, Cohen's d = {sig['cohens_d']:.2f}).",
        f"The largest state-level shocks (baseline to peak) were {impact.iloc[0]['state']} "
        f"(+{impact.iloc[0]['increase_pp']:.1f} pp), {impact.iloc[1]['state']} "
        f"(+{impact.iloc[1]['increase_pp']:.1f} pp) and {impact.iloc[2]['state']} "
        f"(+{impact.iloc[2]['increase_pp']:.1f} pp).",
        f"By {last['date']:%B %Y}, {int((impact['latest_vs_baseline_pp'] <= 0).sum())} of {len(impact)} states were "
        f"back at or below their pre-COVID average rate.",
        f"Employment fell furthest in {loss.iloc[0]['state']} ({loss.iloc[0]['loss_pct']:.0f}% fewer employed "
        f"people at the lockdown trough than in Feb 2020).",
        f"Urban areas ran {ur_pre:+.1f} pp relative to rural areas before COVID; at the peak "
        f"({ur_peak['date']:%B %Y}) urban unemployment was {ur_peak['Urban']:.1f}% vs {ur_peak['Rural']:.1f}% rural.",
        f"In the latest month the lowest-unemployment macro-region was {reg_latest.iloc[0]['macro_region']} "
        f"({reg_latest.iloc[0]['unemployment_rate']:.1f}%) and the highest was "
        f"{reg_latest.iloc[-1]['macro_region']} ({reg_latest.iloc[-1]['unemployment_rate']:.1f}%).",
        f"{vol.iloc[0]['state']} had the most volatile series (std {vol.iloc[0]['std']:.1f} pp, "
        f"range {vol.iloc[0]['min']:.1f}%-{vol.iloc[0]['max']:.1f}%).",
        f"Across state-months, unemployment and labour participation show a {_strength(rho['spearman_rho'])} "
        f"relationship (Spearman rho = {rho['spearman_rho']:.2f}).",
    ]
    return insights

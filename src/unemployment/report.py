"""Static report generation: PNG figures (matplotlib), CSV tables and a markdown summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, safe on servers and CI
import matplotlib.pyplot as plt
import pandas as pd

from . import analysis as an
from . import config

COLORS = {"primary": "#2563eb", "danger": "#dc2626", "good": "#16a34a", "muted": "#64748b",
          "urban": "#7c3aed", "rural": "#d97706"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold",
})


def _lockdown(ax):
    ax.axvspan(config.LOCKDOWN_START, config.LOCKDOWN_END, color=COLORS["danger"], alpha=0.10, lw=0)
    ax.text(config.LOCKDOWN_START, 0.02, " Lockdown", color=COLORS["danger"], fontsize=8,
            va="bottom", transform=ax.get_xaxis_transform())


def _save(fig, name: str, out_dir: Path) -> Path:
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_national_trend(nat: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(nat["date"], nat["unemployment_rate"], marker="o", color=COLORS["primary"], lw=2.5, label="Monthly rate")
    ax.plot(nat["date"], nat["rolling_3m"], ls="--", color=COLORS["muted"], label="3-month average")
    peak = nat.loc[nat["unemployment_rate"].idxmax()]
    ax.annotate(f"Peak {peak['unemployment_rate']:.1f}%\n{peak['date']:%b %Y}", (peak["date"], peak["unemployment_rate"]),
                xytext=(18, -4), textcoords="offset points", fontsize=9, color=COLORS["danger"])
    ax.set(title="National unemployment rate (labour-force weighted)", ylabel="Unemployment rate (%)")
    _lockdown(ax)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "01_national_trend.png", out_dir)


def fig_phases(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = [COLORS["muted"], "#f59e0b", COLORS["danger"], COLORS["good"]]
    bars = ax.bar(summary["label"], summary["avg_rate"], color=cols[: len(summary)])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(title="Average national unemployment by COVID-19 phase", ylabel="Unemployment rate (%)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, "02_covid_phases.png", out_dir)


def fig_state_impact(impact: pd.DataFrame, out_dir: Path, top_n: int = 15) -> Path:
    d = impact.head(top_n).iloc[::-1]
    y = range(len(d))
    fig, ax = plt.subplots(figsize=(9, 6))
    h = 0.27
    ax.barh([i - h for i in y], d["baseline_rate"], h, color=COLORS["muted"], label="Pre-COVID average")
    ax.barh(list(y), d["peak_rate"], h, color=COLORS["danger"], label="Peak (Mar-Jun 2020)")
    ax.barh([i + h for i in y], d["latest_rate"], h, color=COLORS["good"], label="Latest month")
    ax.set_yticks(list(y), d["state"])
    ax.set(title=f"COVID-19 shock: top {top_n} states by rise in unemployment", xlabel="Unemployment rate (%)")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "03_state_impact.png", out_dir)


def fig_heatmap(panel: pd.DataFrame, out_dir: Path, cap: float = 40.0) -> Path:
    pv = panel.pivot_table(index="state", columns="date", values="unemployment_rate")
    pv = pv.loc[pv.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pv.clip(upper=cap).values, aspect="auto", cmap="YlOrRd", vmax=cap)
    ax.set_yticks(range(len(pv)), pv.index, fontsize=8)
    ax.set_xticks(range(len(pv.columns)), [d.strftime("%b %y") for d in pv.columns], rotation=60, fontsize=8)
    ax.grid(False)
    ax.set_title(f"Unemployment rate by state and month (colour capped at {cap:.0f}%)")
    fig.colorbar(im, ax=ax, label="%", shrink=0.8)
    return _save(fig, "04_heatmap.png", out_dir)


def fig_regions(reg: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, g in reg.groupby("macro_region"):
        ax.plot(g["date"], g["unemployment_rate"], marker="o", ms=3, label=name)
    ax.set(title="Unemployment by macro-region", ylabel="Unemployment rate (%)")
    _lockdown(ax)
    ax.legend(frameon=False, ncol=5, loc="upper left")
    return _save(fig, "05_regions.png", out_dir)


def fig_urban_rural(ur: pd.DataFrame, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1.4])
    ax1.plot(ur["date"], ur["Urban"], marker="o", color=COLORS["urban"], lw=2.5, label="Urban")
    ax1.plot(ur["date"], ur["Rural"], marker="o", color=COLORS["rural"], lw=2.5, label="Rural")
    ax1.set(title="Urban vs rural unemployment (May 2019 - Jun 2020)", ylabel="Unemployment rate (%)")
    ax1.legend(frameon=False)
    _lockdown(ax1)
    gap = ur["gap_urban_minus_rural"]
    ax2.bar(ur["date"], gap, width=20, color=[COLORS["urban"] if v >= 0 else COLORS["rural"] for v in gap])
    ax2.axhline(0, color=COLORS["muted"], lw=1)
    ax2.set(ylabel="Urban - rural (pp)")
    return _save(fig, "06_urban_rural.png", out_dir)


def fig_state_trends(panel: pd.DataFrame, out_dir: Path, states=("Delhi", "Maharashtra", "Tamil Nadu", "Bihar", "Kerala")) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for s in states:
        g = panel[panel["state"] == s]
        if not g.empty:
            ax.plot(g["date"], g["unemployment_rate"], marker="o", ms=3, label=s)
    ax.set(title="Selected state trajectories", ylabel="Unemployment rate (%)")
    _lockdown(ax)
    ax.legend(frameon=False, ncol=5, loc="upper left")
    return _save(fig, "07_state_trends.png", out_dir)


def build_report(panel: pd.DataFrame, df_area: pd.DataFrame, out_dir: Path | None = None) -> dict:
    """Write all figures, CSV tables and ``insights.md``. Returns a dict of produced paths."""
    out_dir = Path(out_dir or config.REPORTS_DIR)
    fig_dir, tbl_dir = out_dir / "figures", out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    nat = an.national_trend(panel)
    phases = an.phase_summary(panel)
    impact = an.covid_impact_by_state(panel)
    ur = an.urban_rural_trend(df_area)
    reg = an.regional_trend(panel)

    figures = [
        fig_national_trend(nat, fig_dir), fig_phases(phases, fig_dir), fig_state_impact(impact, fig_dir),
        fig_heatmap(panel, fig_dir), fig_regions(reg, fig_dir), fig_urban_rural(ur, fig_dir),
        fig_state_trends(panel, fig_dir),
    ]
    tables = {
        "national_trend.csv": nat.drop(columns="phase"),
        "phase_summary.csv": phases,
        "covid_impact_by_state.csv": impact,
        "employment_loss.csv": an.employment_loss(panel),
        "urban_rural_trend.csv": ur,
        "regional_trend.csv": reg,
        "state_volatility.csv": an.state_volatility(panel),
        "same_month_yoy.csv": an.same_month_yoy(panel),
        "harmonised_panel.csv": panel,
    }
    for name, df in tables.items():
        df.to_csv(tbl_dir / name, index=False)

    insights = an.generate_insights(panel, df_area)
    sig = an.covid_significance(panel)
    md = ["# Key insights", "", *[f"- {line}" for line in insights], "",
          "## Statistical test (lockdown vs pre-COVID, state-month rates)", "",
          f"- Pre-COVID mean: {sig['pre_mean']:.2f}% (n={sig['n_pre']})",
          f"- Lockdown mean: {sig['lockdown_mean']:.2f}% (n={sig['n_lockdown']})",
          f"- Welch t = {sig['welch_t']:.2f}, p = {sig['welch_p']:.2e}; Mann-Whitney p = {sig['mannwhitney_p']:.2e}; "
          f"Cohen's d = {sig['cohens_d']:.2f}", ""]
    (out_dir / "insights.md").write_text("\n".join(md), encoding="utf-8")
    return {"figures": figures, "tables": list(tables), "insights": insights}

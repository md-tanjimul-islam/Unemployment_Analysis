import pandas as pd

from unemployment import analysis as an


def test_national_trend_peaks_in_lockdown(panel):
    nat = an.national_trend(panel)
    peak = nat.loc[nat["unemployment_rate"].idxmax()]
    assert peak["date"] in (pd.Timestamp("2020-04-30"), pd.Timestamp("2020-05-31"))
    assert peak["unemployment_rate"] > 20


def test_phase_summary_order_and_lockdown_jump(panel):
    s = an.phase_summary(panel).set_index("phase")
    assert s.loc["Lockdown", "avg_rate"] > 2 * s.loc["Pre-COVID", "avg_rate"]
    assert s.loc["Recovery", "avg_rate"] < s.loc["Lockdown", "avg_rate"]


def test_covid_impact_columns_and_logic(panel):
    imp = an.covid_impact_by_state(panel)
    assert {"baseline_rate", "peak_rate", "latest_rate", "increase_pp"} <= set(imp.columns)
    assert (imp["peak_rate"] >= imp["baseline_rate"] - 1e-9).mean() > 0.9
    assert imp["increase_pp"].is_monotonic_decreasing
    assert imp["recovered_share"].dropna().between(0, 1).all()


def test_significance_is_significant(panel):
    sig = an.covid_significance(panel)
    assert sig["difference_pp"] > 5
    assert sig["welch_p"] < 0.001
    assert sig["cohens_d"] > 0.8


def test_urban_rural_gap_definition(df_area):
    ur = an.urban_rural_trend(df_area)
    assert ((ur["Urban"] - ur["Rural"]) - ur["gap_urban_minus_rural"]).abs().max() < 1e-9
    assert len(ur) == 14


def test_same_month_yoy_has_may_and_june(panel):
    yoy = an.same_month_yoy(panel)
    assert {5, 6} <= set(yoy["month"])


def test_state_ranking_sorted(panel):
    r = an.state_ranking(panel)
    assert r["unemployment_rate"].is_monotonic_decreasing


def test_insights_are_generated(panel, df_area):
    lines = an.generate_insights(panel, df_area)
    assert len(lines) >= 8 and all(isinstance(x, str) and x for x in lines)

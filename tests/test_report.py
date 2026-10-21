from unemployment import report


def test_build_report_writes_files(tmp_path, panel, df_area):
    out = report.build_report(panel, df_area, tmp_path)
    assert len(out["figures"]) == 7 and all(p.exists() and p.stat().st_size > 5_000 for p in out["figures"])
    assert (tmp_path / "insights.md").read_text(encoding="utf-8").startswith("# Key insights")
    assert (tmp_path / "tables" / "covid_impact_by_state.csv").exists()

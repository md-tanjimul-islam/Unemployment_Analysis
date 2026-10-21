import pandas as pd

from unemployment import config, data


def test_state_area_cleaning(df_area):
    assert len(df_area) == 740                      # 28 blank trailing rows removed
    assert df_area.isna().sum().sum() == 0
    assert set(df_area["area"]) == {"Urban", "Rural"}
    assert df_area["date"].min() == pd.Timestamp("2019-05-31")
    assert list(df_area.columns[:7]) == config.STATE_AREA_COLUMNS


def test_state_file_cleaning(df_state):
    assert len(df_state) == 267
    assert df_state["state"].nunique() == 27
    assert df_state["macro_region"].isin(["North", "South", "East", "West", "Northeast"]).all()


def test_swapped_coordinates_are_fixed(df_state):
    assert df_state["lat"].between(6, 37).all()
    assert df_state["lon"].between(68, 98).all()


def test_no_leading_whitespace(df_area, df_state):
    for df in (df_area, df_state):
        for col in df.select_dtypes(include=["object", "string"]).columns:
            assert (df[col] == df[col].str.strip()).all()


def test_panel_shape_and_phases(panel):
    assert panel["state"].nunique() == 28
    assert panel["date"].min() == pd.Timestamp("2019-05-31")
    assert panel["date"].max() == pd.Timestamp("2020-10-31")
    assert panel["phase"].notna().all()
    assert not panel.duplicated(["state", "date"]).any()


def test_overlap_between_files_is_consistent(df_area, df_state):
    overlap = data.validate_overlap(df_area, df_state)
    assert len(overlap) > 100
    assert overlap["abs_diff"].median() < 0.1


def test_weighted_rate_is_between_min_and_max(panel):
    sub = panel[panel["date"] == "2020-04-30"]
    r = data.weighted_rate(sub)
    assert sub["unemployment_rate"].min() <= r <= sub["unemployment_rate"].max()

"""Loading, cleaning and harmonising the two unemployment datasets.

Known data quirks handled here (verified on the supplied CSVs):

* Both files have leading spaces in headers and values and a UTF-8 BOM.
* ``unemployment_in_india.csv`` ends with 28 completely blank rows.
* ``unemployment_rate_upto_11_2020.csv`` has two columns named ``Region``
  (state, then macro-region) and its ``longitude`` / ``latitude`` columns are
  swapped (``longitude`` holds values such as 15.9, which are latitudes).
* The frequency column is ``Monthly`` in one file and ``M`` in the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Raw loaders
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def _add_labour_force(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate labour force = employed / (1 - unemployment_rate)."""
    rate = df["unemployment_rate"] / 100.0
    df["labour_force"] = np.where(rate < 1, df["employed"] / (1 - rate), np.nan)
    return df


def load_state_area(path: Path | str = config.STATE_AREA_FILE) -> pd.DataFrame:
    """Load the May 2019 - Jun 2020 dataset (state x Urban/Rural)."""
    df = _read_csv(Path(path))
    df = df.dropna(how="all")
    if df.shape[1] != len(config.STATE_AREA_COLUMNS):
        raise ValueError(f"Unexpected column count in {path}: {df.shape[1]}")
    df.columns = config.STATE_AREA_COLUMNS
    df = _strip_strings(df)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["frequency"] = "Monthly"
    df["macro_region"] = df["state"].map(_macro_region_lookup()).fillna(
        df["state"].map(config.EXTRA_MACRO_REGIONS)
    )
    df = _add_labour_force(df)
    return df.drop_duplicates().sort_values(["state", "area", "date"]).reset_index(drop=True)


def load_state(path: Path | str = config.STATE_FILE) -> pd.DataFrame:
    """Load the Jan - Oct 2020 dataset (state level, with coordinates)."""
    df = _read_csv(Path(path))
    df = df.dropna(how="all")
    if df.shape[1] != len(config.STATE_COLUMNS):
        raise ValueError(f"Unexpected column count in {path}: {df.shape[1]}")
    df.columns = config.STATE_COLUMNS  # resolves the duplicated 'Region' header
    df = _strip_strings(df)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["frequency"] = "Monthly"
    df = _fix_swapped_coordinates(df)
    df["employed"] = df["employed"].astype(float)
    df = _add_labour_force(df)
    return df.drop_duplicates().sort_values(["state", "date"]).reset_index(drop=True)


def _fix_swapped_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """The source file stores latitudes under 'longitude' and vice versa.

    India spans roughly lat 6-37 and lon 68-98, so we detect the swap from the
    data instead of assuming it, which keeps the loader safe if the file is fixed.
    """
    if df["lon"].between(6, 37).all() and df["lat"].between(68, 98).all():
        df = df.rename(columns={"lon": "lat", "lat": "lon"})
    return df


def _macro_region_lookup() -> dict[str, str]:
    """State -> macro-region map, taken from the second file."""
    df = _strip_strings(_read_csv(config.STATE_FILE).dropna(how="all"))
    df.columns = config.STATE_COLUMNS
    return df.drop_duplicates("state").set_index("state")["macro_region"].to_dict()


# --------------------------------------------------------------------------- #
# Harmonised state-level panel
# --------------------------------------------------------------------------- #
def weighted_rate(df: pd.DataFrame) -> float:
    """Labour-force-weighted unemployment rate for a group of rows."""
    d = df.dropna(subset=["unemployment_rate", "labour_force"])
    if d.empty or d["labour_force"].sum() == 0:
        return float("nan")
    return float(np.average(d["unemployment_rate"], weights=d["labour_force"]))


def collapse_areas(df_area: pd.DataFrame) -> pd.DataFrame:
    """Combine Urban + Rural rows into one state-level row per month."""
    rows = []
    for (state, date), g in df_area.groupby(["state", "date"]):
        lf = g["labour_force"].sum()
        rows.append(
            {
                "state": state,
                "date": date,
                "unemployment_rate": weighted_rate(g),
                "employed": g["employed"].sum(),
                "labour_force": lf,
                "labour_participation_rate": np.average(
                    g["labour_participation_rate"], weights=g["employed"]
                ),
                "macro_region": g["macro_region"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def build_panel(df_area: pd.DataFrame | None = None, df_state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one continuous state-level monthly series, May 2019 - Oct 2020.

    The two files overlap for Jan-Jun 2020. Where both exist the state-level
    file is used (it is the more complete source); the Urban/Rural file fills
    in 2019. On the overlap the two files agree closely (see ``validate_overlap``).
    """
    df_area = load_state_area() if df_area is None else df_area
    df_state = load_state() if df_state is None else df_state

    early = collapse_areas(df_area)
    early = early[early["date"] < df_state["date"].min()]
    late = df_state[
        ["state", "date", "unemployment_rate", "employed", "labour_force",
         "labour_participation_rate", "macro_region"]
    ]
    panel = pd.concat([early, late], ignore_index=True)
    panel["macro_region"] = panel["macro_region"].fillna(panel["state"].map(config.EXTRA_MACRO_REGIONS))
    panel = add_phase(panel)
    return panel.sort_values(["state", "date"]).reset_index(drop=True)


def add_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with its COVID-19 phase (categorical, ordered)."""
    df = df.copy()
    phase = pd.Series(index=df.index, dtype="object")
    for name, (start, end) in config.PHASE_BOUNDS.items():
        phase[(df["date"] >= start) & (df["date"] <= end)] = name
    df["phase"] = pd.Categorical(phase, categories=config.PHASE_ORDER, ordered=True)
    return df


def validate_overlap(df_area: pd.DataFrame, df_state: pd.DataFrame) -> pd.DataFrame:
    """Compare the two files on their overlapping state-months (data-quality check)."""
    a = collapse_areas(df_area)[["state", "date", "unemployment_rate"]]
    b = df_state[["state", "date", "unemployment_rate"]]
    m = a.merge(b, on=["state", "date"], suffixes=("_area_file", "_state_file"))
    m["abs_diff"] = (m["unemployment_rate_area_file"] - m["unemployment_rate_state_file"]).abs()
    return m


def data_quality_report(df_area: pd.DataFrame, df_state: pd.DataFrame) -> dict:
    """Small summary of the cleaned data, used in the README/report and dashboard."""
    overlap = validate_overlap(df_area, df_state)
    return {
        "state_area_rows": len(df_area),
        "state_area_period": (df_area["date"].min().date(), df_area["date"].max().date()),
        "state_rows": len(df_state),
        "state_period": (df_state["date"].min().date(), df_state["date"].max().date()),
        "states_in_area_file": df_area["state"].nunique(),
        "states_in_state_file": df_state["state"].nunique(),
        "overlap_rows": len(overlap),
        "overlap_median_abs_diff_pp": float(overlap["abs_diff"].median()),
        "overlap_share_within_0_5pp": float((overlap["abs_diff"] <= 0.5).mean()),
        "missing_values": int(df_area.isna().sum().sum() + df_state.isna().sum().sum()),
    }

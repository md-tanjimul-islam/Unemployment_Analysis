"""Project-wide constants: paths, column names, and COVID-19 phase definitions."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

STATE_AREA_FILE = RAW_DIR / "unemployment_in_india.csv"  # May 2019 - Jun 2020, Urban/Rural
STATE_FILE = RAW_DIR / "unemployment_rate_upto_11_2020.csv"  # Jan 2020 - Oct 2020, state level

# The two raw files use different (and messy) headers; both are mapped to this schema.
STATE_AREA_COLUMNS = [
    "state", "date", "frequency", "unemployment_rate",
    "employed", "labour_participation_rate", "area",
]
STATE_COLUMNS = [
    "state", "date", "frequency", "unemployment_rate",
    "employed", "labour_participation_rate", "macro_region", "lon", "lat",
]

# India's national lockdown began on 25 March 2020 and "Unlock 1.0" began on 1 June 2020.
# Observations are month-end, so March 2020 is a partial-lockdown month.
PHASE_ORDER = ["Pre-COVID", "Onset", "Lockdown", "Recovery"]
PHASE_BOUNDS = {
    "Pre-COVID": (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-02-29")),
    "Onset": (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-31")),
    "Lockdown": (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-31")),
    "Recovery": (pd.Timestamp("2020-06-01"), pd.Timestamp("2099-12-31")),
}
PHASE_LABELS = {
    "Pre-COVID": "Pre-COVID (before Mar 2020)",
    "Onset": "Onset (Mar 2020)",
    "Lockdown": "Lockdown (Apr-May 2020)",
    "Recovery": "Recovery (Jun-Oct 2020)",
}

LOCKDOWN_START = pd.Timestamp("2020-03-25")
LOCKDOWN_END = pd.Timestamp("2020-05-31")

# Chandigarh appears only in the first file, which has no macro-region column.
EXTRA_MACRO_REGIONS = {"Chandigarh": "North"}

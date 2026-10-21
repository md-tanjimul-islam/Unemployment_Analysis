import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unemployment import data  # noqa: E402


@pytest.fixture(scope="session")
def df_area():
    return data.load_state_area()


@pytest.fixture(scope="session")
def df_state():
    return data.load_state()


@pytest.fixture(scope="session")
def panel(df_area, df_state):
    return data.build_panel(df_area, df_state)

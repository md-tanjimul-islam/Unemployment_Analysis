"""Command-line entry point: run the full analysis and write figures/tables/insights.

Usage:
    python run_analysis.py                 # writes to ./reports
    python run_analysis.py --out my_dir    # custom output folder
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from unemployment import data, report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="India unemployment analysis (CodeAlpha Task 2)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: ./reports)")
    args = parser.parse_args()

    print("Loading and cleaning data...")
    df_area, df_state = data.load_state_area(), data.load_state()
    panel = data.build_panel(df_area, df_state)

    quality = data.data_quality_report(df_area, df_state)
    print("Data quality:")
    for key, value in quality.items():
        print(f"  {key}: {value}")

    print("Building report...")
    result = report.build_report(panel, df_area, args.out)

    print(f"\nWrote {len(result['figures'])} figures and {len(result['tables'])} tables.\n")
    print("Key insights")
    print("------------")
    for line in result["insights"]:
        print(f"* {line}")


if __name__ == "__main__":
    main()

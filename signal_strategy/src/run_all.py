"""End-to-end runner for the sentiment->returns signal study.

Runs the five steps in order. Step 1 (build_panel) is the slow one (~6 min, parses
all enriched inference + news archives); the rest are seconds. Pass --skip-panel to
reuse an existing panel.parquet.

    python3 signal_strategy/src/run_all.py            # full pipeline
    python3 signal_strategy/src/run_all.py --skip-panel
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("build_panel.py", "Step 1 - build point-in-time panel"),
    ("ic_analysis.py", "Step 2 - information coefficient"),
    ("fama_macbeth.py", "Step 3 - Fama-MacBeth"),
    ("event_study.py", "Step 4 - event study"),
    ("backtest_ls.py", "Step 5 - long-short backtest + baseline"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-panel", action="store_true",
                    help="reuse existing panel.parquet (skip the slow Step 1)")
    args = ap.parse_args()

    for script, desc in STEPS:
        if args.skip_panel and script == "build_panel.py":
            print(f"== SKIP {desc} ==")
            continue
        print(f"\n{'=' * 70}\n== {desc}\n{'=' * 70}")
        r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
        if r.returncode != 0:
            print(f"!! {script} failed (exit {r.returncode}) — stopping.")
            sys.exit(r.returncode)
    print("\nAll steps complete. See signal_strategy/RESULTS.md and "
          "signal_strategy/outputs/.")


if __name__ == "__main__":
    main()

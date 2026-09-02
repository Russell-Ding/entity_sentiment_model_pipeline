#!/usr/bin/env python3
"""Tabulate retrain ablation arms side-by-side.

Reads each arm's gap_analysis.json (produced by analyze_eval_gaps.py) under
outputs/retrain_<arm>/, plus the current-model baseline
(outputs/eval_new_dataset/gap_analysis.json) and the Phase-0 calibration floor
(outputs/eval_new_dataset/calibration_baseline.json), and emits a side-by-side
markdown + console table on the metrics that decide the retrain.

Reading guide:
  - Pearson/Spearman/CCC up vs the 0.46-0.49 baseline = real improvement.
  - std_ratio -> ~1.0 and very_neg/very_pos F1 > 0 = under-polarization fixed,
    BUT only meaningful if sign_flip / over_neutral did NOT worsen (Codex caveat).
  - Compare armC (encoder unfrozen) vs control (head-only): if armC >> control,
    the ceiling was representational. If similar, the head/loss was the lever.

Usage:
  python3 scripts/evaluation/compare_arms.py
  python3 scripts/evaluation/compare_arms.py --arms armA control armC armD armE
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUTROOT = PROJECT / "outputs"


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row_from_gap(label, j):
    o = j.get("overall", {})
    eb = j.get("extreme_bucket_f1", {})
    return {
        "source": label,
        "N": o.get("n"),
        "pearson": o.get("pearson"),
        "spearman": o.get("spearman"),
        "ccc": o.get("ccc"),
        "std_ratio": o.get("std_ratio"),
        "mae": o.get("mae"),
        "over_neutral": g(j, "over_neutrality", "rate_among_strong"),
        "sign_flip": g(j, "sign_flip", "rate_among_both_signed"),
        "vneg_f1": g(eb, "very_neg", "f1"),
        "vpos_f1": g(eb, "very_pos", "f1"),
        "org_r": next((b["pearson"] for b in j.get("per_type", []) if b["label"] == "ORG"), None),
        "ticker_r": next((b["pearson"] for b in j.get("per_type", []) if b["label"] == "TICKER"), None),
        "person_r": next((b["pearson"] for b in j.get("per_type", []) if b["label"] == "PERSON"), None),
    }


def row_from_calib(label, j):
    r = j.get("affine_calibrated", j.get("raw", {}))
    def pf(s):  # parse "P/R/F1"
        try: return float(str(s).split("/")[2])
        except Exception: return None
    return {
        "source": label, "N": j.get("eval_n"),
        "pearson": r.get("pearson"), "spearman": r.get("spearman"), "ccc": None,
        "std_ratio": r.get("std_ratio"), "mae": r.get("mae"),
        "over_neutral": r.get("over_neutral_rate"), "sign_flip": r.get("sign_flip_rate"),
        "vneg_f1": pf(r.get("very_neg_P/R/F1")), "vpos_f1": pf(r.get("very_pos_P/R/F1")),
        "org_r": None, "ticker_r": None, "person_r": None,
    }


def fnum(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[],
                    help="optional substring filter on run labels; default = all discovered runs")
    ap.add_argument("--output", default=str(OUTROOT / "retrain_comparison.md"))
    args = ap.parse_args()

    rows = []
    # current-model baseline (full new set)
    base = OUTROOT / "eval_new_dataset" / "gap_analysis.json"
    if base.exists():
        rows.append(row_from_gap("v2.0 current (full set)", json.load(open(base))))
    # Phase-0 calibration floor (affine-calibrated current model)
    calib = OUTROOT / "eval_new_dataset" / "calibration_baseline.json"
    if calib.exists():
        rows.append(row_from_calib("v2.0 + affine calib (floor)", json.load(open(calib))))
    # discover EVERY run dir (outputs/retrain_<label>/gap_analysis.json) — handles
    # both untagged (retrain_control) and tagged (retrain_armC__20260620_2230) runs,
    # so no run is hidden and re-runs show as separate rows.
    for gj in sorted(glob.glob(str(OUTROOT / "retrain_*" / "gap_analysis.json"))):
        label = Path(gj).parent.name[len("retrain_"):]
        if args.arms and not any(a in label for a in args.arms):
            continue
        rows.append(row_from_gap(f"{label} (test)", json.load(open(gj))))

    if not rows:
        raise SystemExit("No gap_analysis.json found. Run the eval/ablation first.")

    cols = [("source", "source", 0), ("N", "N", 0), ("pearson", "Pearson", 3),
            ("spearman", "Spearman", 3), ("ccc", "CCC", 3), ("std_ratio", "std_ratio", 3),
            ("mae", "MAE", 3), ("over_neutral", "over_neut", 3), ("sign_flip", "sign_flip", 3),
            ("vneg_f1", "vneg_F1", 2), ("vpos_f1", "vpos_F1", 2),
            ("org_r", "ORG_r", 3), ("ticker_r", "TICKER_r", 3), ("person_r", "PERSON_r", 3)]
    header = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = ["# Retrain ablation — side-by-side", "",
             "Baseline = current v2.0; floor = + affine calibration (Phase 0). Arms eval on "
             "the held-out **test** split. Improvement = Pearson/CCC up; under-polarization "
             "fixed = std_ratio→1 & extreme F1>0 **without** worse sign_flip/over_neutral.", "",
             header, sep]
    for r in rows:
        lines.append("| " + " | ".join(fnum(r.get(k), nd) for k, _, nd in cols) + " |")
    md = "\n".join(lines) + "\n"
    Path(args.output).write_text(md)
    print(md)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

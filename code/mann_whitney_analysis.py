#!/usr/bin/env python3
"""
Mann-Whitney U analysis on replaced dataset.
Unit of analysis : PitchID
Variables        : Q1–Q15 (raw grid items) + Investment %
Comparisons      : Group1 vs Group2 / Group3 / Group4  (one-sided, GroupX > Group1)
Correction       : Bonferroni (α=0.05/3) + Holm per variable set
Output           : mann_whitney_Q1_Q15_invest.csv

Usage
-----
    python mann_whitney_analysis.py
    python mann_whitney_analysis.py --csv path/to/merged.csv
                                    --out path/to/results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────

LIKERT_MAP = {
    "highly ineffective":   1,
    "somewhat ineffective": 2,
    "somewhat effective":   3,
    "highly effective":     4,
}

GRID1_COLS = [f"Grid1_{i}" for i in range(1, 6)]
GRID2_COLS = [f"Grid2_{i}" for i in range(1, 6)]
GRID3_COLS = [f"Grid3_{i}" for i in range(1, 6)]
ALL_GRID   = GRID1_COLS + GRID2_COLS + GRID3_COLS

GRID_NAMES = [
    "Environmental sustainability",
    "Social sustainability",
    "Economic sustainability",
]
Q_NAMES = ["Credibility", "Emotional appeal", "Logical reasoning", "Timing", "Clarity"]

BASELINE    = "Group1"
COMP_GROUPS = ["Group2", "Group3", "Group4"]
ALPHA       = 0.05

# ── Helpers ───────────────────────────────────────────────────────────────────

def norm_label(x: str) -> str:
    x = str(x).strip().lower()
    if x.endswith("."):
        x = x[:-1]
    return x


def build_variable_list() -> list[dict]:
    """Return Q1–Q15 then Invest — no averages."""
    variables = []
    q_num = 1
    for gi, gname in enumerate(GRID_NAMES, 1):
        for qi, qname in enumerate(Q_NAMES, 1):
            variables.append({
                "col":       f"Grid{gi}_{qi}",
                "item":      f"Q{q_num}",
                "grid":      f"Grid{gi}",
                "dimension": gname,
                "question":  qname,
            })
            q_num += 1
    variables.append({
        "col":       "Invest_num",
        "item":      "Invest",
        "grid":      "—",
        "dimension": "Investment",
        "question":  "Investment %",
    })
    return variables


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data/experiment_survey/mergedFinal.csv"),
                        help="Input CSV (default: mergedFinal.csv)")
    parser.add_argument("--out", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "results/mann_whitney_Q1_Q15_invest.csv"),
                        help="Output CSV (default: mann_whitney_Q1_Q15_invest.csv)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] File not found: {csv_path}")

    # ── 1. Load & convert ────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} rows from '{csv_path}'")

    for c in ALL_GRID:
        df[c + "_n"] = df[c].map(lambda x: LIKERT_MAP.get(norm_label(x), np.nan))

    df["Invest_num"] = (
        df["Invest"].astype(str).str.strip().str.rstrip("%")
        .replace("", np.nan).astype(float)
    )

    # ── 2. Aggregate per (PitchID, Group) ────────────────────────────────────
    agg_spec = {"Invest_num": ("Invest_num", "mean")}
    for c in ALL_GRID:
        agg_spec[c] = (c + "_n", "mean")

    agg = df.groupby(["PitchID", "Group"]).agg(**agg_spec).reset_index()
    counts = agg["Group"].value_counts().sort_index().to_dict()
    print(f"[INFO] Aggregated to PitchID level: {agg.shape} | n per group: {counts}")

    g1      = agg[agg["Group"] == BASELINE]
    bonf_a  = ALPHA / len(COMP_GROUPS)
    print(f"[INFO] Bonferroni threshold: {ALPHA}/{len(COMP_GROUPS)} = {bonf_a:.4f}\n")

    # ── 3. Run tests ─────────────────────────────────────────────────────────
    variables = build_variable_list()
    rows = []

    for v in variables:
        var    = v["col"]
        p_list, temp = [], []

        for gx_name in COMP_GROUPS:
            gx      = agg[agg["Group"] == gx_name]
            g1_vals = g1[var].values
            gx_vals = gx[var].values
            u, p    = stats.mannwhitneyu(gx_vals, g1_vals, alternative="greater")
            p_list.append(p)
            temp.append((gx_name, u, p, g1_vals, gx_vals))

        reject_holm, p_holm, _, _ = multipletests(p_list, alpha=ALPHA, method="holm")

        for i, (gx_name, u, p, g1_vals, gx_vals) in enumerate(temp):
            rows.append({
                "Item":          v["item"],
                "Grid":          v["grid"],
                "Dimension":     v["dimension"],
                "Question":      v["question"],
                "Comparison":    f"Group1 vs {gx_name}",
                "n (Group1)":    len(g1_vals),
                "n (GroupX)":    len(gx_vals),
                "Median (G1)":   round(float(np.median(g1_vals)), 4),
                "Median (Gx)":   round(float(np.median(gx_vals)), 4),
                "Mean (G1)":     round(float(np.mean(g1_vals)),   4),
                "Mean (Gx)":     round(float(np.mean(gx_vals)),   4),
                "U-stat":        round(u, 1),
                "p (one-sided)": round(p, 5),
                "Bonf sig":      "✓" if p < bonf_a else "✗",
                "p (Holm-adj)":  round(p_holm[i], 5),
                "Holm sig":      "✓" if reject_holm[i] else "✗",
                "H1":            f"{gx_name} > Group1",
            })

    results = pd.DataFrame(rows)

    # ── 4. Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    results.to_csv(out_path, index=False)
    print(f"[✓] Saved {len(results)} rows to '{out_path}'")
    print(f"    Items: {results['Item'].nunique()} unique  "
          f"(Q1–Q15 + Invest = 16 × 3 comparisons = 48 rows)")

    # ── 5. Print significant results ─────────────────────────────────────────
    sig = results[results["Holm sig"] == "✓"]
    if len(sig):
        print(f"\n[✓] Significant after Holm correction ({len(sig)} rows):")
        print(sig[["Item", "Dimension", "Question", "Comparison",
                    "Median (G1)", "Median (Gx)", "p (one-sided)",
                    "p (Holm-adj)"]].to_string(index=False))
    else:
        print("\n[!] No results significant after Holm correction.")


if __name__ == "__main__":
    main()

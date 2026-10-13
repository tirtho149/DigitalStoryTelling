#!/usr/bin/env python3
"""
make_raw_scores.py
==================================================================
Split the processed experiment survey (mergedFinal.csv) into one
clean RAW-SCORES file per condition (Control, Treatment A, B, C).

Each output row = one participant's rating of one borrower pitch:
  - the 15 perception items (4-point Likert, encoded 1-4)
  - the investment allocation (%)

Likert encoding (as used throughout the paper):
  1 = Highly Ineffective
  2 = Somewhat Ineffective
  3 = Somewhat Effective
  4 = Highly Effective

Grid -> sustainability dimension:  Grid1 = Environmental,
                                   Grid2 = Social,
                                   Grid3 = Economic
Column suffix -> criterion:        _1 Credibility, _2 Emotional Appeal,
                                   _3 Logical Reasoning, _4 Timing, _5 Clarity

Usage:
  python make_raw_scores.py \
      --csv ../data/experiment_survey/mergedFinal.csv \
      --out ../data/experiment_survey/raw_scores_by_condition
"""
import argparse
from pathlib import Path
import pandas as pd

GROUP_TO_CONDITION = {
    "Group1": "control",
    "Group2": "treatmentA",
    "Group3": "treatmentB",
    "Group4": "treatmentC",
}

LIKERT = {
    "Highly Ineffective": 1,
    "Somewhat Ineffective": 2,
    "Somewhat Effective": 3,
    "Highly Effective": 4,
    "Highly Effective.": 4,   # some exports carry a trailing period
}

# (raw column, human-readable Q label)
ITEM_LABELS = {
    "Grid1_1": "Q1_Credibility_Environmental",
    "Grid1_2": "Q2_EmotionalAppeal_Environmental",
    "Grid1_3": "Q3_LogicalReasoning_Environmental",
    "Grid1_4": "Q4_Timing_Environmental",
    "Grid1_5": "Q5_Clarity_Environmental",
    "Grid2_1": "Q6_Credibility_Social",
    "Grid2_2": "Q7_EmotionalAppeal_Social",
    "Grid2_3": "Q8_LogicalReasoning_Social",
    "Grid2_4": "Q9_Timing_Social",
    "Grid2_5": "Q10_Clarity_Social",
    "Grid3_1": "Q11_Credibility_Economic",
    "Grid3_2": "Q12_EmotionalAppeal_Economic",
    "Grid3_3": "Q13_LogicalReasoning_Economic",
    "Grid3_4": "Q14_Timing_Economic",
    "Grid3_5": "Q15_Clarity_Economic",
}


def encode_likert(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().map(LIKERT)


def parse_invest(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("%", "", regex=False).str.strip().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/experiment_survey/mergedFinal.csv")
    ap.add_argument("--out", default="../data/experiment_survey/raw_scores_by_condition")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {args.csv}  ({df.shape[0]} rows, {df.shape[1]} cols)\n")

    summary = []
    for group, cond in GROUP_TO_CONDITION.items():
        g = df[df["Group"] == group].copy()

        clean = pd.DataFrame()
        clean["ResponseId"] = g["ResponseId"]
        clean["PitchID"] = g["PitchID"]
        for raw_col, label in ITEM_LABELS.items():
            clean[label] = encode_likert(g[raw_col])
        clean["Investment_pct"] = parse_invest(g["Invest"])

        fpath = out / f"{cond}_{group.lower()}_raw_scores.csv"
        clean.to_csv(fpath, index=False)

        n_part = g["ResponseId"].nunique()
        mean_inv = clean["Investment_pct"].mean()
        pct_ge75 = (clean["Investment_pct"] >= 75).mean() * 100
        pct_100 = (clean["Investment_pct"] == 100).mean() * 100
        summary.append(
            dict(condition=cond, group=group, participants=n_part, rows=len(clean),
                 mean_invest=round(mean_inv, 2),
                 pct_ge75=round(pct_ge75, 1), pct_100=round(pct_100, 1))
        )
        print(f"  {cond:11s} ({group}) -> {fpath.name:42s} "
              f"n_part={n_part:3d}  rows={len(clean):3d}  "
              f"mean_invest={mean_inv:5.1f}%  >=75%={pct_ge75:4.1f}%  100%={pct_100:4.1f}%")

    pd.DataFrame(summary).to_csv(out / "_condition_summary.csv", index=False)
    print(f"\nWrote per-condition raw scores + _condition_summary.csv -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
study3_hypothesis_tests.py  —  Study 3 (Spring 2026)
==================================================================
Reproduces the paper's inferential framework.

SECTION 2.5  (paper Table 2) — Modality effect
  One-sided Mann-Whitney U tests on all 16 outcome variables
  (15 perception items + investment allocation) across 3 contrasts:
      Control vs Treatment A
      Control vs Treatment B
      Treatment A vs Treatment B
  -> 16 x 3 = 48 comparisons forming ONE Holm-Bonferroni family.

  H0: the treatment/later condition does NOT score higher than the
      control/earlier condition.
  H1: it scores higher (one-sided).  Directions:
      A > Control,   B > Control,   B > A.

SECTION 2.8  (paper Table 3) — Enhancement effect
  The same 16 outcomes for Treatment A vs Treatment C (C > A),
  a SEPARATE 16-comparison Holm-Bonferroni family.

Input  : the Study 3 survey CSV (default: the analysis-ready
         data/experiment_survey/study3_spring2026_survey.csv; the raw
         data/experiment_survey/mergedFinal.csv is also accepted).
Outputs: results/study3_table2_modality_holm.csv   (Section 2.5)
         results/study3_table3_enhancement_holm.csv (Section 2.8)

Usage (run from code/):
    python3 study3_hypothesis_tests.py
    python3 study3_hypothesis_tests.py --csv /path/to/survey.csv --out ../results
"""
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parent.parent

GROUP = {"Group1": "Control", "Group2": "Treatment A",
         "Group3": "Treatment B", "Group4": "Treatment C"}

LIKERT = {"Highly Ineffective": 1, "Somewhat Ineffective": 2,
          "Somewhat Effective": 3, "Highly Effective": 4, "Highly Effective.": 4}

# 15 perception items -> (raw mergedFinal column, dimension, criterion, Q#)
ITEMS = [
    ("Grid1_1", "Environmental", "Credibility",       "Q1"),
    ("Grid1_2", "Environmental", "Emotional Appeal",  "Q2"),
    ("Grid1_3", "Environmental", "Logical Reasoning", "Q3"),
    ("Grid1_4", "Environmental", "Timing",            "Q4"),
    ("Grid1_5", "Environmental", "Clarity",           "Q5"),
    ("Grid2_1", "Social",        "Credibility",       "Q6"),
    ("Grid2_2", "Social",        "Emotional Appeal",  "Q7"),
    ("Grid2_3", "Social",        "Logical Reasoning", "Q8"),
    ("Grid2_4", "Social",        "Timing",            "Q9"),
    ("Grid2_5", "Social",        "Clarity",           "Q10"),
    ("Grid3_1", "Economic",      "Credibility",       "Q11"),
    ("Grid3_2", "Economic",      "Emotional Appeal",  "Q12"),
    ("Grid3_3", "Economic",      "Logical Reasoning", "Q13"),
    ("Grid3_4", "Economic",      "Timing",            "Q14"),
    ("Grid3_5", "Economic",      "Clarity",           "Q15"),
]


def load(csv_path: Path) -> pd.DataFrame:
    """Return a tidy frame with a 'Condition' column, numeric Q1..Q15 and Investment_pct."""
    df = pd.read_csv(csv_path)

    # Analysis-ready file already has Condition + numeric Qn + Investment_pct.
    if "Investment_pct" in df.columns and "Q1_Credibility_Environmental" in df.columns:
        ren = {f"{q}_{cri.replace(' ', '')}_{dim}": q for (_, dim, cri, q) in ITEMS}
        df = df.rename(columns=ren)
        return df

    # Otherwise treat it as raw mergedFinal.csv and convert.
    df["Condition"] = df["Group"].map(GROUP)
    for raw, _, _, q in ITEMS:
        df[q] = df[raw].astype(str).str.strip().map(LIKERT)
    df["Investment_pct"] = df["Invest"].astype(str).str.replace("%", "", regex=False).astype(float)
    return df


def run_family(df, contrasts, outcomes_meta):
    """contrasts: list of (lo_condition, hi_condition) with H1 = hi > lo."""
    rows = []
    for lo, hi in contrasts:
        for col, dim, crit, q in outcomes_meta:
            x = df.loc[df.Condition == lo, col].dropna()
            y = df.loc[df.Condition == hi, col].dropna()
            U = mannwhitneyu(y, x, alternative="greater")
            rows.append(dict(
                outcome=q, dimension=dim, criterion=crit,
                contrast=f"{lo} vs {hi}", H1=f"{hi} > {lo}",
                n_lo=len(x), n_hi=len(y),
                median_lo=x.median(), median_hi=y.median(),
                U=U.statistic, p_one_sided=U.pvalue))
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_one_sided"].values, method="holm")[1]
    out["significant_holm_.05"] = out["p_holm"] < 0.05
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(REPO / "data/experiment_survey/study3_spring2026_survey.csv"))
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():                       # fall back to raw merged file
        csv = REPO / "data/experiment_survey/mergedFinal.csv"
    df = load(csv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {csv}  ({len(df)} rows)")

    outcomes = [(q, dim, crit, q) for (_, dim, crit, q) in ITEMS] + \
               [("Investment_pct", "—", "Investment Allocation %", "Invest")]

    # ---- Section 2.5 : modality (48-comparison Holm family) ----
    t2 = run_family(df, [("Control", "Treatment A"),
                         ("Control", "Treatment B"),
                         ("Treatment A", "Treatment B")], outcomes)
    f2 = out / "study3_table2_modality_holm.csv"
    t2.to_csv(f2, index=False)

    # ---- Section 2.8 : enhancement (separate 16-comparison Holm family) ----
    t3 = run_family(df, [("Treatment A", "Treatment C")], outcomes)
    f3 = out / "study3_table3_enhancement_holm.csv"
    t3.to_csv(f3, index=False)

    print(f"\nSection 2.5 (Table 2) -> {f2}   [{len(t2)} comparisons, one Holm family]")
    print(f"Section 2.8 (Table 3) -> {f3}   [{len(t3)} comparisons, separate Holm family]")
    print("\nSpot-check vs paper Table 2 (raw one-sided p):")
    for q in ["Q1", "Q7", "Invest"]:
        r = t2[t2.outcome == q].set_index("contrast")["p_one_sided"]
        print(f"  {q:6s}  CtrlA={r['Control vs Treatment A']:.4f}  "
              f"CtrlB={r['Control vs Treatment B']:.4f}  "
              f"AvB={r['Treatment A vs Treatment B']:.4f}")


if __name__ == "__main__":
    main()

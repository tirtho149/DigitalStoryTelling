# DigitalStoryTelling

Reproduction package for the paper
**"How multimodal gen AI transforms investment decisions – evidence from storytelling through crowdfunding"**
(working title: *How Does Digital Storytelling Affect Crowdfunding*).

The paper studies how AI-generated video pitches change how investors evaluate
Kiva microfinance borrowers. Marketing students rated borrower profiles under
four modality conditions and chose how much to invest. This repository contains
**the raw survey scores, the analysis code, and the exact commands** that
regenerate every figure and statistic in the paper.

See [Quick start](#quick-start) to reproduce the paper's numbers.

> **Running your own analysis?** The main full study dataset is
> **`data/experiment_survey/mergedFinal.csv`** — 1088 rows × 29 columns, one row
> per participant×pitch rating, covering all four conditions (Group1–Group4).
> A demographics-annotated copy is in `data/experiment_survey/mergedFinal_demo.csv`.
> Start there for any new analysis.

---

## Conditions (the four surveys)

| Condition | Group code | What the participant saw |
|-----------|-----------|--------------------------|
| **Control**     | `Group1` | static photo + written narrative (current Kiva interface) |
| **Treatment A** | `Group2` | photo + narrative **+** AI-generated 12 s video (neutral prompt) |
| **Treatment B** | `Group3` | narrative + AI-generated video **replacing** the photo |
| **Treatment C** | `Group4` | photo + narrative + AI video from a **prompt-engineered** script |

Each participant rated pitches on a **15-item perception grid** (3 sustainability
dimensions × 5 rhetorical criteria, 4-point Likert) and chose an investment
allocation of **25 / 50 / 75 / 100 %**.

| | Credibility | Emotional Appeal | Logical Reasoning | Timing | Clarity |
|--|--|--|--|--|--|
| **Environmental** | Q1 | Q2 | Q3 | Q4 | Q5 |
| **Social**        | Q6 | Q7 | Q8 | Q9 | Q10 |
| **Economic**      | Q11 | Q12 | Q13 | Q14 | Q15 |

---

## Repository layout

```
DigitalStoryTelling/
├── README.md                 ← you are here
├── requirements.txt
├── run_all.sh                ← one command to reproduce everything (except the heavy RF)
├── paper/
│   └── How Does Digital Storytelling Affect Crowdfunding.pdf
├── data/
│   ├── experiment_survey/
│   │   ├── mergedFinal.csv            ← MAIN raw survey (1088 rows, 4 conditions)
│   │   ├── mergedFinal_demo.csv       ← same + demographics
│   │   └── raw_scores_by_condition/   ← one clean RAW-SCORES file per survey
│   │        ├── control_group1_raw_scores.csv
│   │        ├── treatmentA_group2_raw_scores.csv
│   │        ├── treatmentB_group3_raw_scores.csv
│   │        ├── treatmentC_group4_raw_scores.csv
│   │        └── _condition_summary.csv
│   ├── random_forest/
│   │   ├── Result - GroundTruth_numeric.csv     ← human perception scores
│   │   └── Result - predictionLLama_numeric.csv ← LLM-predicted scores (used by Fig 4)
│   ├── country_ranking_survey/study2.csv        ← supplementary cross-country ranking
│   └── video_ab_survey/{study3.csv, study3Final.csv}  ← supplementary A/B video preference
├── code/
│   ├── make_raw_scores.py      ← split mergedFinal.csv → per-condition raw scores
│   ├── mann_whitney_analysis.py← Mann–Whitney U tests  (Table 2 & 3)
│   ├── generate_figures.py     ← all experiment figures (Fig 2,3,5,6,7 + MWU xlsx)
│   ├── random_forest.py        ← funding-duration Random Forest (Fig 4) — HEAVY, use Slurm
│   ├── country_ranking.py      ← OLS/OLR cross-country ranking (supplementary)
│   └── video_ab.py             ← A/B video-preference charts (supplementary)
├── figures/                    ← reference copies of generated figures
│   ├── experiment/   (fig01…fig09 PNG+SVG, mwu_results.xlsx)
│   ├── random_forest/
│   ├── country_ranking/
│   └── video_ab/
└── results/                    ← reference copies of generated stat tables
    ├── mann_whitney_Q1_Q15_invest.csv
    ├── mann_whitney_report.csv
    ├── rf_groundtruth_*_FINAL.csv
    └── rf_predictionllama_*_FINAL.csv
```

---

## Quick start

```bash
# 1. install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2. reproduce everything except the heavy Random Forest
bash run_all.sh
```

Outputs land in `figures/` and `results/`. That's it.

---

## Which command produces which file

Run all commands **from the `code/` directory**. Paths are relative to `code/`.

### Study 1 — The experiment (Control / A / B / C)  → paper Fig 2, 3, 5, 6, 7; Table 2, 3

| Command | Produces | Paper element |
|---------|----------|---------------|
| `python3 make_raw_scores.py` | `data/experiment_survey/raw_scores_by_condition/*.csv` | the raw scores for each survey |
| `python3 mann_whitney_analysis.py` | `results/mann_whitney_Q1_Q15_invest.csv` | **Table 2** (Ctrl vs A/B) & **Table 3** (A vs C) p-values |
| `python3 generate_figures.py` | `figures/experiment/fig01_delta_heatmap_AB.*` | **Fig 2** (perception Δ, Ctrl vs A/B) |
|  | `figures/experiment/fig02_investment_AB.*` | **Fig 3** (investment allocation, Ctrl/A/B) |
|  | `figures/experiment/fig05_delta_heatmap_C.*` | **Fig 5** (perception Δ, Ctrl vs C) |
|  | `figures/experiment/fig06_investment_C.*` | **Fig 6** (investment allocation, Ctrl vs C) |
|  | `figures/experiment/fig07_mwu_A_vs_C.*` | MWU heatmap for **Table 3** |
|  | `figures/experiment/fig03,04,08,09.*`, `mwu_results.xlsx` | supporting panels |

Both scripts accept `--csv PATH` and `--out PATH` to point at other data/output
locations (defaults already point inside this repo).

### Study 2 — Random Forest funding-duration prediction  → paper Fig 4

| Command | Produces | Paper element |
|---------|----------|---------------|
| `python3 random_forest.py` | `results/random_forest_run/PredictionLlama_*` + `figures/random_forest/PredictionLlama_r2_distribution.png`, `PredictionLlama_feature_importance_overall.png` | **Fig 4** (R² distribution + Gini importances) |
|  | `results/random_forest_run/GroundTruth_*` | human-rating counterpart |

⚠️ **This is heavy**: 1000 bootstrap iterations × 2 datasets with a
`RandomizedSearchCV`-tuned Random Forest ≈ **9 hours on CPU**. Do **not** run it
on a login node — submit it to Slurm. Pre-computed reference results are already
in `results/rf_*_FINAL.csv` and `figures/random_forest/`.

### Supplementary surveys

| Command | Produces |
|---------|----------|
| `python3 country_ranking.py` | `results/country_ranking_run/` — OLS/OLR cross-country feature-importance ranking + figures |
| `python3 video_ab.py` | `figures/video_ab/{country,sector,feature}_percentage.png` — enhanced-vs-barebone video preference |

---

## Verified reproductions (data → paper)

These were checked digit-for-digit against the PDF:

| Paper claim | Reproduced from | Value |
|-------------|-----------------|-------|
| Investment ≥75 %: Control / A / B / C | `_condition_summary.csv` | 37.6 / 49.6 / 64.2 / 67.8 % ✓ |
| Investment =100 %: Control / A / B / C | `_condition_summary.csv` | 10.1 / 23.5 / 31.9 / 47.6 % ✓ (paper rounds to 10/23/32/48) |
| MWU Invest p: Ctrl vs A / B / C | `mann_whitney_Q1_Q15_invest.csv` | 0.00004 / 0.00000 / 0.00000 ✓ |
| Q7 Emotional-Appeal-Social significant (Ctrl vs C) | same | p=0.00561, Holm 0.0168 ✓ |
| Fig 5 Δ Logical Reasoning (Env/Soc/Econ) | per-condition means | −0.29 / −0.13 / −0.32 ✓ |
| Random Forest mean R² | `rf_predictionllama_r2_FINAL.csv` | 0.7615 (paper: 0.762) ✓ |

---

## Notes & limitations

- The **raw Qualtrics per-group exports** and the script that merged them into
  `mergedFinal.csv` are not included (they were not in the source project). Every
  analysis **downstream of `mergedFinal.csv`** is fully reproducible from this repo.
- The ~37 GB of image/video stimuli and the Kiva API scrapers are **excluded** —
  they are not needed to reproduce the reported numbers and are covered by
  `.gitignore`.
- All scripts were de-Mac-ified: original hardcoded `/Users/tirthoroy/...` paths
  were replaced with repo-relative paths.

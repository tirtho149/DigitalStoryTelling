#!/usr/bin/env python3
"""
generate_figures_v11.py — v10 + fully colorblind-safe palette (coral/mint/gold/sky)
All figures and Excel now use the same Nature-light CB-safe colors as Figure 1.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ══════════════════════════════════════════════════════════════════════════════
# FONT
# ══════════════════════════════════════════════════════════════════════════════
def _best_font():
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in ["IBM Plex Sans", "Noto Sans", "Source Sans 3",
               "Source Sans Pro", "Open Sans", "Lato", "DejaVu Sans"]:
        if c in avail:
            return c
    return "DejaVu Sans"

FONT_FAMILY = _best_font()

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════════════════
LIGHT = dict(
    sky      = "#77AADD",
    mint     = "#44BB99",
    coral    = "#EE8866",
    gold     = "#EEDD88",
    lavender = "#BBBBEE",
    pink     = "#FFAABB",
    teal     = "#99DDFF",
    olive    = "#AAAA44",
    white    = "#FFFFFF",
    off_white= "#FDFCF8",
)

NAT = dict(
    bg        = "#FDFCF8",
    panel_bg  = "#F0F4F8",
    grid      = "#DCE6EF",
    text_dark = "#1C2B3A",
    text_mid  = "#4A6070",
    pos       = LIGHT["mint"],
    neg       = LIGHT["coral"],
    neutral   = LIGHT["gold"],
    accent1   = LIGHT["sky"],
)

CB = dict(
    bg          = "#FDFCF8",
    cell_ns     = "#F5F5F0",
    grid_line   = "#DCE6EF",
    dim_sep     = "#AAAAAA",
    text_dark   = "#1C2B3A",
    text_mid    = "#4A6070",
    text_light  = "#8FA8BA",
    # CB-safe: mint replaces blue, coral replaces red — same as Fig 1
    blue_strong = LIGHT["mint"],      # "#44BB99"
    red_strong  = LIGHT["coral"],     # "#EE8866"
    # Dim bands: pale tints of gold/mint/sky/lavender
    dim_bands   = ["#FDF8DC", "#DCF4EC", "#DDEEFF", "#EEEEFF"],
)

GROUP_COLORS = dict(
    Group1 = LIGHT["gold"],
    Group2 = LIGHT["sky"],
    Group3 = LIGHT["mint"],
    Group4 = LIGHT["coral"],
)

STACK_COLORS = [LIGHT["coral"], LIGHT["gold"], LIGHT["mint"], LIGHT["sky"]]

GROUP_SHORT = {
    "Group1": "Control\n(Static Image + Written Text)",
    "Group2": "Treatment A\n(Static Image + Standard Video + Written Text)",
    "Group3": "Treatment B\n(Standard Video + Written Text, No Image)",
    "Group4": "Treatment C\n(Enhanced Video + Static Image + Written Text)",
}
GROUP_LABEL = {
    "Group1": "Control (Image + Text)",
    "Group2": "Treat. A (Image + Std. Video + Text)",
    "Group3": "Treat. B (Video + Text, No Image)",
    "Group4": "Treat. C (Enh. Video + Image + Text)",
}

ALL_GROUPS     = ["Group1", "Group2", "Group3", "Group4"]
BASELINE_GROUP = "Group1"

DIMENSION_LABELS = [
    "Environmental Sustainability",
    "Social Sustainability",
    "Economic Sustainability",
]
CRITERION_LABELS = [
    "Credibility", "Emotional Appeal",
    "Logical Reasoning", "Timing", "Clarity",
]
ALL_GRID   = [f"Grid{d}_{c}" for d in range(1, 4) for c in range(1, 6)]
LIKERT_MAP = {
    "highly ineffective": 1, "somewhat ineffective": 2,
    "somewhat effective": 3, "highly effective": 4,
}
ALPHA = 0.05
DPI   = 300

plt.rcParams.update({
    "figure.facecolor"  : NAT["bg"],
    "axes.facecolor"    : NAT["panel_bg"],
    "axes.edgecolor"    : NAT["grid"],
    "axes.labelcolor"   : NAT["text_dark"],
    "axes.grid"         : True,
    "grid.color"        : NAT["grid"],
    "grid.linewidth"    : 0.7,
    "text.color"        : NAT["text_dark"],
    "xtick.color"       : NAT["text_mid"],
    "ytick.color"       : NAT["text_mid"],
    "font.family"       : FONT_FAMILY,
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.titlepad"     : 11,
    "figure.dpi"        : DPI,
    "savefig.dpi"       : DPI,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : NAT["bg"],
})

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════
def make_synthetic_df(n_per_group: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    gb  = {"Group1": 0.0, "Group2": 0.25, "Group3": 0.15, "Group4": 0.35}
    im  = {"Group1": 52,  "Group2": 61,   "Group3": 57,   "Group4": 65}
    rows = []
    for grp in ALL_GROUPS:
        for i in range(n_per_group):
            row = {"ResponseId": f"{grp}_{i:03d}", "Group": grp}
            for col in ALL_GRID:
                p = np.clip(0.5 + gb[grp] + rng.normal(0, 0.15), 0.05, 0.95)
                row[col] = rng.choice(
                    [1, 2, 3, 4],
                    p=[max(0, (1-p)*0.4), max(0, (1-p)*0.6),
                       max(0, p*0.6),     max(0, p*0.4)],
                )
            lvls = [25, 50, 75, 100]
            pv   = np.exp(-0.5 * ((np.array(lvls) - im[grp]) / 20)**2)
            pv  /= pv.sum()
            row["Invest"] = rng.choice(lvls, p=pv)
            rows.append(row)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _norm(v):
    return str(v).strip().lower().rstrip(".")

def parse_investment(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.rstrip("%").replace("", np.nan).astype(float)

def encode_likert(s: pd.Series) -> pd.Series:
    return s.map(lambda x: LIKERT_MAP.get(_norm(x),
                 x if isinstance(x, (int, float)) else np.nan))

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ALL_GRID:
        if col in df.columns:
            df[col] = encode_likert(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Invest_num"] = parse_investment(df["Invest"])
    return df

def build_item_registry() -> list:
    items, n = [], 1
    for di, dim in enumerate(DIMENSION_LABELS, 1):
        for ci, crit in enumerate(CRITERION_LABELS, 1):
            items.append({"item_id": f"Q{n}", "column": f"Grid{di}_{ci}",
                          "dimension": dim, "criterion": crit, "dim_idx": di - 1})
            n += 1
    items.append({"item_id": "Invest", "column": "Invest_num",
                  "dimension": "Investment Intent",
                  "criterion": "Proportion Allocated (%)", "dim_idx": 3})
    return items

def _sig_star(p) -> str:
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"

def _pstr(p) -> str:
    if pd.isna(p): return "—"
    if p < 0.001:  return "p<.001"
    return f"p={p:.3f}"

def _save(fig, name: str, out_dir: Path, bg: str = None) -> Path:
    bg = bg or NAT["bg"]
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=DPI,  bbox_inches="tight", facecolor=bg)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    print(f"  ✓ {png_path.name}  +  {svg_path.name}")
    return png_path

def _nature_div_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "nat_div",
        ["#F4A98A", NAT["bg"], "#7DD4BE"],
        N=256,
    )

def _mwu_div_cmap():
    # Coral (treatment lower) → off-white → mint (treatment higher) — same as Fig 1
    return mcolors.LinearSegmentedColormap.from_list(
        "mwu_div",
        [LIGHT["coral"], "#FAE8E0", NAT["bg"], "#C8EDE4", LIGHT["mint"]],
        N=512,
    )

def _ax_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NAT["grid"])
    ax.spines["bottom"].set_color(NAT["grid"])
    ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════════════
# EFFECT SIZE — Cliff's Delta
# ══════════════════════════════════════════════════════════════════════════════
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    m = len(b)
    greater = sum(x > y for x in b for y in a)
    less    = sum(x < y for x in b for y in a)
    return (greater - less) / (n * m)


def effect_label(delta: float) -> str:
    d = abs(delta)
    if d < 0.11:  return "negligible"
    elif d < 0.28: return "small"
    elif d < 0.47: return "medium"
    else:          return "large"


# ══════════════════════════════════════════════════════════════════════════════
# MWU COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_mwu_pairs(df: pd.DataFrame, pairs: list) -> pd.DataFrame:
    df    = _prepare_df(df)
    items = build_item_registry()
    grouped = {g: d for g, d in df.groupby("Group")}
    rows      = []
    all_pvals = []

    for ga, gb_grp, pair_label in pairs:
        for it in items:
            col = it["column"]
            if ga not in grouped or gb_grp not in grouped:
                continue
            av = grouped[ga][col].dropna().values
            bv = grouped[gb_grp][col].dropna().values
            if len(av) == 0 or len(bv) == 0:
                continue
            _, p  = stats.mannwhitneyu(bv, av, alternative="greater", method="auto")
            delta = cliffs_delta(av, bv)
            diff  = float(np.mean(bv)) - float(np.mean(av))
            row = {
                "pair"       : pair_label,
                "ga"         : ga,
                "gb"         : gb_grp,
                "item_id"    : it["item_id"],
                "dim"        : it["dimension"],
                "crit"       : it["criterion"],
                "dim_idx"    : it["dim_idx"],
                "p_raw"      : p,
                "diff"       : diff,
                "cliff_delta": delta,
                "n_ga"       : len(av),
                "n_gb"       : len(bv),
            }
            rows.append(row)
            all_pvals.append(p)

    results_df = pd.DataFrame(rows)
    if len(all_pvals) > 0:
        rej, p_adj, _, _ = multipletests(all_pvals, alpha=ALPHA, method="holm")
        results_df["p_holm"] = p_adj
        results_df["sig"]    = rej
    else:
        results_df["p_holm"] = np.nan
        results_df["sig"]    = False

    results_df["effect_size"] = results_df["cliff_delta"].apply(effect_label)
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# EXCEL TABLE EXPORT  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
def export_mwu_excel(mwu_df: pd.DataFrame, out_path: Path):
    """
    Pivot table: items as rows, one column per comparison pair showing p (Holm).
    Mint = significant, warm-white = not significant.
    """
    thin = Side(style="thin", color="BBBBBB")
    bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)

    def hfill(hex_col):
        return PatternFill("solid", fgColor=hex_col)

    mid = Alignment(horizontal="center", vertical="center", wrap_text=True)
    lft = Alignment(horizontal="left",   vertical="center", wrap_text=True)

    dim_colors = {
        "Environmental Sustainability": "FDF8DC",
        "Social Sustainability":        "DCF4EC",
        "Economic Sustainability":      "DDEEFF",
        "Investment Intent":            "EEEEFF",
    }

    wb    = Workbook()
    ws    = wb.active
    ws.title = "p-values (Holm)"
    ws.freeze_panes = "C2"

    pairs = mwu_df["pair"].unique().tolist()
    items = mwu_df[["item_id", "dim", "crit"]].drop_duplicates("item_id")

    # ── Header row ────────────────────────────────────────────────────────────
    for ci, label in enumerate(["Item", "Criterion"] + pairs, 1):
        c = ws.cell(row=1, column=ci, value=label)
        c.font      = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.fill      = hfill("1C2B3A")
        c.alignment = mid
        c.border    = bdr
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 30
    for pi in range(len(pairs)):
        ws.column_dimensions[get_column_letter(3 + pi)].width = 22
    ws.row_dimensions[1].height = 32

    # ── Data rows ─────────────────────────────────────────────────────────────
    for ri, (_, item) in enumerate(items.iterrows(), 2):
        bg = dim_colors.get(item["dim"], "F5F5F0")

        c1 = ws.cell(row=ri, column=1, value=item["item_id"])
        c1.font = Font(name="Arial", size=9, bold=True); c1.fill = hfill(bg)
        c1.border = bdr; c1.alignment = mid

        c2 = ws.cell(row=ri, column=2,
                     value=f"{item['crit']} ({item['dim'].replace(' Sustainability', '')})")
        c2.font = Font(name="Arial", size=9); c2.fill = hfill(bg)
        c2.border = bdr; c2.alignment = lft

        for pi, pair in enumerate(pairs):
            match = mwu_df[(mwu_df["pair"] == pair) & (mwu_df["item_id"] == item["item_id"])]
            if match.empty:
                continue
            r      = match.iloc[0]
            p_h    = r["p_holm"]
            is_sig = r["sig"]
            stars  = _sig_star(p_h)
            label  = f"{p_h:.4f}  {stars}" if not pd.isna(p_h) else "—"

            c = ws.cell(row=ri, column=3 + pi, value=label)
            c.font   = Font(name="Arial", size=9, bold=is_sig,
                            color="1F7A5E" if is_sig else "555555")
            c.fill   = hfill("C8EDE4" if is_sig else bg)
            c.border = bdr
            c.alignment = mid

        ws.row_dimensions[ri].height = 16

    # ── Footer note ───────────────────────────────────────────────────────────
    last_row = len(items) + 2
    ws.merge_cells(f"A{last_row}:{get_column_letter(2 + len(pairs))}{last_row}")
    note = ws.cell(row=last_row, column=1,
                   value="p (Holm) = Holm–Bonferroni corrected p-value  |  "
                         "* p<.05  ** p<.01  *** p<.001  ns = p≥.05  |  "
                         "Mint = significant  |  One-sided MWU: H₁ = treatment > control")
    note.font      = Font(name="Arial", size=8, italic=True, color="4A6070")
    note.alignment = lft

    wb.save(out_path)
    print(f"  ✓ {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MWU HEATMAP RENDERER  (Fig 03 & Fig 07)
# ══════════════════════════════════════════════════════════════════════════════
def _cell_color(diff, p_holm, sig):
    if not sig or pd.isna(p_holm) or pd.isna(diff):
        return mcolors.to_rgba(CB["cell_ns"])
    FLOOR     = 1e-7
    log_p     = -np.log10(max(p_holm, FLOOR))
    intensity = min(log_p / 3.0, 1.0)
    if diff > 0:
        # mint: #44BB99 = (0.267, 0.733, 0.600)
        r = 1.0 - intensity * (1.0 - 0.267)
        g = 1.0 - intensity * (1.0 - 0.733)
        b = 1.0 - intensity * (1.0 - 0.600)
    else:
        # coral: #EE8866 = (0.933, 0.533, 0.400)
        r = 1.0 - intensity * (1.0 - 0.933)
        g = 1.0 - intensity * (1.0 - 0.533)
        b = 1.0 - intensity * (1.0 - 0.400)
    return (r, g, b, 1.0)


def _render_mwu_heatmap(mwu_df, fig_number, fig_title, out_dir, out_name):
    items   = build_item_registry()
    n_items = len(items)
    pairs   = mwu_df["pair"].unique().tolist()
    n_pairs = len(pairs)

    DIM_RANGES = [(0, 4), (5, 9), (10, 14), (15, 15)]
    DIM_BREAKS = [4.5, 9.5, 14.5]
    DIM_MIDS   = [2.0, 7.0, 12.0, 15.0]
    DIM_SHORT  = ["Environmental", "Social", "Economic", "Investment\nIntent"]

    CELL_H   = 1.05
    CELL_W   = 7.2
    LEFT_MAR = 5.2
    CB_W     = 1.1
    TOP_PAD  = 2.5
    BOT_PAD  = 0.5

    fig_w = LEFT_MAR + CELL_W * n_pairs + CB_W + 0.5
    fig_h = TOP_PAD + BOT_PAD + n_items * CELL_H

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor(CB["bg"])

    top_frac = 1.0 - TOP_PAD / fig_h
    bot_frac = BOT_PAD / fig_h

    wr = [CELL_W] * n_pairs + [CB_W]
    gs = gridspec.GridSpec(
        1, n_pairs + 1,
        width_ratios=wr,
        wspace=0.035,
        left=LEFT_MAR / fig_w,
        right=1.0 - (CB_W + 0.45) / fig_w,
        top=top_frac,
        bottom=bot_frac,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_pairs)]
    cax  = fig.add_subplot(gs[0, n_pairs])

    T = fig_h
    def _fy(inches_from_top):
        return 1.0 - inches_from_top / T

    lines = fig_title.split("\n")
    fig.text(0.5, _fy(0.28), lines[0], ha="center", va="top",
             fontsize=15, fontweight="bold", color=CB["text_dark"],
             transform=fig.transFigure)
    if len(lines) > 1:
        fig.text(0.5, _fy(0.72), lines[1], ha="center", va="top",
                 fontsize=10.5, color=CB["text_mid"],
                 transform=fig.transFigure)
    fig.text(
        0.5, _fy(1.12),
        "Cell color: Blue = treatment (gb) HIGHER  ·  Red = treatment lower  ·  "
        "Intensity ∝ −log₁₀(p_Holm)  ·  Grey = not significant  ·  "
        "Global Holm–Bonferroni across all comparisons",
        ha="center", va="top", fontsize=9, color=CB["text_mid"],
        transform=fig.transFigure,
    )

    patch_top = _fy(1.62)
    patch_h_f = 0.018
    patches_info = [
        (LIGHT["mint"],  "Treatment higher, sig. (p<.05)"),
        (CB["cell_ns"],  "Not significant (ns)"),
        (LIGHT["coral"], "Treatment lower, sig. (unexpected)"),
    ]
    patch_xs = [0.22, 0.46, 0.70]
    for px, (pc, pl) in zip(patch_xs, patches_info):
        fig.add_artist(
            mpatches.FancyBboxPatch(
                (px - 0.013, patch_top - patch_h_f / 2), 0.020, patch_h_f,
                boxstyle="round,pad=0.003",
                facecolor=pc, edgecolor="#AABBCC", linewidth=0.8,
                transform=fig.transFigure, zorder=5,
            )
        )
        fig.text(px + 0.013, patch_top, pl,
                 ha="left", va="center", fontsize=8.5, color=CB["text_dark"],
                 transform=fig.transFigure)

    y_pos = np.arange(n_items)

    for ax_i, (ax, pair_label) in enumerate(zip(axes, pairs)):
        sub    = (mwu_df[mwu_df["pair"] == pair_label]
                  .sort_values("dim_idx").reset_index(drop=True))
        ga_key = sub["ga"].iloc[0]
        gb_key = sub["gb"].iloc[0]
        n_ga_val = int(sub["n_ga"].iloc[0]) if "n_ga" in sub.columns else "?"
        n_gb_val = int(sub["n_gb"].iloc[0]) if "n_gb" in sub.columns else "?"

        ax.set_facecolor(CB["bg"])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(n_items - 0.5, -0.5)

        for bi, (lo, hi) in enumerate(DIM_RANGES):
            ax.axhspan(lo - 0.5, hi + 0.5,
                       facecolor=CB["dim_bands"][bi], alpha=0.45, zorder=0)

        for item_i in range(n_items):
            if item_i >= len(sub):
                continue
            rs    = sub.iloc[item_i]
            p_raw = rs["p_raw"]
            p_h   = rs["p_holm"]
            diff  = rs["diff"]
            sig   = rs["sig"]
            delta = rs.get("cliff_delta", np.nan)
            eff   = rs.get("effect_size", "")

            rgba = _cell_color(diff, p_h, sig)
            rect = mpatches.FancyBboxPatch(
                (-0.46, item_i - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0.015",
                facecolor=rgba, edgecolor="#D0D8E0",
                linewidth=0.8, zorder=2,
            )
            ax.add_patch(rect)

            r_, g_, b_, _ = rgba
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            tc_main  = "#FFFFFF" if lum < 0.55 else CB["text_dark"]
            tc_sub   = "#FFFFFF" if lum < 0.55 else CB["text_mid"]
            tc_faded = "#A0B4C4"

            if sig:
                ax.text(0, item_i - 0.26, _sig_star(p_h),
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=tc_main, zorder=4)
            else:
                ax.text(0, item_i - 0.26, "ns",
                        ha="center", va="center",
                        fontsize=11, style="italic",
                        color=tc_faded, zorder=4)

            p_str = _pstr(p_raw)
            ax.text(0, item_i + 0.02, p_str,
                    ha="center", va="center",
                    fontsize=10 if sig else 9,
                    fontweight="bold" if sig else "normal",
                    color=tc_main if sig else tc_faded,
                    zorder=4)

            if not pd.isna(delta):
                ds = f"δ={delta:+.2f} ({eff})"
                ax.text(0, item_i + 0.30, ds,
                        ha="center", va="center",
                        fontsize=7.5,
                        color=tc_sub if sig else tc_faded,
                        zorder=4)

        for yb in DIM_BREAKS:
            ax.axhline(yb, color=CB["dim_sep"], linewidth=1.6,
                       linestyle="--", zorder=3)

        n_pos = int(((sub["sig"]) & (sub["diff"] > 0.005)).sum())
        n_neg = int(((sub["sig"]) & (sub["diff"] < -0.005)).sum())
        n_ns  = int((~sub["sig"]).sum())
        verdict = f"✓ {n_pos} improved"
        if n_neg:
            verdict += f"  ✗ {n_neg} worsened"
        verdict += f"  ○ {n_ns} n.s."

        ax.set_title(
            f"H₁: {GROUP_LABEL[gb_key]} (n={n_gb_val})\n"
            f"     > {GROUP_LABEL[ga_key]} (n={n_ga_val})\n{verdict}",
            fontsize=9.5, fontweight="bold",
            color=CB["text_dark"], pad=10, linespacing=1.6,
        )

        ax.set_xticks([]); ax.set_yticks(y_pos)
        ax.tick_params(axis="y", length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if ax_i == 0:
            ylabels = [f"{it['item_id']}  {it['criterion']}" for it in items]
            ax.set_yticklabels(ylabels, fontsize=9.5, color=CB["text_dark"],
                               ha="right")
            ax.tick_params(axis="y", length=0, pad=6)
        else:
            ax.set_yticklabels([])

    ax0 = axes[0]
    for mi, (mid_y, name) in enumerate(zip(DIM_MIDS, DIM_SHORT)):
        disp  = ax0.transData.transform((0, mid_y))
        fig_y = fig.transFigure.inverted().transform(disp)[1]
        fig.text(
            0.005, fig_y, name,
            ha="left", va="center",
            fontsize=8.5, fontweight="bold", color=CB["text_mid"],
            transform=fig.transFigure,
        )

    cmap_cb = mcolors.LinearSegmentedColormap.from_list(
        "cb_manual",
        [LIGHT["coral"], "#FAE8E0", CB["cell_ns"], "#C8EDE4", LIGHT["mint"]],
        N=256,
    )
    norm_cb = mcolors.Normalize(vmin=-3, vmax=3)
    sm = plt.cm.ScalarMappable(cmap=cmap_cb, norm=norm_cb)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("Color key\n(−log₁₀ p, signed)", fontsize=9,
                 color=CB["text_mid"], labelpad=6)
    cb.set_ticks([-3, -2, -np.log10(0.05), 0,
                   np.log10(0.05**-1), 2, 3])
    cb.set_ticklabels(
        ["p<.001\n(coral/lower)", "p=.01\n(lower)", "p=.05\n(lower)",
         "n.s.",
         "p=.05\n(higher)", "p=.01\n(higher)", "p<.001\n(mint/higher)"],
        fontsize=7.5,
    )
    cax.yaxis.set_tick_params(color=CB["text_mid"], length=3)
    for s in cax.spines.values():
        s.set_visible(False)

    fig.text(
        0.5, 0.008,
        "One-sided MWU: H₁ = treatment (gb) > control (ga)  ·  "
        "Global Holm–Bonferroni correction  ·  "
        "* p<.05  ** p<.01  *** p<.001  ns = p≥.05  ·  "
        "δ = Cliff's Delta (effect size)",
        ha="center", va="bottom",
        fontsize=8.5, color=CB["text_mid"], style="italic",
    )
    return _save(fig, out_name, out_dir, bg=CB["bg"])


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01
# ══════════════════════════════════════════════════════════════════════════════
def fig01_delta_heatmap_ab(df: pd.DataFrame, out_dir: Path) -> Path:
    df = _prepare_df(df)
    g1m = {f"Grid{di}_{ci}":
           df[df["Group"] == BASELINE_GROUP][f"Grid{di}_{ci}"].mean()
           for di in range(1, 4) for ci in range(1, 6)}
    deltas = {}
    for g in ["Group2", "Group3"]:
        sub = df[df["Group"] == g]
        deltas[g] = np.array(
            [[sub[f"Grid{di}_{ci}"].mean() - g1m[f"Grid{di}_{ci}"]
              for ci in range(1, 6)] for di in range(1, 4)],
            dtype=float,
        )
    vlim = max(abs(np.nanmin(list(deltas.values()))),
               abs(np.nanmax(list(deltas.values()))), 0.05)
    cmap = _nature_div_cmap()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    fig.patch.set_facecolor(NAT["bg"])
    fig.suptitle(
        "Figure 1 — Mean Score Difference vs Control\n"
        "Treatment A (Static Image + Standard Video + Written Text)   |   "
        "Treatment B (Standard Video + Written Text, No Image)",
        fontsize=13, fontweight="bold", color=NAT["text_dark"],
        y=1.02, linespacing=1.65,
    )
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.30, wspace=0.22)

    for ax, grp in zip(axes, ["Group2", "Group3"]):
        sns.heatmap(
            deltas[grp], ax=ax,
            annot=True, fmt="+.2f", cmap=cmap,
            vmin=-vlim, vmax=vlim, center=0, cbar=False,
            linewidths=1.6, linecolor=NAT["bg"],
            annot_kws={"fontsize": 13, "fontweight": "bold",
                       "color": NAT["text_dark"]},
        )
        ax.set_title(
            f"{GROUP_SHORT[grp]}\nvs Control (Static Image + Written Text)",
            fontsize=11, fontweight="bold", color=GROUP_COLORS[grp],
            pad=14, linespacing=1.55,
        )
        ax.set_xticklabels(CRITERION_LABELS, rotation=40, ha="right",
                           fontsize=11, rotation_mode="anchor")
        ax.set_yticklabels(
            [d.replace(" Sustainability", "") for d in DIMENSION_LABELS],
            rotation=0, fontsize=11,
        )
        ax.tick_params(axis="x", pad=7)
        ax.set_facecolor(NAT["panel_bg"])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vlim, vlim))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.030, pad=0.38, aspect=45)
    cbar.set_label(
        "Mean Score Δ vs Control   |   Mint = Improvement   |   Coral = Decline",
        fontsize=10, color=NAT["text_mid"],
    )
    return _save(fig, "fig01_delta_heatmap_AB", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02
# ══════════════════════════════════════════════════════════════════════════════
def fig02_investment_ab(df: pd.DataFrame, out_dir: Path) -> Path:
    df = df.copy()
    df["Invest_num"] = parse_investment(df["Invest"])
    groups   = ["Group1", "Group2", "Group3"]
    HIGH_CUT = 75
    high_rate = (
        df[df["Group"].isin(groups)]
        .assign(h=lambda d: d["Invest_num"] >= HIGH_CUT)
        .groupby("Group")["h"].mean()
        .reindex(groups) * 100
    )
    invest_levels = [25, 50, 75, 100]
    dist = (
        df[df["Group"].isin(groups)]
        .groupby(["Group", "Invest_num"])["ResponseId"].count()
        .unstack(fill_value=0)
        .reindex(index=groups, columns=invest_levels, fill_value=0)
    )
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    x            = np.arange(len(groups))
    xtick_labels = [GROUP_SHORT[g] for g in groups]
    bar_colors   = [GROUP_COLORS[g] for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    fig.patch.set_facecolor(NAT["bg"])
    fig.suptitle(
        "Figure 2 — Investment Allocation\n"
        "Control (Image + Text)   ·   "
        "Treatment A (Image + Std. Video + Text)   ·   "
        "Treatment B (Video + Text, No Image)",
        fontsize=13, fontweight="bold", color=NAT["text_dark"],
        y=1.02, linespacing=1.65,
    )
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.88,
                        top=0.88, bottom=0.30)

    ax1.bar(x, high_rate.values, color=bar_colors,
            edgecolor=NAT["bg"], linewidth=1.5, width=0.55)
    for xi, r in zip(x, high_rate.values):
        ax1.text(xi, r + 1.5, f"{r:.1f}%",
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=NAT["text_dark"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, rotation=0, ha="center",
                        fontsize=9.5, linespacing=1.45)
    ax1.set_ylim(0, min(100, high_rate.max() + 26))
    ax1.set_ylabel("% of Respondents", fontsize=11, color=NAT["text_mid"])
    ax1.set_title("Panel A — Proportion Allocating ≥ 75%", fontsize=12)
    ax1.tick_params(axis="x", pad=10)
    _ax_clean(ax1)

    bottoms = np.zeros(len(groups))
    for lvl, sc in zip(invest_levels, STACK_COLORS):
        vals = dist_pct[lvl].values
        ax2.bar(x, vals, bottom=bottoms, color=sc,
                edgecolor=NAT["bg"], label=f"{lvl}%", width=0.55)
        for xi, v, cy in zip(x, vals, bottoms + vals / 2):
            if v >= 7:
                ax2.text(xi, cy, f"{v:.0f}%",
                         ha="center", va="center",
                         fontsize=10, fontweight="bold",
                         color=NAT["text_dark"])
        bottoms += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(xtick_labels, rotation=0, ha="center",
                        fontsize=9.5, linespacing=1.45)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of Respondents", fontsize=11, color=NAT["text_mid"])
    ax2.set_title("Panel B — Investment Allocation Distribution", fontsize=12)
    ax2.tick_params(axis="x", pad=10)
    ax2.legend(title="Invest Level", bbox_to_anchor=(1.03, 1),
               loc="upper left", fontsize=10, title_fontsize=10)
    _ax_clean(ax2)
    return _save(fig, "fig02_investment_AB", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03
# ══════════════════════════════════════════════════════════════════════════════
def fig03_mwu_ab(df: pd.DataFrame, out_dir: Path) -> Path:
    pairs = [
        ("Group1", "Group2", "Control vs Treatment A"),
        ("Group1", "Group3", "Control vs Treatment B"),
        ("Group2", "Group3", "Treatment A vs Treatment B"),
    ]
    mwu_df = compute_mwu_pairs(df, pairs)
    return _render_mwu_heatmap(
        mwu_df, 3,
        "Figure 3 — One-Sided Mann–Whitney U: Does the Treatment Outperform?\n"
        "H₁: Treatment (gb) > Control/Comparison (ga)  ·  Global Holm–Bonferroni correction",
        out_dir, "fig03_mwu_heatmap_AB",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04
# ══════════════════════════════════════════════════════════════════════════════
def fig04_spearman(df: pd.DataFrame, out_dir: Path) -> Path:
    df        = _prepare_df(df)
    items_all = [it for it in build_item_registry() if it["item_id"] != "Invest"]
    item_ids  = [it["item_id"] for it in items_all]
    item_label = {
        it["item_id"]: f"{it['item_id']}  {it['criterion']} "
                       f"({it['dimension'].replace(' Sustainability', '')})"
        for it in items_all
    }
    pooled_rho = {}
    for it in items_all:
        col    = it["column"]
        common = df[["Invest_num", col]].dropna()
        if len(common) < 5:
            pooled_rho[it["item_id"]] = 0.0; continue
        rho, _ = stats.spearmanr(common[col], common["Invest_num"])
        pooled_rho[it["item_id"]] = abs(rho)
    sort_order = sorted(item_ids, key=lambda k: pooled_rho[k], reverse=True)

    lim = 0.65
    fig, axes = plt.subplots(1, 4, figsize=(32, 16), sharey=True)
    fig.patch.set_facecolor(NAT["bg"])
    fig.suptitle(
        "Figure 4 — Spearman's ρ: Feature Perception → Investment Allocation\n"
        "All four groups  ·  Items ranked by pooled |ρ|  ·  Faded = p ≥ .05",
        fontsize=14, fontweight="bold", color=NAT["text_dark"],
        y=1.02, linespacing=1.65,
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.93,
                        bottom=0.07, wspace=0.10)

    for ax, grp in zip(axes, ALL_GROUPS):
        ax.set_facecolor(NAT["panel_bg"])
        grp_data = df[df["Group"] == grp]
        rhos, bar_cols, alphas, ylabels = [], [], [], []

        for iid in sort_order:
            it     = next(i for i in items_all if i["item_id"] == iid)
            col    = it["column"]
            common = grp_data[["Invest_num", col]].dropna()
            if len(common) < 5:
                rhos.append(0.0)
                bar_cols.append(NAT["neutral"])
                alphas.append(0.2)
            else:
                rho, p = stats.spearmanr(common[col], common["Invest_num"])
                sig    = p < ALPHA
                rhos.append(rho)
                bar_cols.append(NAT["pos"] if rho >= 0 else NAT["neg"])
                alphas.append(1.0 if sig else 0.30)
            ylabels.append(item_label[iid])

        y    = np.arange(len(sort_order))
        bars = ax.barh(y, rhos, color=bar_cols,
                       edgecolor=NAT["bg"], linewidth=0.5, height=0.75)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)

        for yi, iid in enumerate(sort_order):
            it     = next(i for i in items_all if i["item_id"] == iid)
            col    = it["column"]
            common = grp_data[["Invest_num", col]].dropna()
            if len(common) < 5: continue
            rho, p = stats.spearmanr(common[col], common["Invest_num"])
            if abs(rho) < 0.005: continue
            sig  = p < ALPHA
            xoff = 0.012 if rho >= 0 else -0.012
            lbl  = f"{rho:+.2f}" + (f" {_sig_star(p)}" if sig else "")
            ax.text(rho + xoff, yi, lbl,
                    va="center", ha="left" if rho >= 0 else "right",
                    fontsize=8.5,
                    fontweight="bold" if sig else "normal",
                    color=NAT["text_dark"])

        ax.axvline(0, color=NAT["text_dark"], linewidth=1.0, zorder=5)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(y)
        ax.set_xlabel("Spearman's ρ", fontsize=11, color=NAT["text_mid"])
        ax.set_title(GROUP_SHORT[grp], fontsize=11, fontweight="bold",
                     color=GROUP_COLORS[grp], pad=12, linespacing=1.5)
        _ax_clean(ax)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, color=NAT["grid"])
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.set_yticklabels(ylabels, fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    legend_h = [
        mpatches.Patch(color=NAT["pos"],  alpha=1.0,  label="Positive ρ (sig.)"),
        mpatches.Patch(color=NAT["neg"],  alpha=1.0,  label="Negative ρ (sig.)"),
        mpatches.Patch(color=NAT["pos"],  alpha=0.30, label="Positive ρ (n.s.)"),
        mpatches.Patch(color=NAT["neg"],  alpha=0.30, label="Negative ρ (n.s.)"),
    ]
    fig.legend(handles=legend_h, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.03),
               facecolor=NAT["panel_bg"], edgecolor=NAT["grid"])
    return _save(fig, "fig04_spearman", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05
# ══════════════════════════════════════════════════════════════════════════════
def fig05_delta_heatmap_c(df: pd.DataFrame, out_dir: Path) -> Path:
    df = _prepare_df(df)
    g1m = {f"Grid{di}_{ci}":
           df[df["Group"] == BASELINE_GROUP][f"Grid{di}_{ci}"].mean()
           for di in range(1, 4) for ci in range(1, 6)}
    sub       = df[df["Group"] == "Group4"]
    delta_mat = np.array(
        [[sub[f"Grid{di}_{ci}"].mean() - g1m[f"Grid{di}_{ci}"]
          for ci in range(1, 6)] for di in range(1, 4)],
        dtype=float,
    )
    vlim = max(abs(np.nanmin(delta_mat)), abs(np.nanmax(delta_mat)), 0.05)
    cmap = _nature_div_cmap()

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(NAT["bg"])
    fig.suptitle(
        "Figure 5 — Mean Score Difference vs Control\n"
        "Treatment C (Enhanced Video + Static Image + Written Text)",
        fontsize=13, fontweight="bold", color=NAT["text_dark"],
        y=1.04, linespacing=1.65,
    )
    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.30)

    sns.heatmap(
        delta_mat, ax=ax,
        annot=True, fmt="+.2f", cmap=cmap,
        vmin=-vlim, vmax=vlim, center=0, cbar=False,
        linewidths=1.6, linecolor=NAT["bg"],
        annot_kws={"fontsize": 14, "fontweight": "bold",
                   "color": NAT["text_dark"]},
    )
    ax.set_title(
        f"{GROUP_SHORT['Group4']}\nvs Control (Static Image + Written Text)",
        fontsize=11, fontweight="bold", color=GROUP_COLORS["Group4"],
        pad=14, linespacing=1.55,
    )
    ax.set_xticklabels(CRITERION_LABELS, rotation=40, ha="right",
                       fontsize=12, rotation_mode="anchor")
    ax.set_yticklabels(
        [d.replace(" Sustainability", "") for d in DIMENSION_LABELS],
        rotation=0, fontsize=12,
    )
    ax.tick_params(axis="x", pad=7)
    ax.set_facecolor(NAT["panel_bg"])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vlim, vlim))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.052, pad=0.38, aspect=32)
    cbar.set_label(
        "Mean Score Δ vs Control   |   Mint = Improvement   |   Coral = Decline",
        fontsize=10, color=NAT["text_mid"],
    )
    return _save(fig, "fig05_delta_heatmap_C", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06
# ══════════════════════════════════════════════════════════════════════════════
def fig06_investment_c(df: pd.DataFrame, out_dir: Path) -> Path:
    df = df.copy()
    df["Invest_num"] = parse_investment(df["Invest"])
    groups   = ["Group1", "Group4"]
    HIGH_CUT = 75
    high_rate = (
        df[df["Group"].isin(groups)]
        .assign(h=lambda d: d["Invest_num"] >= HIGH_CUT)
        .groupby("Group")["h"].mean()
        .reindex(groups) * 100
    )
    invest_levels = [25, 50, 75, 100]
    dist = (
        df[df["Group"].isin(groups)]
        .groupby(["Group", "Invest_num"])["ResponseId"].count()
        .unstack(fill_value=0)
        .reindex(index=groups, columns=invest_levels, fill_value=0)
    )
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    x            = np.arange(len(groups))
    xtick_labels = [GROUP_SHORT[g] for g in groups]
    bar_colors   = [GROUP_COLORS[g] for g in groups]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11))
    fig.patch.set_facecolor(NAT["bg"])
    fig.suptitle(
        "Figure 6 — Investment Allocation\n"
        "Control (Static Image + Written Text)   vs   "
        "Treatment C (Enhanced Video + Static Image + Written Text)",
        fontsize=13, fontweight="bold", color=NAT["text_dark"],
        y=1.02, linespacing=1.65,
    )
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.88,
                        top=0.88, bottom=0.28)

    ax1.bar(x, high_rate.values, color=bar_colors,
            edgecolor=NAT["bg"], linewidth=1.5, width=0.45)
    for xi, r in zip(x, high_rate.values):
        ax1.text(xi, r + 1.5, f"{r:.1f}%",
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=NAT["text_dark"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(xtick_labels, rotation=0, ha="center",
                        fontsize=10, linespacing=1.45)
    ax1.set_ylim(0, min(100, high_rate.max() + 26))
    ax1.set_ylabel("% of Respondents", fontsize=11, color=NAT["text_mid"])
    ax1.set_title("Panel A — Proportion Allocating ≥ 75%", fontsize=12)
    ax1.tick_params(axis="x", pad=10)
    _ax_clean(ax1)

    bottoms = np.zeros(len(groups))
    for lvl, sc in zip(invest_levels, STACK_COLORS):
        vals = dist_pct[lvl].values
        ax2.bar(x, vals, bottom=bottoms, color=sc,
                edgecolor=NAT["bg"], label=f"{lvl}%", width=0.45)
        for xi, v, cy in zip(x, vals, bottoms + vals / 2):
            if v >= 7:
                ax2.text(xi, cy, f"{v:.0f}%",
                         ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color=NAT["text_dark"])
        bottoms += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(xtick_labels, rotation=0, ha="center",
                        fontsize=10, linespacing=1.45)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% of Respondents", fontsize=11, color=NAT["text_mid"])
    ax2.set_title("Panel B — Investment Allocation Distribution", fontsize=12)
    ax2.tick_params(axis="x", pad=10)
    ax2.legend(title="Invest Level", bbox_to_anchor=(1.03, 1),
               loc="upper left", fontsize=10, title_fontsize=10)
    _ax_clean(ax2)
    return _save(fig, "fig06_investment_C", out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07
# ══════════════════════════════════════════════════════════════════════════════
def fig07_mwu_a_vs_c(df: pd.DataFrame, out_dir: Path) -> Path:
    pairs = [("Group2", "Group4", "Treatment A vs Treatment C")]
    mwu_df = compute_mwu_pairs(df, pairs)
    return _render_mwu_heatmap(
        mwu_df, 7,
        "Figure 7 — One-Sided Mann–Whitney U: Treatment C vs Treatment A\n"
        "H₁: Treatment C (Enh. Video + Image + Text) > Treatment A (Std. Video + Image + Text)",
        out_dir, "fig07_mwu_A_vs_C",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08
# ══════════════════════════════════════════════════════════════════════════════
def _plot_cliff_delta_single(ax, sub_df, pair_label, items):
    sub = sub_df[sub_df["pair"] == pair_label].copy()
    id_order = [it["item_id"] for it in items]
    sub["y_pos"] = sub["item_id"].map({iid: i for i, iid in enumerate(id_order)})
    sub = sub.dropna(subset=["y_pos", "cliff_delta"]).sort_values("y_pos")

    y = sub["y_pos"].values
    x = sub["cliff_delta"].values
    colors = [LIGHT["mint"] if v > 0 else LIGHT["coral"] for v in x]

    ax.axvline(0, color="#9AABB8", linewidth=1.2, linestyle="--", zorder=1)
    for threshold, ls in [(0.11, ":"), (0.28, "-."), (0.47, "--")]:
        ax.axvline( threshold, color="#BBCCDD", linewidth=0.8, linestyle=ls, zorder=1)
        ax.axvline(-threshold, color="#BBCCDD", linewidth=0.8, linestyle=ls, zorder=1)

    ax.scatter(x, y, c=colors, s=110, zorder=3, edgecolors="#FFFFFF", linewidths=0.6)

    for xi, yi in zip(x, y):
        xoff = 0.025 if xi >= 0 else -0.025
        ha   = "left" if xi >= 0 else "right"
        ax.text(xi + xoff, yi, f"{xi:+.2f}", va="center", ha=ha,
                fontsize=7.5, color=CB["text_dark"])

    n_ga = int(sub["n_ga"].iloc[0]) if len(sub) else "?"
    n_gb = int(sub["n_gb"].iloc[0]) if len(sub) else "?"
    ga   = sub["ga"].iloc[0]  if len(sub) else ""
    gb   = sub["gb"].iloc[0]  if len(sub) else ""

    ax.set_title(
        f"{pair_label}\n"
        f"{GROUP_LABEL.get(gb, gb)} (n={n_gb}) vs {GROUP_LABEL.get(ga, ga)} (n={n_ga})",
        fontsize=9, fontweight="bold", color=CB["text_dark"], pad=8, linespacing=1.5,
    )
    ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{it['item_id']}  {it['criterion']}" for it in items], fontsize=8.5)
    ax.set_xlabel("Cliff's Delta (δ)", fontsize=9, color=CB["text_mid"])
    ax.invert_yaxis()
    ax.set_facecolor(CB["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color=NAT["grid"])
    ax.set_axisbelow(True)


def fig08_cliff_delta(df: pd.DataFrame, out_dir: Path) -> Path:
    pairs_ab = [
        ("Group1", "Group2", "Control vs Treatment A"),
        ("Group1", "Group3", "Control vs Treatment B"),
        ("Group2", "Group3", "Treatment A vs Treatment B"),
    ]
    pairs_c  = [("Group2", "Group4", "Treatment A vs Treatment C")]
    all_pairs = pairs_ab + pairs_c

    mwu_df = compute_mwu_pairs(df, all_pairs)
    items  = build_item_registry()
    pair_labels = [p[2] for p in all_pairs]
    n_pairs = len(pair_labels)

    fig, axes = plt.subplots(
        1, n_pairs,
        figsize=(7.5 * n_pairs, max(10, len(items) * 0.65 + 2)),
        sharey=False,
    )
    if n_pairs == 1:
        axes = [axes]

    fig.patch.set_facecolor(CB["bg"])
    fig.suptitle(
        "Figure 8 — Cliff's Delta: Nonparametric Effect Size\n"
        "Positive (green) = treatment higher  ·  Negative (red) = treatment lower  ·  "
        "Threshold lines: |δ|=0.11 (small), 0.28 (medium), 0.47 (large)",
        fontsize=13, fontweight="bold", color=CB["text_dark"],
        y=1.02, linespacing=1.65,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.06, wspace=0.35)

    for ax, pair_label in zip(axes, pair_labels):
        _plot_cliff_delta_single(ax, mwu_df, pair_label, items)
        if ax != axes[0]:
            ax.set_yticklabels([])
            ax.set_ylabel("")

    legend_patches = [
        mpatches.Patch(color=LIGHT["mint"],  label="δ > 0 (treatment higher)"),
        mpatches.Patch(color=LIGHT["coral"], label="δ < 0 (treatment lower)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02),
               facecolor=NAT["panel_bg"], edgecolor=NAT["grid"])
    return _save(fig, "fig08_cliff_delta", out_dir, bg=CB["bg"])


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09
# ══════════════════════════════════════════════════════════════════════════════
def fig09_effect_significance(df: pd.DataFrame, out_dir: Path) -> Path:
    pairs_ab = [
        ("Group1", "Group2", "Control vs Treatment A"),
        ("Group1", "Group3", "Control vs Treatment B"),
        ("Group2", "Group3", "Treatment A vs Treatment B"),
    ]
    pairs_c  = [("Group2", "Group4", "Treatment A vs Treatment C")]
    all_pairs = pairs_ab + pairs_c

    mwu_df = compute_mwu_pairs(df, all_pairs)
    items  = build_item_registry()
    pair_labels = [p[2] for p in all_pairs]
    n_pairs = len(pair_labels)

    fig, axes = plt.subplots(
        1, n_pairs,
        figsize=(8.0 * n_pairs, max(10, len(items) * 0.65 + 2)),
        sharey=False,
    )
    if n_pairs == 1:
        axes = [axes]

    fig.patch.set_facecolor(CB["bg"])
    fig.suptitle(
        "Figure 9 — Effect Size & Statistical Significance (Cliff's Delta)\n"
        "Crimson = significant after Holm–Bonferroni  ·  Grey = not significant  ·  "
        "p-values shown (Holm-corrected)",
        fontsize=13, fontweight="bold", color=CB["text_dark"],
        y=1.02, linespacing=1.65,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.06, wspace=0.40)

    for ax, pair_label in zip(axes, pair_labels):
        sub = mwu_df[mwu_df["pair"] == pair_label].copy()
        id_order = [it["item_id"] for it in items]
        sub["y_pos"] = sub["item_id"].map({iid: i for i, iid in enumerate(id_order)})
        sub = sub.dropna(subset=["y_pos", "cliff_delta"]).sort_values("y_pos")

        y      = sub["y_pos"].values
        x      = sub["cliff_delta"].values
        colors = [LIGHT["coral"] if s else LIGHT["gold"] for s in sub["sig"].values]

        ax.axvline(0, color="#9AABB8", linewidth=1.2, linestyle="--", zorder=1)
        for threshold, ls in [(0.11, ":"), (0.28, "-."), (0.47, "--")]:
            ax.axvline( threshold, color="#BBCCDD", linewidth=0.8, linestyle=ls, zorder=1)
            ax.axvline(-threshold, color="#BBCCDD", linewidth=0.8, linestyle=ls, zorder=1)

        ax.scatter(x, y, c=colors, s=120, zorder=3, edgecolors="#FFFFFF", linewidths=0.6)

        for xi, yi, p_h, sig, eff in zip(
            x, y, sub["p_holm"].values, sub["sig"].values, sub["effect_size"].values,
        ):
            p_str = _pstr(p_h)
            lbl   = f"{p_str} ({eff})" if sig else f"{p_str}"
            xoff  = 0.025 if xi >= 0 else -0.025
            ha    = "left"  if xi >= 0 else "right"
            ax.text(xi + xoff, yi, lbl, va="center", ha=ha,
                    fontsize=7.0,
                    fontweight="bold" if sig else "normal",
                    color=LIGHT["coral"] if sig else NAT["text_mid"])

        n_ga = int(sub["n_ga"].iloc[0]) if len(sub) else "?"
        n_gb = int(sub["n_gb"].iloc[0]) if len(sub) else "?"
        ga   = sub["ga"].iloc[0]  if len(sub) else ""
        gb   = sub["gb"].iloc[0]  if len(sub) else ""

        ax.set_title(
            f"{pair_label}\n"
            f"{GROUP_LABEL.get(gb, gb)} (n={n_gb}) vs {GROUP_LABEL.get(ga, ga)} (n={n_ga})",
            fontsize=9, fontweight="bold", color=CB["text_dark"], pad=8, linespacing=1.5,
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(
            [f"{it['item_id']}  {it['criterion']}" for it in items], fontsize=8.5)
        ax.set_xlabel("Cliff's Delta (δ)", fontsize=9, color=CB["text_mid"])
        ax.invert_yaxis()
        ax.set_facecolor(CB["bg"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color=NAT["grid"])
        ax.set_axisbelow(True)

        if ax != axes[0]:
            ax.set_yticklabels([])
            ax.set_ylabel("")

    legend_patches = [
        mpatches.Patch(color=LIGHT["coral"], label="Significant (p_Holm < .05)"),
        mpatches.Patch(color=LIGHT["gold"],  label="Not significant"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02),
               facecolor=NAT["panel_bg"], edgecolor=NAT["grid"])
    return _save(fig, "fig09_effect_significance", out_dir, bg=CB["bg"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate all 9 study figures (PNG + SVG) at 300 DPI + Excel MWU table."
    )
    parser.add_argument("--csv", default=None, help="Path to mergedFinal.csv")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: ./figures_output/)")
    args = parser.parse_args()

    DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data/experiment_survey/mergedFinal.csv"
    DEFAULT_OUT = Path(__file__).resolve().parent.parent / "figures/experiment"
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    out_dir  = Path(args.out) if args.out  else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        print(f"\n  Loading data: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"\n  ⚠  CSV not found at {csv_path}")
        print("     Falling back to synthetic representative data.\n")
        df = make_synthetic_df(n_per_group=60)

    print(f"  Font       : {FONT_FAMILY}")
    print(f"  Data shape : {df.shape}")
    print(f"  Output dir : {out_dir}")
    print(f"  Formats    : PNG (300 DPI) + SVG (vector) + Excel\n")

    print("━━ PART 1: Control vs Treatment A & B ━━━━━━━━━━━━━━━━━")
    print("[1/9] Score delta heatmap (A & B vs Control) …")
    fig01_delta_heatmap_ab(df, out_dir)
    print("[2/9] Investment allocation (Control, A, B) …")
    fig02_investment_ab(df, out_dir)
    print("[3/9] MWU heatmap — one-sided, global Holm (3 pairs) …")
    fig03_mwu_ab(df, out_dir)
    print("[4/9] Spearman ρ (all 4 groups) …")
    fig04_spearman(df, out_dir)

    print("\n━━ PART 2: Treatment C ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[5/9] Score delta heatmap (C vs Control) …")
    fig05_delta_heatmap_c(df, out_dir)
    print("[6/9] Investment allocation (Control vs C) …")
    fig06_investment_c(df, out_dir)
    print("[7/9] MWU heatmap — one-sided, global Holm (A vs C) …")
    fig07_mwu_a_vs_c(df, out_dir)

    print("\n━━ PART 3: Effect Size Figures ━━━━━━━━━━━━━━━━━━━━━━━")
    print("[8/9] Cliff's Delta dot plots (all pairs) …")
    fig08_cliff_delta(df, out_dir)
    print("[9/9] Effect size + significance combined (all pairs) …")
    fig09_effect_significance(df, out_dir)

    print("\n━━ PART 4: Excel MWU Table ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[10/10] Exporting MWU results to Excel …")
    all_pairs = [
        ("Group1", "Group2", "Control vs Treatment A"),
        ("Group1", "Group3", "Control vs Treatment B"),
        ("Group2", "Group3", "Treatment A vs Treatment B"),
        ("Group2", "Group4", "Treatment A vs Treatment C"),
    ]
    mwu_df = compute_mwu_pairs(df, all_pairs)
    export_mwu_excel(mwu_df, out_dir / "mwu_results.xlsx")

    pngs = sorted(out_dir.glob("fig0*.png"))
    svgs = sorted(out_dir.glob("fig0*.svg"))
    print(f"\n  All files saved → {out_dir.resolve()}")
    print(f"  PNG files ({len(pngs)}): " + ", ".join(p.name for p in pngs))
    print(f"  SVG files ({len(svgs)}): " + ", ".join(p.name for p in svgs))
    print(f"  Excel: mwu_results.xlsx")


if __name__ == "__main__":
    main()
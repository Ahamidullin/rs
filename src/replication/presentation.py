"""Build the English defense decks from the archived and computed results."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


INK = "#172332"
MUTED = "#46586D"
BLUE = "#2152A1"
TEAL = "#00857B"
LIGHT = "#EAF1FA"
BG = "#FCFDFF"
DATASETS = ["industry11", "international9", "factors3", "ff21", "ff24", "ff28", "industry52"]
LABELS = ["Industry\n+ market", "Countries\n+ world", "FF3", "FF-1", "FF-4", "FF25\n+ FF3", "49 industries\n+ FF3"]


def page(title, header, source):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.065, 0.933, header, fontsize=16, color=BLUE, weight="bold")
    fig.text(0.065, 0.845, title, fontsize=37, color=INK, weight="bold")
    fig.text(0.065, 0.068, source, fontsize=13, color=MUTED)
    return fig


def label(fig, x, y, value, size=19, color=INK, **kwargs):
    fig.text(x, y, value, fontsize=size, color=color, linespacing=1.45, **kwargs)


def intro():
    fig = page("When combining 1/N with Markowitz helps", "MARKOWITZ MEETS TALMUD",
               "Tu and Zhou (2011), Journal of Financial Economics, 99, pp. 204–215.")
    label(fig, 0.065, 0.73, "Estimation errors in means and covariances can lead\nto large, unstable positions.", 24)
    label(fig, 0.065, 0.53, r"$w=(1-\delta)e+\delta v$", 55, BLUE)
    label(fig, 0.065, 0.41, "e denotes equal weights (1/N)    •    v denotes the estimated portfolio", 21, MUTED)
    label(fig, 0.065, 0.26, "1. Do the blends improve on the estimated rules?\n2. In which datasets do they beat 1/N?", 24, weight="bold")
    return fig


def methods():
    fig = page("Four estimated portfolio rules", "PAPER · METHODS",
               "Tu and Zhou (2011), §2.1–2.5, Equations (1), (7), (12), (15), (18).")
    for i, (abbr, text) in enumerate((
        ("ML", "Sample means and the inverse covariance matrix."),
        ("PJ", "Shrinks means toward the estimated GMV return."),
        ("MP", "Links covariance to the structure of the means."),
        ("KZ", "Combines the tangency and minimum-variance portfolios."),
    )):
        y = 0.71 - i * 0.092
        label(fig, 0.065, y, abbr, 23, BLUE, weight="bold")
        label(fig, 0.145, y, text, 23)
    label(fig, 0.065, 0.23, "Example with δ = 0.25", 21, BLUE, weight="bold")
    label(fig, 0.065, 0.166, "0.75 × (0.5, 0.5) + 0.25 × (1.2, −0.1) = (0.675, 0.35).  Cash = −2.5%.", 21)
    return fig


def simulations():
    fig = page("Blends reduce the cost of estimation error", "PART 2 · SIMULATIONS",
               "Tu and Zhou (2011), Tables 1A and 2B. Team simulations for footnotes 8–9.")
    label(fig, 0.065, 0.732, "Published utility, annual pp · T = 120, γ = 3", 21, MUTED)
    for i, (head, a, b) in enumerate((
        ("Scenario", "1A · One factor", "2B · Three factors"),
        ("1/N", "3.89", "3.85"), ("ML", "−85.72", "−81.09"), ("CKZ", "3.71", "5.09"),
    )):
        y = 0.635 - i * 0.087
        for x, value in zip((0.065, 0.35, 0.64), (head, a, b), strict=True):
            label(fig, x, y, value, 23, BLUE if i == 0 else INK, weight="bold" if i == 0 else "normal")
    label(fig, 0.065, 0.253, "Sensitivity check for N = 50", 20, TEAL, weight="bold")
    label(fig, 0.065, 0.157, "CML, CPJ, and CKZ beat 1/N from T = 480, and CMP from T = 3,000.\n"
          "The sensitivity checks use the team's own specifications.", 19)
    return fig


def empirical_design():
    fig = page("One-month-ahead forecasts across seven datasets", "PART 3 · BACKTEST",
               "Tu and Zhou (2011), §3.3, pp. 211–214, Equation (22).")
    label(fig, 0.065, 0.725, "First forecast for industry11 with M = 120", 21, MUTED)
    label(fig, 0.065, 0.56, "Jul 1963 – Jun 1973\nEstimate weights", 25, BLUE, weight="bold",
          bbox={"facecolor": LIGHT, "edgecolor": LIGHT, "pad": 14})
    label(fig, 0.485, 0.594, "→", 42, BLUE)
    label(fig, 0.60, 0.56, "Jul 1973\nObserve returns", 25, TEAL, weight="bold")
    label(fig, 0.065, 0.36, r"$CER_{\mathrm{annual}}=1200\times(\overline{r}_{\mathrm{excess}}-1.5\,s^2)$", 31, INK)
    label(fig, 0.065, 0.22, "Estimation windows of 120 and 240 months. Annual CER in percentage points.\n"
          "In-sample ML uses the full dataset as a retrospective benchmark.", 22)
    return fig


def empirical_plot(pivot):
    fig = page("CKZ beats 1/N in 10 of 14 comparisons", "PART 3 · TABLE 6",
               "Tu and Zhou (2011), Table 6A/B. CKZ wins 11 of 14 comparisons in the published results.")
    label(fig, 0.065, 0.744, "CER difference between CKZ and 1/N, annual pp", 21, MUTED)
    chart(fig, pivot)
    return fig


def chart(fig, pivot):
    for i, window in enumerate((120, 240)):
        ax = fig.add_axes((0.075 + 0.478 * i, 0.20, 0.402, 0.46))
        part = pivot.xs(window, level="window").reindex(DATASETS)
        x = np.arange(len(DATASETS))
        for j, (method, name, color) in enumerate((("published", "Published", BLUE),
                                                  ("computed", "Replication", TEAL))):
            ax.bar(x + (j - 0.5) * 0.35, part[(method, "ckz")] - part[(method, "ew")],
                   0.34, color=color, label=name)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(x, LABELS, fontsize=11)
        ax.set_title(f"M = {window} months", loc="left", pad=12, fontsize=19, color=INK, weight="bold")
        ax.set_ylim(-16.5, 27)
        ax.set_yticks([-15, -10, -5, 0, 5, 10, 15, 20, 25])
        ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=7)
        ax.set_facecolor(BG)
        ax.grid(axis="y", color="#DDE6EE")
        ax.set_axisbelow(True)
        ax.spines[:].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=12, loc="lower right", bbox_to_anchor=(1, 1.01))


def empirical_limits(matches, total):
    fig = page("Replication checks for all 140 cells", "PART 3 · VALIDATION",
               "See docs/data.md and results/replication for sources, vintages, and paired checks.")
    label(fig, 0.065, 0.698, "CMP50 vs MP", 21, BLUE, weight="bold")
    label(fig, 0.065, 0.595, "14 / 14", 48, BLUE, weight="bold")
    label(fig, 0.51, 0.698, "CMP50 vs 1/N", 21, TEAL, weight="bold")
    label(fig, 0.51, 0.595, "0 / 14", 48, TEAL, weight="bold")
    label(fig, 0.065, 0.465, f"{matches} of {total} cells match within rounding tolerance.", 24, weight="bold")
    label(fig, 0.065, 0.321, "DGU · 10 of 10 matches for 1/N and 8 of 10 for rolling ML.\n"
          "French · Two datasets use FIZ 202412. The original vintages are unavailable.", 21)
    label(fig, 0.065, 0.19, "CKZ and CPJ use the implementation assumptions documented in Part 2.\n"
          "Checks cover data vintages, date ranges, and the jackknife version of CMP.", 21)
    return fig


def main():
    table = pd.read_csv("results/replication/table6.csv")
    pivot = table.pivot(index=["case", "window"], columns="rule", values=["published", "computed"])
    empirical = [empirical_design(), empirical_plot(pivot),
                 empirical_limits(table.status.eq("match").sum(), len(table))]
    overview = [intro(), methods(), simulations()]
    for name, pages in (("slides.pdf", overview + empirical), ("part3_slides.pdf", empirical)):
        with PdfPages(Path("defense") / name) as pdf:
            for fig in pages:
                pdf.savefig(fig, facecolor=BG)
    plt.close("all")


if __name__ == "__main__":
    main()

"""Build four presentation figures from the replication and extension exports."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np
import pandas as pd

from .data import checksum, json_write
from .extensions import DATASETS


WINDOWS = (120, 240)
LABELS = ("Industry", "Intl", "FF3", "FF-1", "FF-4", "FF25", "Indu49")
PALETTE = ("#3567A9", "#408576", "#98537B", "#B3612A", "#6E7399", "#2985AC", "#777C35")
BLUE = "#2463A7"
TEAL = "#00857B"
INK = "#182734"
MUTED = "#526475"
GRID = "#DEE5EB"
PAPER = "#F3F6F8"
RULES = ("ml_is", "ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz")
RULE_LABELS = ("ML in-sample", "1/N", "ML", "PJ", "MP", "KZ", "CML", "CPJ", "CMP50", "CKZ")


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    fig.text(.055, .947, title, fontsize=24, color=INK, weight="bold")
    fig.text(.055, .904, subtitle, fontsize=12, color=MUTED)
    return fig


def finish(fig, out, footer):
    fig.text(.055, .042, footer, fontsize=10, color=MUTED)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def style(ax, *, zero=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(axis="both", labelsize=10, length=0, colors=INK, pad=5)
    if zero:
        ax.axhline(0, linewidth=1, color=MUTED, linestyle=":", zorder=1)


def simulation(comparison, authors_2b, out):
    fig = canvas("When do the blends catch up with 1/N?",
                 "Tu–Zhou, Tables 1A, 2A and 2B · 10,000 replications · γ = 3")
    layout = fig.add_gridspec(2, 3, height_ratios=(3, 1), left=.07, right=.97,
                             bottom=.18, top=.81, hspace=.23, wspace=.30)
    styles = {"true": ("True", INK, "-"), "ew": ("1/N", "#777D86", "-"),
              "ckz": ("CKZ", TEAL, "-"), "cml": ("CML", BLUE, "-"),
              "kz": ("KZ", TEAL, "--"), "ml": ("ML", BLUE, "--")}
    for col, (table, panel, title) in enumerate([
        (1, "A", "1A · One factor, α = 0"),
        (2, "A", "2A · One factor, α ≠ 0"),
        (2, "B", "2B · Three factors, α ≠ 0"),
    ]):
        core = fig.add_subplot(layout[0, col])
        tail = fig.add_subplot(layout[1, col], sharex=core)
        only_published = panel == "B"
        if only_published:
            data = authors_2b.melt(id_vars="T", var_name="rule", value_name="published").rename(columns={"T": "case"})
        else:
            data = comparison.query("table == @table and panel == @panel and metric == 'U'")
        low = -6 if only_published else -4
        core.set_ylim(low, 15.6 if only_published else (7.3 if table == 2 else 5.2))
        tail.set_ylim(-92, 16)
        for rule, (_, color, line) in styles.items():
            points = data.loc[data.rule.eq(rule)].sort_values("case")
            if len(points) != 6 or points.case.nunique() != 6:
                raise ValueError(f"Expected six T values for Table {table}{panel} {rule}")
            x = points.case.to_numpy()
            if only_published:
                core.plot(x, points.published, linestyle=line, color=color, marker="o", markersize=4)
            else:
                core.plot(x, points.computed, linestyle=line, color=color, marker="o", markersize=4)
                core.fill_between(x, points.computed - 1.96 * points.mcse,
                                  points.computed + 1.96 * points.mcse, color=color, alpha=.19)
                if rule not in ("true", "ew"):
                    core.scatter(x, points.published, marker="x", color=color, linewidth=1, s=21, zorder=4)
            if rule == "ml":
                column = "published" if only_published else "computed"
                tail.plot(x, points[column], linestyle="--", marker="o", color=BLUE, markersize=3)
                if not only_published:
                    tail.fill_between(x, points.computed - 1.96 * points.mcse,
                                      points.computed + 1.96 * points.mcse, color=BLUE, alpha=.19)
                    tail.scatter(x, points.published, marker="x", color=BLUE, linewidth=1, s=17)
        core.set_title(title + ("\nPublished values only" if only_published else "\nReplication and published values"),
                       fontsize=13, weight="bold", color=INK, pad=8)
        for ax in (core, tail):
            ax.set_xscale("log")
            ax.set_xlim(105, 6900)
            style(ax, zero=True)
        core.tick_params(axis="x", which="both", labelbottom=False)
        tail.xaxis.set_major_locator(FixedLocator([120, 240, 480, 960, 3000, 6000]))
        tail.xaxis.set_major_formatter(FixedFormatter(["120", "240", "480", "960", "3000", "6000"]))
        tail.xaxis.set_minor_locator(NullLocator())
        tail.set_yticks([-80, -40, 0])
        tail.set_xlabel("T, months (log scale)", fontsize=10)
        tail.text(.97, .13, "ML · full scale", transform=tail.transAxes,
                  ha="right", fontsize=9, color=MUTED)
        if col == 0:
            core.set_ylabel("Expected utility, annual pp", fontsize=11)
    handles = [Line2D([], [], label=name, color=color, ls=line, linewidth=2)
               for name, color, line in styles.values()]
    handles.append(Line2D([], [], label="Published (1A/2A)", color=MUTED, marker="x", ls=""))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .083),
               ncol=7, frameon=False, fontsize=11)
    finish(fig, out, "Shading shows ±1.96 MCSE. True and 1/N match the published values in 1A and 2A after rounding.")


def empirical_gap(table, out):
    fig = canvas("Where does the replication differ from Table 6?",
                 "Left: replicated minus published CER, annual pp (symmetric log scale) · Right: CKZ minus 1/N")
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, .045, .95), left=.11, right=.96,
                           bottom=.18, top=.80, wspace=.35)
    max_gap = max(1, float(table.delta_pp.abs().max()))
    norm = SymLogNorm(linthresh=.03, vmin=-max_gap, vmax=max_gap, base=10)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#D7DDE2")
    for i, window in enumerate(WINDOWS):
        ax = fig.add_subplot(grid[0, i])
        part = table.loc[table.window.eq(window)]
        pivot = part.pivot(index="rule", columns="case", values="delta_pp").reindex(index=RULES, columns=DATASETS)
        image = ax.imshow(pivot.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"M = {window}", fontsize=15, weight="bold", color=INK, pad=11)
        ax.set_xticks(range(len(DATASETS)), [*LABELS[:5], "FF25 ◇", "Indu49 ◇"], rotation=37, ha="right")
        ax.set_yticks(range(len(RULES)), RULE_LABELS if i == 0 else [])
        ax.tick_params(length=0, pad=6, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        values = pivot.stack()
        if len(values):
            for rule, dataset in {values.idxmin(), values.idxmax()}:
                value = pivot.loc[rule, dataset]
                ax.text(DATASETS.index(dataset), RULES.index(rule), f"{value:+.1f}",
                        va="center", ha="center", fontsize=9, weight="bold",
                        color="white" if abs(norm(value) - .5) > .29 else INK)
    ticks = [t for t in [-100, -10, -1, -.1, 0, .1, 1, 10, 100] if abs(t) <= max_gap]
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]), ticks=ticks)
    bar.ax.tick_params(labelsize=9)
    side = fig.add_subplot(grid[0, 3])
    side.set_title("CKZ minus 1/N", fontsize=14, weight="bold", color=INK, pad=11)
    ticks = np.arange(len(DATASETS))
    for i, window in enumerate(WINDOWS):
        sub = table.loc[table.window.eq(window)]
        y = ticks + (i - .5) * .28
        for column, marker, color, name in [("published", "x", MUTED, "Published"),
                                                ("computed", "o", BLUE if i == 0 else TEAL, "Replication")]:
            p = sub.pivot(index="case", columns="rule", values=column).reindex(DATASETS)
            side.scatter(p.ckz - p.ew, y, s=32, marker=marker, color=color,
                         label=f"{window} · {name}")
    style(side)
    side.axvline(0, color=MUTED, linestyle=":", linewidth=1)
    side.set_yticks(ticks, LABELS)
    side.invert_yaxis()
    side.set_xlabel("CER difference, annual pp", fontsize=10)
    side.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.48, -.12), ncol=2)
    matches = int(table.status.eq("match").sum())
    finish(fig, out, f"{matches}/{len(table)} cells within 0.005 pp · ◇ French FIZ 202412. The other five datasets come from the DGU archive.")


def turnover(summary, out):
    fig = canvas("Does reducing CKZ turnover pay off?",
                 "Net CER difference relative to standard CKZ, with equal trading costs per unit of turnover")
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=.08, right=.97, top=.82, bottom=.23, wspace=.13)
    paired = summary.loc[summary.control.eq("ckz")]
    for ax, window in zip(axes, WINDOWS, strict=True):
        data = paired.loc[paired.window.eq(window)]
        for dataset, label, color in zip(DATASETS, LABELS, PALETTE, strict=True):
            series = data.loc[data.dataset.eq(dataset)].sort_values("cost_bps")
            ax.plot(series.cost_bps, series.delta_net_cer_pp, color=color, marker="o",
                    markersize=4, linewidth=1.2, alpha=.83, label=label)
        for cost, missing in data.loc[~data.complete].groupby("cost_bps"):
            for i, dataset in enumerate(missing.dataset):
                offset = 1.25 * (i - (len(missing) - 1) / 2)
                ax.scatter(cost + offset, .024, marker="x", color=PALETTE[DATASETS.index(dataset)],
                           transform=ax.get_xaxis_transform(), zorder=4)
        med = data.groupby("cost_bps").delta_net_cer_pp.median()
        ax.plot(med.index, med, marker="D", markersize=5, color=INK, linewidth=2.3, label="Median")
        m25 = data.loc[data.cost_bps.eq(25) & data.complete]
        note = (f"Turnover at 25 bps · median {m25.control_turnover.median():.2f} → "
                f"{m25.turnover.median():.2f} NAV" if len(m25) else "No complete pair at 25 bps")
        ax.text(.02, 1.015, note, transform=ax.transAxes, fontsize=10, color=MUTED)
        ax.set_title(f"M = {window}", fontsize=15, weight="bold", color=INK, pad=26)
        ax.set_xticks(med.index)
        ax.set_xlabel("Trading cost, bps per unit of turnover")
        style(ax, zero=True)
    axes[0].set_ylabel("Δ net CER, annual pp")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .096),
               ncol=8, frameon=False, fontsize=10)
    foot = "Turnover sums absolute changes in risky weights. At zero cost the rule reproduces CKZ exactly."
    if (~paired.complete).any():
        foot += "  × Incomplete pair."
    finish(fig, out, foot)


def stability(summary, groups, out):
    fig = canvas("Does additional shrinkage improve CKZ?",
                 "Paired gross CER differences at zero cost · Right: differences vs CKZ by monthly score group")
    grid = fig.add_gridspec(2, 2, width_ratios=(2.9, 1.1), left=.075, right=.97,
                           bottom=.16, top=.82, hspace=.45, wspace=.20)
    for i, window in enumerate(WINDOWS):
        ax = fig.add_subplot(grid[i, 0])
        panel = summary.query("window == @window and cost_bps == 0")
        for control, offset, color, marker in [("ckz", -.13, TEAL, "o"),
                                               ("ckz50", .13, BLUE, "D")]:
            pair = panel.loc[panel.control.eq(control)].set_index("dataset").reindex(DATASETS)
            y = pair.delta_gross_cer_pp.to_numpy()
            x = np.arange(len(DATASETS)) + offset
            ax.vlines(x, pair.gross_ci_lo_pp, pair.gross_ci_hi_pp, color=color, alpha=.58, linewidth=1.5)
            ax.scatter(x, y, color=color, marker=marker, label=f"vs {control.upper()}", zorder=4)
            missing = np.isnan(y)
            ax.scatter(x[missing], np.full(missing.sum(), .06), transform=ax.get_xaxis_transform(),
                       marker="x", color=color)
        ax.set_xticks(range(len(DATASETS)), LABELS)
        ax.set_xlim(-.45, len(LABELS) - .55)
        ax.set_title(f"M = {window}", loc="left", fontsize=14, weight="bold", color=INK)
        ax.set_ylabel("Δ gross CER, annual pp")
        style(ax, zero=True)
        if i == 0:
            ax.legend(loc="best", frameon=False, ncol=2, fontsize=10)
        tab = fig.add_subplot(grid[i, 1])
        tab.axis("off")
        tab.set_title("High and low s · Δ CER (n)", fontsize=12, loc="left", color=INK)
        sub = groups.query("window == @window and cost_bps == 0 and control == 'ckz'").set_index(["dataset", "group"])
        cells = []
        for dataset, label in zip(DATASETS, LABELS, strict=True):
            row = [label]
            for group in ("high", "low"):
                result = sub.loc[(dataset, group)]
                value = result.delta_gross_cer_pp
                row.append(f"{value:+.2f} ({result.n})" if np.isfinite(value) else f"— ({result.n})")
            cells.append(row)
        table = tab.table(cellText=cells, colLabels=["Dataset", "High", "Low"], cellLoc="left",
                          colWidths=[.25, .37, .37], bbox=[0, -.04, 1, .98])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(0)
            cell.set_facecolor(PAPER if r % 2 == 0 else "white")
            cell.set_text_props(color=INK, weight="bold" if r == 0 or c == 0 else "normal")
    foot = "95% paired circular block bootstrap intervals. n is the number of months in each score group."
    if (~summary.loc[summary.cost_bps.eq(0)].complete).any():
        foot += "  × Incomplete pair."
    finish(fig, out, foot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication", type=Path, default=Path("results/replication"))
    parser.add_argument("--extensions", type=Path, default=Path("results/extensions"))
    parser.add_argument("--simulation", type=Path, default=Path("results/sim_full/tables1_3_comparison.csv"))
    parser.add_argument("--published-2b", type=Path, default=Path("references/table2b_figure.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "simulation": args.simulation,
        "published_2b": args.published_2b,
        "table6": args.replication / "table6.csv",
        "turnover": args.extensions / "turnover" / "summary.csv",
        "stability": args.extensions / "stability" / "summary.csv",
        "groups": args.extensions / "stability" / "groups.csv",
    }
    tables = {key: pd.read_csv(path) for key, path in paths.items()}
    for extension in ("turnover", "stability"):
        manifest = json.loads((args.extensions / extension / "manifest.json").read_text())
        keys = ("stability", "groups") if extension == "stability" else ("turnover",)
        for key in keys:
            if (set(tables[key].run_id) != {manifest["run_id"]}
                    or checksum(paths[key]) != manifest["output_sha256"][paths[key].name]):
                raise ValueError(f"{key}: figure input differs from the extension manifest")
        if checksum(paths["table6"]) != manifest["baseline_sha256"]["table6.csv"]:
            raise ValueError(f"{extension}: the supplied Table 6 differs from the extension baseline")
    args.out.mkdir(parents=True, exist_ok=True)
    figures = [
        ("01_simulation.png", simulation, ("simulation", "published_2b")),
        ("02_empirical_gap.png", empirical_gap, ("table6",)),
        ("03_turnover.png", turnover, ("turnover",)),
        ("04_stability.png", stability, ("stability", "groups")),
    ]
    for name, function, keys in figures:
        function(*(tables[key] for key in keys), args.out / name)
    json_write(args.out / "manifest.json", {
        "renderer": matplotlib.__version__, "code_sha256": checksum(Path(__file__)),
        "source_sha256": {str(path): checksum(path) for path in paths.values()},
        "output_sha256": {name: checksum(args.out / name) for name, _, _ in figures},
        "run_ids": {key: tables[key].run_id.unique().tolist()
                    for key in ("table6", "turnover", "stability")},
    })


if __name__ == "__main__":
    main()

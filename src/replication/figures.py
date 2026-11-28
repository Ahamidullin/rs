"""Build the two main-paper replication figures."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np
import pandas as pd
from .data import checksum, json_write
from .extensions import DATASETS

WINDOWS = (120, 240)
LABELS = ("Industry", "Intl", "FF3", "FF-1", "FF-4", "FF25", "Indu49")
PALETTE = ("#8065A8", "#B75D89", "#9E97A6", "#61576B", "#A783B5", "#C38DAA", "#706C79")
BLUE = "#8065A8"
TEAL = "#B75D89"
INK = "#3E3C46"
MUTED = "#706C79"
GRID = "#E4E1E9"
PAPER = "#F7F5FA"
RULES = ("ml_is", "ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz")
RULE_LABELS = (
    "ML in-sample",
    "1/N",
    "ML",
    "PJ",
    "MP",
    "KZ",
    "CML",
    "CPJ",
    "CMP50",
    "CKZ",
)


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor="white")
    fig.text(0.055, 0.947, title, fontsize=24, color=INK, weight="bold")
    fig.text(0.055, 0.904, subtitle, fontsize=12, color=MUTED)
    return fig


def finish(fig, out, footer):
    fig.text(0.055, 0.042, footer, fontsize=10, color=MUTED)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def style(ax, *, zero=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(axis="both", labelsize=10, length=0, colors=INK, pad=5)
    if zero:
        ax.axhline(0, linewidth=1, color=MUTED, linestyle=":", zorder=1)


def simulation(comparison, authors_2b, out, panel_dir=None):
    repetitions = comparison.loc[comparison.n.gt(0), "n"].unique()
    rep_label = str(int(repetitions[0])) if len(repetitions) == 1 else "see manifest"
    fig = canvas(
        "When do the blends catch up with 1/N?",
        f"Tu–Zhou, Tables 1A, 2A and 2B · {rep_label} replications · γ = 3",
    )
    layout = fig.add_gridspec(
        2,
        3,
        height_ratios=(3, 1),
        left=0.07,
        right=0.97,
        bottom=0.18,
        top=0.81,
        hspace=0.23,
        wspace=0.3,
    )
    styles = {
        "true": ("True", INK, "-"),
        "ew": ("1/N", "#777D86", "-"),
        "ckz": ("CKZ", TEAL, "-"),
        "cml": ("CML", BLUE, "-"),
        "kz": ("KZ", TEAL, "--"),
        "ml": ("ML", BLUE, "--"),
    }
    panel_axes = []
    for col, (table, panel, title) in enumerate(
        [
            (1, "A", "1A · One factor, α = 0"),
            (2, "A", "2A · One factor, α ≠ 0"),
            (2, "B", "2B · Three factors, α ≠ 0"),
        ]
    ):
        core = fig.add_subplot(layout[0, col])
        tail = fig.add_subplot(layout[1, col], sharex=core)
        available = comparison.query(
            "table == @table and panel == @panel and metric == 'U'"
        )
        only_published = available.empty or not available.computed.notna().any()
        if only_published:
            data = authors_2b.melt(
                id_vars="T", var_name="rule", value_name="published"
            ).rename(columns={"T": "case"})
        else:
            data = available
        value_columns = ["published"] if only_published else ["computed", "published"]
        central = data.loc[data.rule.ne("ml"), value_columns].to_numpy()
        central = central[np.isfinite(central)]
        core.set_ylim(min(-4, central.min() - 1), max(5.2, central.max() + 1))
        ml_values = data.loc[data.rule.eq("ml"), value_columns].to_numpy()
        tail.set_ylim(
            min(-92, np.nanmin(ml_values) - 5), max(16, np.nanmax(ml_values) + 2)
        )
        for rule, (_, color, line) in styles.items():
            points = data.loc[data.rule.eq(rule)].sort_values("case")
            if len(points) != 6 or points.case.nunique() != 6:
                raise ValueError(
                    f"Expected six T values for Table {table}{panel} {rule}"
                )
            x = points.case.to_numpy()
            if only_published:
                core.plot(
                    x,
                    points.published,
                    linestyle=line,
                    color=color,
                    marker="o",
                    markersize=4,
                )
            else:
                core.plot(
                    x,
                    points.computed,
                    linestyle=line,
                    color=color,
                    marker="o",
                    markersize=4,
                )
                core.fill_between(
                    x,
                    points.computed - 1.96 * points.mcse,
                    points.computed + 1.96 * points.mcse,
                    color=color,
                    alpha=0.19,
                )
                if rule not in ("true", "ew"):
                    core.scatter(
                        x,
                        points.published,
                        marker="x",
                        color=color,
                        linewidth=1,
                        s=21,
                        zorder=4,
                    )
            if rule == "ml":
                column = "published" if only_published else "computed"
                tail.plot(
                    x,
                    points[column],
                    linestyle="--",
                    marker="o",
                    color=BLUE,
                    markersize=3,
                )
                if not only_published:
                    tail.fill_between(
                        x,
                        points.computed - 1.96 * points.mcse,
                        points.computed + 1.96 * points.mcse,
                        color=BLUE,
                        alpha=0.19,
                    )
                    tail.scatter(
                        x, points.published, marker="x", color=BLUE, linewidth=1, s=17
                    )
        core.set_title(
            title
            + (
                "\nPublished values only"
                if only_published
                else "\nReplication and published values"
            ),
            fontsize=13,
            weight="bold",
            color=INK,
            pad=8,
        )
        for ax in (core, tail):
            ax.set_xscale("log")
            ax.set_xlim(105, 6900)
            style(ax, zero=True)
        core.tick_params(axis="x", which="both", labelbottom=False)
        tail.xaxis.set_major_locator(FixedLocator([120, 240, 480, 960, 3000, 6000]))
        tail.xaxis.set_major_formatter(
            FixedFormatter(["120", "240", "480", "960", "3000", "6000"])
        )
        tail.xaxis.set_minor_locator(NullLocator())
        tail.set_yticks([-80, -40, 0])
        tail.set_xlabel("T, months (log scale)", fontsize=10)
        tail.text(
            0.97,
            0.13,
            "ML · full scale",
            transform=tail.transAxes,
            ha="right",
            fontsize=9,
            color=MUTED,
        )
        if col == 0:
            core.set_ylabel("Expected utility, annual pp", fontsize=11)
        panel_axes.append((table, panel, core, tail))
    handles = [
        Line2D([], [], label=name, color=color, ls=line, linewidth=2)
        for name, color, line in styles.values()
    ]
    handles.append(Line2D([], [], label="Published", color=MUTED, marker="x", ls=""))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.083),
        ncol=7,
        frameon=False,
        fontsize=11,
    )
    if panel_dir is not None:
        from matplotlib.transforms import Bbox
        panel_dir.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for table, panel, core, tail in panel_axes:
            bounds = Bbox.union([core.get_tightbbox(renderer), tail.get_tightbbox(renderer)])
            fig.savefig(panel_dir / f"simulation_{table}{panel}.png", dpi=180,
                        bbox_inches=bounds.transformed(fig.dpi_scale_trans.inverted()), pad_inches=.08)
            for label, ax in (("core", core), ("ml", tail)):
                if label == "core":
                    core.tick_params(axis="x", which="major", labelbottom=True)
                    core.set_xlabel("T, months (log scale)", fontsize=10)
                    fig.canvas.draw()
                one = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(panel_dir / f"simulation_{table}{panel}_{label}.png", dpi=180,
                            bbox_inches=one, pad_inches=.08)
                if label == "core":
                    core.tick_params(axis="x", which="both", labelbottom=False)
                    core.set_xlabel("")
    finish(
        fig,
        out,
        "Shading shows ±1.96 MCSE. Population reconstruction and data vintage are recorded in the simulation manifest.",
    )


def empirical_gap(table, out):
    fig = canvas(
        "Historical replication: comparison with Table 6",
        "Left: replicated minus published CER, annual pp (symmetric log scale) · Right: CKZ minus 1/N",
    )
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1, 1, 0.045, 0.95),
        left=0.11,
        right=0.96,
        bottom=0.18,
        top=0.8,
        wspace=0.35,
    )
    max_gap = max(1, float(table.delta_pp.abs().max()))
    norm = SymLogNorm(linthresh=0.03, vmin=-max_gap, vmax=max_gap, base=10)
    cmap = LinearSegmentedColormap.from_list("lavender_pink", ["#8065A8", "#FFFFFF", "#B75D89"]).with_extremes(bad="#E4E1E9")
    for i, window in enumerate(WINDOWS):
        ax = fig.add_subplot(grid[0, i])
        part = table.loc[table.window.eq(window)]
        pivot = part.pivot(index="rule", columns="case", values="delta_pp").reindex(
            index=RULES, columns=DATASETS
        )
        image = ax.imshow(pivot.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"M = {window}", fontsize=15, weight="bold", color=INK, pad=11)
        ax.set_xticks(
            range(len(DATASETS)),
            [*LABELS[:5], "FF25 ◇", "Indu49 ◇"],
            rotation=37,
            ha="right",
        )
        ax.set_yticks(range(len(RULES)), RULE_LABELS if i == 0 else [])
        ax.tick_params(length=0, pad=6, labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        values = pivot.stack()
        if len(values):
            for rule, dataset in {values.idxmin(), values.idxmax()}:
                value = pivot.loc[rule, dataset]
                ax.text(
                    DATASETS.index(dataset),
                    RULES.index(rule),
                    f"{value:+.1f}",
                    va="center",
                    ha="center",
                    fontsize=9,
                    weight="bold",
                    color="white" if abs(norm(value) - 0.5) > 0.29 else INK,
                )
    ticks = [t for t in [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100] if abs(t) <= max_gap]
    bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]), ticks=ticks)
    bar.ax.tick_params(labelsize=9)
    side = fig.add_subplot(grid[0, 3])
    side.set_title("CKZ minus 1/N", fontsize=14, weight="bold", color=INK, pad=11)
    ticks = np.arange(len(DATASETS))
    for i, window in enumerate(WINDOWS):
        sub = table.loc[table.window.eq(window)]
        y = ticks + (i - 0.5) * 0.28
        for column, marker, color, name in [
            ("published", "x", MUTED, "Published"),
            ("computed", "o", BLUE if i == 0 else TEAL, "Replication"),
        ]:
            p = sub.pivot(index="case", columns="rule", values=column).reindex(DATASETS)
            side.scatter(
                p.ckz - p.ew,
                y,
                s=32,
                marker=marker,
                color=color,
                label=f"{window} · {name}",
            )
    style(side)
    side.axvline(0, color=MUTED, linestyle=":", linewidth=1)
    side.set_yticks(ticks, LABELS)
    side.invert_yaxis()
    side.set_xlabel("CER difference, annual pp", fontsize=10)
    side.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.12),
        ncol=2,
    )
    matches = int(table.status.eq("match").sum())
    finish(
        fig,
        out,
        f"{matches}/{len(table)} cells within 0.005 pp · ◇ French FIZ 202412. The other five datasets come from the DGU archive.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build the two main-paper replication figures"
    )
    parser.add_argument("--replication", type=Path, default=Path("results/article/empirical"))
    parser.add_argument(
        "--simulation",
        type=Path,
        default=Path("results/article/simulation/tables1_5_comparison.csv"),
    )
    parser.add_argument(
        "--published-2b", type=Path, default=None
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "simulation": args.simulation,
        "table6": args.replication / "table6.csv",
    }
    tables = {key: pd.read_csv(path) for key, path in paths.items()}
    if args.published_2b is not None:
        paths["published_2b"] = args.published_2b
        tables["published_2b"] = pd.read_csv(args.published_2b)
    else:
        panel = tables["simulation"].query("table == 2 and panel == 'B' and metric == 'U'")
        tables["published_2b"] = panel.pivot(index="case", columns="rule", values="published").rename_axis("T").reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    simulation(
        tables["simulation"], tables["published_2b"], args.out / "01_simulation.png"
    )
    empirical_gap(tables["table6"], args.out / "02_empirical_gap.png")
    json_write(
        args.out / "manifest.json",
        dict(
            renderer=matplotlib.__version__,
            code_sha256=checksum(Path(__file__)),
            source_sha256={str(p): checksum(p) for p in paths.values()},
            output_sha256={
                name: checksum(args.out / name)
                for name in ("01_simulation.png", "02_empirical_gap.png")
            },
            run_ids={"table6": tables["table6"].run_id.unique().tolist()},
        ),
    )


if __name__ == "__main__":
    main()

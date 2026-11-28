"""Rebuild publication figures from compact CSV results; never run experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replication.data import checksum, json_write
from replication.figures import empirical_gap, simulation

ROOT = Path(__file__).resolve().parents[1]
INK, PURPLE, PINK, GRID = "#3E3C46", "#8065A8", "#B75D89", "#E4E1E9"


def build(results, out, panel_dir=None):
    out.mkdir(parents=True, exist_ok=True)
    inputs = [results / "article/simulation/tables1_5_comparison.csv",
              results / "article/empirical/table6.csv", results / "summary.csv"]
    comparison, table, summary = (pd.read_csv(p) for p in inputs)
    # The simulation renderer uses this fallback only when a panel has no computed values.
    published = comparison.loc[comparison.table.eq(2) & comparison.panel.eq("B") & comparison.metric.eq("U")]
    published = published.pivot(index="case", columns="rule", values="published").rename_axis("T").reset_index()
    simulation(comparison, published, out / "01_simulation.png", panel_dir=panel_dir)
    empirical_gap(table, out / "02_empirical_gap.png")
    titles = dict(regularization="Historical regularization: all models",
                  dimension_window="Assets and estimation windows: all models",
                  markets="Developed and Emerging markets: all models")
    for study, title in titles.items():
        rows = summary.loc[summary.study.eq(study)].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
        values = rows.median_delta_net_cer_pp.to_numpy()
        y = np.arange(len(rows))
        ax.barh(y, values, color=np.where(values >= 0, PURPLE, PINK), height=.64)
        ax.set(yticks=y, yticklabels=rows.model.str.upper(), xlabel="Median change in annual net CER (percentage points)",
               title=title)
        ax.invert_yaxis()
        ax.axvline(0, color=INK, linewidth=.8)
        ax.grid(axis="x", color=GRID);ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        pad = max(np.abs(values).max() * .02, .01)
        for i, row in rows.iterrows():
            label = f"{row.median_delta_net_cer_pp:+.2f}   [{row.complete_pairs}/{row.expected_pairs} pairs]"
            ax.text(row.median_delta_net_cer_pp + (pad if row.median_delta_net_cer_pp >= 0 else -pad),
                    i, label, va="center", ha="left" if row.median_delta_net_cer_pp >= 0 else "right", fontsize=10, color=INK)
        lo, hi = ax.get_xlim();span = hi-lo;ax.set_xlim(lo-span*.38, hi+span*.52)
        fig.supxlabel("Regularized minus own original, costs 25 bps. Medians use complete pairs; counts in brackets.", fontsize=10)
        fig.savefig(out / f"{study}_all_models.png", dpi=150)
        fig.savefig(out / f"{study}_all_models.svg")
        plt.close(fig)
    from replication.all_model_studies import make_plots
    grid_path = results / "extensions/dimension_window/aggregate.csv"
    make_plots(out, pd.read_csv(grid_path), "dimension_window", panel_dir=panel_dir)
    inputs.append(grid_path)
    # Strip insignificant SVG path-line padding for a clean source-control diff.
    for svg in out.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    outputs = {p.name: checksum(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "manifest.json"}
    json_write(out / "manifest.json", dict(
        inputs_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): checksum(p) for p in inputs},
        renderer=matplotlib.__version__, code_sha256=checksum(Path(__file__)), outputs_sha256=outputs))
    print(f"Saved six publication figures to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--panel-dir", type=Path, help="Optional enlarged original plot panels for a talk")
    args = parser.parse_args()
    build(args.results.resolve(), args.out.resolve(), args.panel_dir.resolve() if args.panel_dir else None)

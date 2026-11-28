"""Export compact, verified completed results without fitting any model."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz", "cmp50")
STUDIES = ("regularization", "dimension_window", "markets")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(article, studies, out):
    """Copy exact saved artifacts and derive the 27-row all-model summary."""
    provenance = {}

    def copy(src, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        provenance[relative.as_posix()] = dict(source=str(src.relative_to(ROOT)).replace("\\", "/"), sha256=sha(src))

    simulation = article / "simulation"
    spec = json.loads((simulation / "sim_manifest.json").read_text())
    if spec["status"] != "complete" or spec["spec"]["reps"] != 10000:
        raise ValueError("Publication requires the completed 10,000-replication run")
    copy(simulation / "sim_manifest.json", Path("article/simulation/execution_manifest.json"))
    for source in sorted((simulation / "tables").glob("*.csv")):
        copy(source, Path("article/simulation") / source.name)
    baseline = article / "replication"
    empirical = json.loads((baseline / "empirical_manifest.json").read_text())
    if empirical["mode"] != "full" or empirical["ckz_mode"] != "eq13_scaled":
        raise ValueError("Require the full Eq. (13) baseline")
    for name, expected in empirical["output_sha256"].items():
        if sha(baseline / name) != expected:
            raise ValueError(f"Empirical output checksum mismatch: {name}")
    copy(baseline / "empirical_manifest.json", Path("article/empirical/execution_manifest.json"))
    for name in ("table6.csv", "empirical_checks.csv"):
        copy(baseline / name, Path("article/empirical") / name)

    records = []
    for study in STUDIES:
        source = studies / study
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete study: {study}")
        # Validate the complete source run, including the large locally retained paths.
        for name, expected in manifest["output_sha256"].items():
            if sha(source / name) != expected:
                raise ValueError(f"Study checksum mismatch: {study}/{name}")
        for name in ("aggregate.csv", "summary.csv", "scores.csv", "cases.json"):
            copy(source / name, Path("extensions") / study / name)
        copy(source / "manifest.json", Path("extensions") / study / "execution_manifest.json")
        frame = pd.read_csv(source / "summary.csv")
        primary = frame.loc[frame.period.eq("full") & frame.cost_bps.eq(25)]
        for model in MODELS:
            pairs = primary.loc[primary.comparison.eq("regularization_effect") & primary.control.eq(model)]
            complete = pairs.loc[pairs.complete]
            ew = primary.loc[primary.comparison.eq("versus_ew") & primary.variant.eq("lw_" + model) & primary.complete]
            records.append(dict(study=study, model=model, expected_pairs=len(pairs),
                complete_pairs=len(complete), wins_vs_original=int(complete.delta_net_cer_pp.gt(0).sum()),
                median_delta_net_cer_pp=complete.delta_net_cer_pp.median(),
                positive_95_intervals=int(complete.net_ci_lo_pp.gt(0).sum()),
                complete_vs_ew=len(ew), wins_vs_ew=int(ew.delta_net_cer_pp.gt(0).sum())))
    pd.DataFrame(records).to_csv(out / "summary.csv", index=False)
    (out / "publication_manifest.json").write_text(json.dumps(dict(
        description="Compact completed-run export; no models were refitted",
        original_execution_manifests="Record original run hashes, including large files not distributed here",
        files=provenance, summary_sha256=sha(out / "summary.csv")), indent=2) + "\n")
    print(f"Exported completed article results and all three studies to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", type=Path, default=ROOT / "runs/article_gpu_v1")
    parser.add_argument("--studies", type=Path, default=ROOT / "runs/all_models_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    build(args.article.resolve(), args.studies.resolve(), args.out.resolve())

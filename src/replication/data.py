"""Prepare monthly excess-return panels from the DGU and French archives."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
import tomllib
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


DGU_ZIP = "DGU_Replication_Code_and_Data.zip"
DGU_ROOT = "DGU_Replication_Code_and_Data/Data/"
DGU_URL = (
    "https://www.dropbox.com/scl/fi/lx8v8wre2i9to79gfnmd1/"
    "DGU_Replication_Code_and_Data.zip?rlkey=yrp2vf99wje7l2xe6rcfi7zuu&dl=1"
)
DGU_SHA256 = "d7a1fffc774c0fddea80dead8a1832a57704df769f0277ddf22c5170ca0b2213"
FRENCH_ROOT = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french"
FRENCH_PATH = {"fiz": "ftp_202412", "ciz": "ftp"}
FF3_ZIP = "F-F_Research_Data_Factors_CSV.zip"
FACTORS = ["Mkt-RF", "SMB", "HML"]


def load_config(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def read_member(path, url, member=None, download=False, expected_hash=None):
    fetched = path.with_suffix(".download.json")
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing. Run with --download or supply the archive.")
        with urlopen(url, timeout=60) as response:
            payload = response.read()
        if payload[:2] != b"PK":
            raise ValueError(f"{url} did not return a ZIP archive")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        json_write(fetched, {"downloaded_at_utc": datetime.now(timezone.utc).isoformat()})

    digest = checksum(path)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest}")

    with ZipFile(path) as archive:
        if member is None:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in {path}, got {members}")
            member = members[0]
        contents = archive.read(member).decode("latin-1")
    info = {"archive": path.name, "url": url, "member": member, "sha256": digest}
    if fetched.exists():
        info.update(json.loads(fetched.read_text()))
    return contents, info


def month_label(value):
    return value[:4] + "-" + value[4:6]


def parse_dgu(contents, spec):
    lines = [line for line in contents.splitlines() if line.strip()]
    split = str.split if spec.get("space_delimited") else lambda line: line.split("\t")
    header = [name.strip() for name in split(lines[0])]
    records = []
    months = []
    for line in lines[1:]:
        fields = split(line)
        date = fields[0].strip()
        if not re.fullmatch(r"\d{8}", date):
            raise ValueError(f"Invalid DGU date: {date!r}")
        month = month_label(date)
        if not spec["start"] <= month <= spec["end"]:
            continue
        if len(fields) != len(header):
            raise ValueError(f"{spec['member']} {month}: {len(fields)} columns, expected {len(header)}")
        months.append(month)
        records.append([float(field) if field.strip() else np.nan for field in fields[1:]])
    raw = pd.DataFrame(records, index=months, columns=header[1:])
    raw = raw.rename(columns={header[1]: "rf"})
    return raw, header


def parse_french(contents, weighting=None):
    lines = contents.splitlines()
    if weighting is None:
        start = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    else:
        label = f"average {weighting} weighted returns -- monthly"
        start = next(i + 1 for i, line in enumerate(lines) if line.strip().lower() == label)
    block = [lines[start]]
    for line in lines[start + 1:]:
        if not re.match(r"^\s*\d{6}\s*,", line):
            break
        block.append(line)
    rows = csv.reader(StringIO("\n".join(block)))
    header = [name.strip() for name in next(rows)][1:]
    months = []
    values = []
    for date, *fields in rows:
        month = month_label(date.strip())
        if len(fields) != len(header):
            raise ValueError(f"French {weighting or 'FF3'} {month}: wrong number of columns")
        months.append(month)
        values.append([float(field) if field.strip() else np.nan for field in fields])
    frame = pd.DataFrame(values, index=months, columns=header)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.columns.is_unique:
        raise ValueError("French archive must contain unique, ordered months and unique series")
    return frame.replace([-99.99, -999.0], np.nan) / 100, header


def french_panel(raw, portfolio_zip, vintage, weighting, download=False):
    directory = raw / "french" / vintage
    url = f"{FRENCH_ROOT}/{FRENCH_PATH[vintage]}"
    frames = []
    metadata = []
    for filename, section in ((portfolio_zip, weighting), (FF3_ZIP, None)):
        contents, info = read_member(directory / filename, f"{url}/{filename}", download=download)
        frame, header = parse_french(contents, section)
        frames.append(frame)
        metadata.append({
            **info,
            "release": re.search(r"(\d{6}) CRSP database", contents).group(1),
            "header": header,
            "first_month": frame.index[0],
            "last_month": frame.index[-1],
            "weighting": section,
        })
    portfolios, factors = frames
    aligned = portfolios.join(factors[["RF", *FACTORS]], how="inner", validate="one_to_one")
    aligned[portfolios.columns] = aligned[portfolios.columns].sub(aligned["RF"], axis="index")
    aligned = aligned.rename(columns={"RF": "rf"})
    coverage = {
        "portfolios_only": portfolios.index.difference(factors.index).tolist(),
        "factors_only": factors.index.difference(portfolios.index).tolist(),
        "first_complete_month": aligned.dropna().index[0],
        "missing_months_by_series": {
            col: aligned.index[aligned[col].isna()].tolist()
            for col in aligned if aligned[col].isna().any()
        },
    }
    return aligned, metadata, coverage


def select_panel(frame, start, end, n, label):
    expected = pd.period_range(start, end, freq="M").astype(str)
    selected = frame.loc[start:end].copy()
    if not np.array_equal(selected.index, expected):
        missing = expected.difference(selected.index).tolist()
        raise ValueError(f"{label}: months out of order, repeated or missing: {missing}")
    if len(selected.columns) != n + 1 or not selected.columns.is_unique or "rf" not in selected:
        raise ValueError(f"{label}: expected RF and {n} unique asset columns")
    for col in selected:
        invalid = selected.index[~np.isfinite(selected[col])].tolist()
        if invalid:
            raise ValueError(f"{label} {col}: invalid values at {invalid}")
    assets = [col for col in selected if col != "rf"]
    panel = selected[["rf", *assets]].rename(columns={name: "asset_" + name for name in assets})
    return panel.rename_axis("month").reset_index()


def prepare(config, raw, out, download=False):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "units": {"rf": "monthly fraction", "asset_*": "monthly excess return fraction"},
        "datasets": {},
    }
    for dataset, spec in config["dataset"].items():
        if "member" in spec:
            contents, info = read_member(raw / "dgu" / DGU_ZIP, DGU_URL, DGU_ROOT + spec["member"],
                                         download, DGU_SHA256)
            frame, header = parse_dgu(contents, spec)
            sources, vintage, coverage = [{**info, "header": header}], "dgu", None
        else:
            vintage = spec["vintage"]
            frame, sources, coverage = french_panel(raw, spec["portfolio_zip"], vintage,
                                                     spec["weighting"], download)
        panel = select_panel(frame, spec["start"], spec["end"], spec["n"], f"{dataset}/{vintage}")
        target = out / f"{dataset}.parquet"
        panel.to_parquet(target, index=False)
        manifest["datasets"][dataset] = {
            "vintage": vintage,
            "weighting": spec.get("weighting"),
            "start": spec["start"],
            "end": spec["end"],
            "T": len(panel),
            "N": spec["n"],
            "assets": [col.removeprefix("asset_") for col in panel.columns[2:]],
            "sources": sources,
            "coverage": coverage,
            "parquet_sha256": checksum(target),
        }
        print(f"{dataset}: {spec['start']} to {spec['end']}, T={len(panel)}, N={spec['n']}", flush=True)
    json_write(out / "empirical_data_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/empirical.toml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--download", action="store_true", help="Fetch missing source ZIPs")
    args = parser.parse_args()
    prepare(load_config(args.config), args.raw, args.out, args.download)


if __name__ == "__main__":
    main()

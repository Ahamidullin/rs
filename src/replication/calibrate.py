"""FF3 simulation calibration from Tu--Zhou section 3.1, July 1963--August 2007."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


HISTORICAL_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/Data_Library/"
    "Historical_Archives/08%202008%20Update/ftp/F-F_Research_Data_Factors_TXT.zip"
)


def calibrate(archive, output, download=False):
    archive, output = Path(archive), Path(output)
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"Missing {archive}; use --download")
        payload = urlopen(HISTORICAL_URL, timeout=60).read()
        if not payload.startswith(b"PK"):
            raise ValueError("Expected a factor ZIP")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(payload)
    with ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))]
        if len(members) != 1:
            raise ValueError("Expected one factor data file")
        text = z.read(members[0]).decode("latin-1")
    months, rows = [], []
    for line in text.splitlines():
        fields = re.split(r"[\s,]+", line.strip())
        if len(fields) != 5 or not re.fullmatch(r"\d{6}", fields[0]):
            continue
        if "196307" <= fields[0] <= "200708":
            months.append(fields[0])
            rows.append([float(value) / 100 for value in fields[1:4]])
    expected = [
        f"{year}{month:02d}"
        for year in range(1963, 2008)
        for month in range(1, 13)
        if "196307" <= f"{year}{month:02d}" <= "200708"
    ]
    X = np.asarray(rows)
    if months != expected or X.shape != (530, 3) or not np.isfinite(X).all():
        raise ValueError("Need 530 consecutive finite monthly FF3 observations")
    covariance = np.cov(X, rowvar=False, ddof=1)
    np.linalg.cholesky(covariance)
    value = {
        "source": HISTORICAL_URL,
        "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
        "member": members[0],
        "period": [months[0], months[-1]],
        "n": len(X),
        "factors": ["Mkt-RF", "SMB", "HML"],
        "units": "monthly fractions",
        "covariance_ddof": 1,
        "mean": X.mean(axis=0).tolist(),
        "covariance": covariance.tolist(),
        "vintage_note": "July 2008 archive; exact authors' downloaded vintage is unknown",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/simulation/ff3_200807.zip")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/ff3_calibration.json")
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = calibrate(args.archive, args.out, args.download)
    print(f"FF3 calibration: {result['n']} months, {args.out}")


if __name__ == "__main__":
    main()

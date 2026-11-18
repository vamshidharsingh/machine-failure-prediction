"""Load and validate the Scania Component X files.

Each split (train, validation, test) has three files:

- <split>_operational_readouts.csv: vehicle_id, time_step, then 105 columns
  from 14 anonymised variables (8 counters, 6 histograms with numbered bins)
- <split>_specifications.csv: vehicle_id, Spec_0 .. Spec_7 (categories)
- train_tte.csv (train) or <split>_labels.csv (validation, test)

CSVs are read once with float32 values, then cached as parquet in
<data_dir>/cache/ and loaded from there on later runs.
"""

from pathlib import Path

import pandas as pd

COUNTERS = ["171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]
HISTOGRAMS = {"167": 10, "272": 10, "291": 11, "158": 10, "459": 20, "397": 36}
HIST_COLS = {h: [f"{h}_{i}" for i in range(n)] for h, n in HISTOGRAMS.items()}
VALUE_COLS = COUNTERS + [c for cols in HIST_COLS.values() for c in cols]
SPEC_COLS = [f"Spec_{i}" for i in range(8)]
SPLITS = ("train", "validation", "test")
N_CLASSES = 5


class ScaniaDataError(ValueError):
    """Raised when a Scania file is missing or malformed."""


def _spec_path(data_dir: Path, split: str) -> Path:
    """The specifications file.

    Accepts small spelling differences, e.g. validation_specifications.csv or
    validation_specification.csv.
    """
    exact = data_dir / f"{split}_specifications.csv"
    if exact.exists():
        return exact
    found = sorted(data_dir.glob(f"{split}_spec*.csv"))
    return found[0] if found else exact


def file_paths(data_dir: Path, split: str) -> dict:
    """Paths of the three files of one split."""
    data_dir = Path(data_dir)
    third = (
        ("tte", data_dir / "train_tte.csv")
        if split == "train"
        else ("labels", data_dir / f"{split}_labels.csv")
    )
    return {
        "readouts": data_dir / f"{split}_operational_readouts.csv",
        "specs": _spec_path(data_dir, split),
        third[0]: third[1],
    }


def _dtypes(kind: str) -> dict:
    if kind == "readouts":
        return {"vehicle_id": "int64", "time_step": "float32", **{c: "float32" for c in VALUE_COLS}}
    if kind == "specs":
        return {"vehicle_id": "int64", **{c: "string" for c in SPEC_COLS}}
    if kind == "tte":
        return {
            "vehicle_id": "int64",
            "length_of_study_time_step": "float32",
            "in_study_repair": "int64",
        }
    return {"vehicle_id": "int64", "class_label": "int64"}


def _read(path: Path, kind: str, use_cache: bool) -> pd.DataFrame:
    if not path.exists():
        raise ScaniaDataError(
            f"Missing file: {path}. Download the Scania Component X dataset and place the "
            "nine CSV files in data/scania/ (see data/README.md)."
        )
    cache = path.parent / "cache" / f"{path.stem}.parquet"
    if use_cache and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache)
    dtypes = _dtypes(kind)
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in dtypes if c not in header]
    if missing:
        raise ScaniaDataError(
            f"{path.name}: missing columns {missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )
    df = pd.read_csv(path, dtype=dtypes, usecols=list(dtypes))
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def validate(tables: dict, split: str) -> None:
    """Integrity checks for one split. Raises ScaniaDataError on the first problem."""
    readouts = tables["readouts"]
    if readouts.empty:
        raise ScaniaDataError(f"{split}: no operational readouts")
    dup = int(readouts.duplicated(["vehicle_id", "time_step"]).sum())
    if dup:
        raise ScaniaDataError(
            f"{split}_operational_readouts: {dup} duplicate (vehicle_id, time_step) rows"
        )
    vehicles = set(readouts["vehicle_id"])
    for kind, df in tables.items():
        if kind == "readouts":
            continue
        if df["vehicle_id"].duplicated().any():
            raise ScaniaDataError(f"{split} {kind}: more than one row for some vehicle_id")
        other = set(df["vehicle_id"])
        if other != vehicles:
            raise ScaniaDataError(
                f"{split}: vehicle IDs differ between readouts and {kind} "
                f"({len(vehicles - other)} only in readouts, {len(other - vehicles)} only in "
                f"{kind})"
            )
    if "labels" in tables:
        bad = ~tables["labels"]["class_label"].between(0, N_CLASSES - 1)
        if bad.any():
            raise ScaniaDataError(
                f"{split}_labels: class_label outside 0-4 for {int(bad.sum())} vehicles"
            )
    if "tte" in tables:
        if not tables["tte"]["in_study_repair"].isin([0, 1]).all():
            raise ScaniaDataError("train_tte: in_study_repair must be 0 or 1")


def load_split(data_dir: Path, split: str, use_cache: bool = True) -> dict:
    """Return {"readouts", "specs", "tte" or "labels"} for one split, validated."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    tables = {
        kind: _read(path, kind, use_cache) for kind, path in file_paths(data_dir, split).items()
    }
    validate(tables, split)
    return tables

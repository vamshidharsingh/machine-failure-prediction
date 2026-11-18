"""Load and validate the NASA C-MAPSS files."""

from pathlib import Path

import pandas as pd

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"op{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = ID_COLS + SETTING_COLS + SENSOR_COLS


class DataError(ValueError):
    """Raised when the input files are missing or malformed."""


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(
            f"Missing file: {path}. Download the C-MAPSS data and place the "
            "train, test and RUL files in the data/ folder (see data/README.md)."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(ALL_COLS):
        raise DataError(f"{path.name}: expected {len(ALL_COLS)} columns, found {df.shape[1]}")
    df.columns = ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def validate(df: pd.DataFrame, name: str) -> None:
    """Basic integrity checks. Raises DataError on the first problem found."""
    if df.empty:
        raise DataError(f"{name}: no rows")
    if df[ALL_COLS].isna().any().any():
        raise DataError(f"{name}: contains missing values")
    if df.duplicated(["unit", "cycle"]).any():
        raise DataError(f"{name}: duplicate (unit, cycle) rows")
    for unit, g in df.groupby("unit"):
        cycles = g["cycle"].to_numpy()
        if cycles[0] != 1 or (pd.Series(cycles).diff().dropna() != 1).any():
            raise DataError(f"{name}: unit {unit} cycles are not consecutive from 1")


def load_cmapss(data_dir: Path, subset: str = "FD001"):
    """Return (train, test, test_rul) for one C-MAPSS subset.

    test_rul has one row per test engine: the true remaining useful life
    after that engine's last recorded cycle.
    """
    data_dir = Path(data_dir)
    train = _read_table(data_dir / f"train_{subset}.txt")
    test = _read_table(data_dir / f"test_{subset}.txt")

    rul_path = data_dir / f"RUL_{subset}.txt"
    if not rul_path.exists():
        raise DataError(f"Missing file: {rul_path}")
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].astype(int)

    validate(train, f"train_{subset}")
    validate(test, f"test_{subset}")

    test_units = sorted(test["unit"].unique())
    if len(rul) != len(test_units):
        raise DataError(f"RUL_{subset}: {len(rul)} values but {len(test_units)} test engines")
    test_rul = pd.DataFrame({"unit": test_units, "true_rul": rul.to_numpy()})
    return train, test, test_rul

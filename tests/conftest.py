"""Small synthetic data in the exact C-MAPSS file format.

Used only to test the code. Real results always come from the NASA files.
"""

import numpy as np
import pandas as pd
import pytest

from mfp.data import ALL_COLS


def make_engines(n_units: int, seed: int, truncate: bool = False):
    """Engines whose sensors drift as they approach failure."""
    rng = np.random.default_rng(seed)
    rows, true_rul = [], []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 260))
        stop = int(rng.integers(30, life - 5)) if truncate else life
        for cycle in range(1, stop + 1):
            wear = (cycle / life) ** 3
            ops = [rng.normal(0, 0.002), rng.normal(0, 0.0003), 100.0]
            sensors = []
            for s in range(1, 22):
                if s in (1, 5, 6, 10, 16, 18, 19):  # constant in FD001
                    sensors.append(float(s))
                else:
                    direction = 1 if s % 2 else -1
                    sensors.append(500 + s + direction * 8 * wear + rng.normal(0, 0.3))
            rows.append([unit, cycle, *ops, *sensors])
        true_rul.append(life - stop)
    df = pd.DataFrame(rows, columns=ALL_COLS)
    return df, true_rul


def write_cmapss(folder, n_train=30, n_test=12, seed=0):
    train, _ = make_engines(n_train, seed)
    test, rul = make_engines(n_test, seed + 1, truncate=True)
    fmt = ["%d", "%d"] + ["%.6f"] * (len(ALL_COLS) - 2)
    np.savetxt(folder / "train_FD001.txt", train.to_numpy(), fmt=fmt)
    np.savetxt(folder / "test_FD001.txt", test.to_numpy(), fmt=fmt)
    np.savetxt(folder / "RUL_FD001.txt", np.array(rul), fmt="%d")
    return folder


@pytest.fixture
def cmapss_dir(tmp_path):
    return write_cmapss(tmp_path)


# Six operating conditions (altitude, Mach, throttle), like FD002 and FD004.
REGIME_SETTINGS = [
    (0.0, 0.0, 100.0),
    (10.0, 0.25, 100.0),
    (20.0, 0.70, 100.0),
    (25.0, 0.62, 60.0),
    (35.0, 0.84, 100.0),
    (42.0, 0.84, 100.0),
]
CONSTANT_PER_REGIME = (1, 5, 6, 10, 16, 18, 19)


def make_multiregime_engines(n_units: int, seed: int, truncate: bool = False):
    """Engines that switch between six operating conditions every cycle.

    Each condition shifts every sensor by a large fixed offset (much larger
    than the wear signal), so raw readings mostly show the condition, not the
    wear. Some sensors are constant within each condition.
    """
    offsets = np.random.default_rng(123).uniform(-60, 60, (len(REGIME_SETTINGS), 21))
    rng = np.random.default_rng(seed)
    rows, true_rul = [], []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 260))
        stop = int(rng.integers(30, life - 5)) if truncate else life
        for cycle in range(1, stop + 1):
            wear = (cycle / life) ** 3
            r = int(rng.integers(len(REGIME_SETTINGS)))
            alt, mach, tra = REGIME_SETTINGS[r]
            ops = [alt + rng.normal(0, 0.002), mach + rng.normal(0, 0.0003), tra]
            sensors = []
            for s in range(1, 22):
                base = 500 + s + offsets[r, s - 1]
                if s in CONSTANT_PER_REGIME:
                    sensors.append(base)
                else:
                    direction = 1 if s % 2 else -1
                    sensors.append(base + direction * 8 * wear + rng.normal(0, 0.3))
            rows.append([unit, cycle, *ops, *sensors])
        true_rul.append(life - stop)
    return pd.DataFrame(rows, columns=ALL_COLS), true_rul


def write_multiregime(folder, subset="FD002", n_train=30, n_test=12, seed=0):
    train, _ = make_multiregime_engines(n_train, seed)
    test, rul = make_multiregime_engines(n_test, seed + 1, truncate=True)
    fmt = ["%d", "%d"] + ["%.6f"] * (len(ALL_COLS) - 2)
    np.savetxt(folder / f"train_{subset}.txt", train.to_numpy(), fmt=fmt)
    np.savetxt(folder / f"test_{subset}.txt", test.to_numpy(), fmt=fmt)
    np.savetxt(folder / f"RUL_{subset}.txt", np.array(rul), fmt="%d")
    return folder


@pytest.fixture
def multiregime_dir(tmp_path):
    return write_multiregime(tmp_path)


def _scania_vehicle(rng, vid, n_steps, life, specs):
    """Readouts for one vehicle up to time step n_steps (failure at `life`).

    Counters and histograms are cumulative. As failure nears, the counters
    climb faster and the histogram mass moves to higher bins.
    """
    from mfp.scania.data import COUNTERS, HIST_COLS

    times = np.cumsum(rng.uniform(2.0, 6.0, size=400))
    times = np.round(times[times <= n_steps], 1)
    if len(times) == 0:
        times = np.array([round(min(n_steps, 1.0), 1)])
    rows = []
    counters = np.zeros(len(COUNTERS))
    hists = {h: np.zeros(len(cols)) for h, cols in HIST_COLS.items()}
    prev = 0.0
    for t in times:
        dt = t - prev
        prev = t
        wear = max(0.0, 1.0 - (life - t) / 60.0)  # 0 until 60 steps before failure
        counters += dt * (1.0 + 4.0 * wear) * rng.uniform(0.8, 1.2, len(COUNTERS))
        row = {"vehicle_id": vid, "time_step": t}
        row.update(dict(zip(COUNTERS, counters.copy(), strict=True)))
        for h, cols in HIST_COLS.items():
            centre = (len(cols) - 1) * (0.2 + 0.7 * wear)
            weights = np.exp(-0.5 * ((np.arange(len(cols)) - centre) / 1.5) ** 2)
            hists[h] += dt * 10 * weights / weights.sum()
            row.update(dict(zip(cols, hists[h].copy(), strict=True)))
        rows.append(row)
    df = pd.DataFrame(rows)
    # A few missing histogram readouts, like the real data (whole histogram).
    miss = rng.random(len(df)) < 0.03
    df.loc[miss, HIST_COLS["291"]] = np.nan
    return df


def write_scania(folder, n_train=80, n_val=40, n_test=40, seed=0, unseen_test_category=True):
    """Nine CSVs named like the real dataset. Returns the folder."""
    from mfp.scania.data import SPEC_COLS

    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)

    def specs_for(vid, allow_unseen):
        cats = [f"Cat{int(rng.integers(0, 3))}" for _ in SPEC_COLS]
        if allow_unseen and vid % 7 == 0:
            cats[1] = "Cat28"  # never in training
        return {"vehicle_id": vid, **dict(zip(SPEC_COLS, cats, strict=True))}

    readouts, specs, tte = [], [], []
    for vid in range(n_train):
        repaired = vid % 2 == 0
        life = float(rng.uniform(120, 220))
        study = life if repaired else float(rng.uniform(150, 260))
        readouts.append(_scania_vehicle(rng, vid, study, life if repaired else study + 200, None))
        specs.append(specs_for(vid, False))
        tte.append(
            {
                "vehicle_id": vid,
                "length_of_study_time_step": round(study, 1),
                "in_study_repair": int(repaired),
            }
        )
    pd.concat(readouts).to_csv(folder / "train_operational_readouts.csv", index=False)
    pd.DataFrame(specs).to_csv(folder / "train_specifications.csv", index=False)
    pd.DataFrame(tte).to_csv(folder / "train_tte.csv", index=False)

    def eval_split(name, n, offset, allow_unseen):
        from mfp.scania.labels import time_left_to_class

        readouts, specs, labels = [], [], []
        for i in range(n):
            vid = offset + i
            life = float(rng.uniform(120, 220))
            # Half the vehicles stop close to failure, half long before it.
            left = float(rng.uniform(0, 48)) if i % 2 == 0 else float(rng.uniform(60, 100))
            df = _scania_vehicle(rng, vid, life - left, life, None)
            readouts.append(df)
            specs.append(specs_for(vid, allow_unseen))
            labels.append(
                {
                    "vehicle_id": vid,
                    "class_label": int(time_left_to_class(life - df["time_step"].iloc[-1])),
                }
            )
        pd.concat(readouts).to_csv(folder / f"{name}_operational_readouts.csv", index=False)
        pd.DataFrame(specs).to_csv(folder / f"{name}_specifications.csv", index=False)
        pd.DataFrame(labels).to_csv(folder / f"{name}_labels.csv", index=False)

    eval_split("validation", n_val, 10_000, False)
    eval_split("test", n_test, 20_000, unseen_test_category)
    return folder


@pytest.fixture
def scania_dir(tmp_path):
    return write_scania(tmp_path / "scania")
